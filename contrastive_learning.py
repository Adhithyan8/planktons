from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from data import Padding, PlanktonDataModule
from losses import InfoNCECosineSelfSupervised
from model import LightningContrastive
from transforms import CONTRASTIVE_TRANSFORM, INFERENCE_TRANSFORM

torch.set_float32_matmul_precision("high")


def main(args):
    model = LightningContrastive(
        head_dim=args.head_dim,
        pretrained=args.pretrained,
        loss=InfoNCECosineSelfSupervised(),
        n_epochs=args.epochs,
    )

    dataset = PlanktonDataModule(
        [
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
        ],
        CONTRASTIVE_TRANSFORM,
        INFERENCE_TRANSFORM,
        Padding.REFLECT,
        ignore_mix=True,
        batch_size=args.batch_size,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp",
    )
    trainer.fit(model, dataset)

    # save the model
    torch.save(model.backbone.state_dict(), f"{args.name}_backbone.pth")
    torch.save(model.projection_head.state_dict(), f"{args.name}_head.pth")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="selfsupcauchy_resnet18")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    main(args)
