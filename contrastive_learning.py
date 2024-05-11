from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L

from config import CONTRASTIVE_TRANSFORM
from losses import InfoNCECosineSelfSupervised
from data import Padding, contrastive_datapipe
from model import LightningContrastive

torch.set_float32_matmul_precision("high")

"""
Dataset sizes:
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676
NUM_TOTAL = NUM_TRAIN + NUM_TEST


def main(args):
    model = LightningContrastive(
        head_dim=args.head_dim,
        pretrained=args.pretrained,
        loss=InfoNCECosineSelfSupervised(),
        n_epochs=args.epochs,
    )

    # transforms and dataloaders
    datapipe = contrastive_datapipe(
        [
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
        ],
        num_images=NUM_TOTAL,
        transforms=CONTRASTIVE_TRANSFORM,
        padding=Padding.REFLECT,
        ignore_mix=True,
        mask_label=True,
    )

    dataloader = DataLoader(
        datapipe,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.devices * 4,
        persistent_workers=True,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp",
    )
    trainer.fit(model, dataloader)

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
