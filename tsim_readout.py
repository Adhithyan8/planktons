from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from time import time

import pytorch_lightning as L
import torch

from data import Padding, PlanktonDataModule, make_data
from losses import CombinedLoss, InfoNCECauchySelfSupervised, InfoNCECauchySupervised
from model import LightningTsimnce
from transforms import CONTRASTIVE_TRANSFORM, INFERENCE_TRANSFORM

torch.set_float32_matmul_precision("high")


def main(args):
    loss1 = InfoNCECauchySelfSupervised()
    loss2 = InfoNCECauchySupervised()

    model = LightningTsimnce(
        name=args.name,
        old_head_dim=args.old_head_dim,
        new_head_dim=args.new_head_dim,
        loss=CombinedLoss(loss1, loss2, 0.35),
        n_epochs=args.readout_epochs,
        phase="readout",
    )

    data = make_data(
        [
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
        ],
        Padding.REFLECT,
        ignore_mix=True,
    )

    dataset = PlanktonDataModule(
        data,
        CONTRASTIVE_TRANSFORM,
        INFERENCE_TRANSFORM,
        batch_size=args.batch_size,
    )

    trainer = L.Trainer(
        max_epochs=args.readout_epochs,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp",
    )
    trainer.fit(model, dataset)

    # save the model
    torch.save(model.backbone.state_dict(), f"model_weights/read_{args.name}_bb.pth")
    torch.save(
        model.projection_head.state_dict(), f"model_weights/read_{args.name}_ph.pth"
    )


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="selfcauchy")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--readout-epochs", type=int, default=50)
    parser.add_argument("--old-head-dim", type=int, default=128)
    parser.add_argument("--new-head-dim", type=int, default=2)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    start = time()
    main(args)
    print(f"Time taken: {time() - start}")
