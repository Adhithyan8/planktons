from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from time import time

import pytorch_lightning as L
import torch

from data import CUBDataModule, Padding, PlanktonDataModule, make_data
from losses import (
    CombinedLoss,
    InfoNCECauchySelfSupervised,
    InfoNCECauchySupervised,
    InfoNCECosineSelfSupervised,
    InfoNCECosineSupervised,
)
from model import LightningContrastive
from transforms import (
    CONTRASTIVE_TRANSFORM,
    CUB_CONTRASTIVE,
    CUB_INFERENCE,
    INFERENCE_TRANSFORM,
)

torch.set_float32_matmul_precision("high")


def main(args):
    loss1 = InfoNCECauchySelfSupervised(temperature=1.0) # TODO: use lightly NT-Xent loss
    loss2 = InfoNCECauchySupervised(temperature=0.07)    # TODO: use NTXentLossSupervised

    model = LightningContrastive(                        # TODO: replace with lightly's version
        head_dim=args.head_dim,
        pretrained=args.pretrained,
        loss=CombinedLoss(loss1, loss2, 0.35),
        n_epochs=args.epochs,
        arch="vit",
    )

    if args.data == "whoi_plankton":
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
    elif args.data == "cub":
        dataset = CUBDataModule(
            "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
            CUB_CONTRASTIVE,
            CUB_INFERENCE,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError("Invalid dataset")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
    )
    trainer.fit(model, dataset)

    # save the model
    torch.save(model.backbone.state_dict(), f"model_weights/{args.name}_bb.pth")
    torch.save(model.projection_head.state_dict(), f"model_weights/{args.name}_ph.pth")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="selfsupcauchy_resnet18")
    parser.add_argument("--data", default="whoi_plankton")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    start = time()
    main(args)
    print(f"Time taken: {time() - start}")
