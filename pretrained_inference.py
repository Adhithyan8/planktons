from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pytorch_lightning as L
import torch

from data import CUBDataModule, Padding, PlanktonDataModule, make_data
from model import LightningPretrained
from transforms import (
    CUB_CONTRASTIVE,
    CUB_INFERENCE,
    INFER_VIT_TRANSFORM,
    INFERENCE_TRANSFORM,
)

torch.set_float32_matmul_precision("high")


def main(args):
    model = LightningPretrained(args.name)

    if args.data == "whoi_plankton":
        trn_data = make_data(
            [
                "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
            ],
            Padding.REFLECT,
            ignore_mix=True,
            mask=False,
        )
        tst_data = make_data(
            [
                "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
            ],
            Padding.REFLECT,
            ignore_mix=True,
            mask=False,
        )
        if args.name == "vitb14-dinov2":
            trn_dataset = PlanktonDataModule(
                trn_data,
                INFER_VIT_TRANSFORM,
                INFER_VIT_TRANSFORM,
                batch_size=args.batch_size,
            )
            tst_dataset = PlanktonDataModule(
                tst_data,
                INFER_VIT_TRANSFORM,
                INFER_VIT_TRANSFORM,
                batch_size=args.batch_size,
            )
        else:
            trn_dataset = PlanktonDataModule(
                trn_data,
                INFERENCE_TRANSFORM,
                INFERENCE_TRANSFORM,
                batch_size=args.batch_size,
            )
            tst_dataset = PlanktonDataModule(
                tst_data,
                INFERENCE_TRANSFORM,
                INFERENCE_TRANSFORM,
                batch_size=args.batch_size,
            )
    elif args.data == "cub":
        trn_dataset = CUBDataModule(
            "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
            CUB_CONTRASTIVE,
            CUB_INFERENCE,
            batch_size=args.batch_size,
            uuid=False,
            split="train",
        )
        tst_dataset = CUBDataModule(
            "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
            CUB_CONTRASTIVE,
            CUB_INFERENCE,
            batch_size=args.batch_size,
            uuid=False,
            split="test",
        )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp",
    )

    out_trn = trainer.predict(model, trn_dataset)
    out_tst = trainer.predict(model, tst_dataset)

    output = np.concatenate(
        (
            torch.cat([out[0] for out in out_trn]).cpu().numpy(),
            torch.cat([out[0] for out in out_tst]).cpu().numpy(),
        )
    )
    labels = np.concatenate(
        (
            torch.cat([out[1] for out in out_trn]).cpu().numpy(),
            torch.cat([out[1] for out in out_tst]).cpu().numpy(),
        )
    )

    np.save(f"embeddings/output_{args.name}.npy", output)
    np.save(f"embeddings/labels_{args.name}.npy", labels)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="resnet18")
    parser.add_argument("--data", default="whoi_plankton")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    main(args)
