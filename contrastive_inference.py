from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pytorch_lightning as L
import torch

from data import CUBDataModule, Padding, PlanktonDataModule, make_data
from model import LightningContrastive
from transforms import (
    CONTRASTIVE_TRANSFORM,
    CUB_CONTRASTIVE,
    CUB_INFERENCE,
    INFERENCE_TRANSFORM,
)

torch.set_float32_matmul_precision("high")


def main(args):
    model = LightningContrastive(
        head_dim=args.head_dim,
        pretrained=True,
        loss=None,
        n_epochs=0,
        use_head=args.head,
        uuid=args.uuid,
        arch="vit",
    )
    model.backbone.load_state_dict(torch.load(f"model_weights/{args.name}_bb.pth"))
    model.projection_head.load_state_dict(
        torch.load(f"model_weights/{args.name}_ph.pth")
    )

    if args.data == "whoi_plankton":
        trn_data = make_data(
            [
                "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
            ],
            Padding.REFLECT,
            ignore_mix=True,
            mask=False,
            uuid=args.uuid,
        )
        tst_data = make_data(
            [
                "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
            ],
            Padding.REFLECT,
            ignore_mix=True,
            mask=False,
            uuid=args.uuid,
        )
        trn_dataset = PlanktonDataModule(
            trn_data,
            CONTRASTIVE_TRANSFORM,
            INFERENCE_TRANSFORM,
            batch_size=args.batch_size,
            uuid=args.uuid,
        )
        tst_dataset = PlanktonDataModule(
            tst_data,
            CONTRASTIVE_TRANSFORM,
            INFERENCE_TRANSFORM,
            batch_size=args.batch_size,
            uuid=args.uuid,
        )
    elif args.data == "cub":
        trn_dataset = CUBDataModule(
            "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
            CUB_CONTRASTIVE,
            CUB_INFERENCE,
            batch_size=args.batch_size,
            uuid=args.uuid,
            split="train",
        )
        tst_dataset = CUBDataModule(
            "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
            CUB_CONTRASTIVE,
            CUB_INFERENCE,
            batch_size=args.batch_size,
            uuid=args.uuid,
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
    if args.uuid:
        fnames = np.concatenate(
            (
                np.concatenate([np.array([*out[2]]) for out in out_trn]),
                np.concatenate([np.array([*out[2]]) for out in out_tst]),
            )
        )
        np.save(
            f"embeddings/fnames_{args.name}{'_ph' if args.head else ''}.npy", fnames
        )

    np.save(f"embeddings/output_{args.name}{'_ph' if args.head else ''}.npy", output)
    np.save(f"embeddings/labels_{args.name}{'_ph' if args.head else ''}.npy", labels)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="resnet18")
    parser.add_argument("--data", default="whoi_plankton")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--uuid", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    main(args)
