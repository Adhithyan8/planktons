from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from time import time

import numpy as np
import pytorch_lightning as L
import torch

from data import MuCUBDataModule, MuPlanktonDataModule, Padding, make_data
from model import LightningMuContrastive
from transforms import (
    CUB_MU_INFERENCE,
    CUB_MU_STUDENT,
    CUB_MU_TEACHER,
    INFERENCE_TRANSFORM,
    WHOI_STUDENT,
    WHOI_TEACHER,
)

torch.set_float32_matmul_precision("high")


def main(args):
    model = LightningMuContrastive(
        args.name,
        out_dim=args.out_dim,
        loss=None,
        n_epochs=0,
        uuid=args.uuid,
        arch="vit",
    )
    model.teacher_backbone.load_state_dict(
        torch.load(f"model_weights/{args.name}_tb.pth")
    )
    model.teacher_head.load_state_dict(torch.load(f"model_weights/{args.name}_th.pth"))
    model.student_backbone.load_state_dict(
        torch.load(f"model_weights/{args.name}_sb.pth")
    )
    model.student_head.load_state_dict(torch.load(f"model_weights/{args.name}_sh.pth"))

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
        trn_dataset = MuPlanktonDataModule(
            trn_data,
            WHOI_TEACHER,
            WHOI_STUDENT,
            INFERENCE_TRANSFORM,
            batch_size=args.batch_size,
        )
        tst_dataset = MuPlanktonDataModule(
            tst_data,
            WHOI_TEACHER,
            WHOI_STUDENT,
            INFERENCE_TRANSFORM,
            batch_size=args.batch_size,
        )
    elif args.data == "cub":
        trn_dataset = MuCUBDataModule(
            "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
            CUB_MU_TEACHER,
            CUB_MU_STUDENT,
            CUB_MU_INFERENCE,
            batch_size=args.batch_size,
            uuid=args.uuid,
            split="train",
        )
        tst_dataset = MuCUBDataModule(
            "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
            CUB_MU_TEACHER,
            CUB_MU_STUDENT,
            CUB_MU_INFERENCE,
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
        np.save(f"embeddings/fnames_{args.name}.npy", fnames)

    np.save(f"embeddings/output_{args.name}.npy", output)
    np.save(f"embeddings/labels_{args.name}.npy", labels)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="selfsupcauchy_resnet18")
    parser.add_argument("--data", default="cub")
    parser.add_argument("--out-dim", type=int, default=230)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--uuid", action="store_true")
    args = parser.parse_args()

    start = time()
    main(args)
    print(f"Time taken: {time() - start}")
