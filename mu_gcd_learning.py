from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from time import time

import pytorch_lightning as L
import torch

from data import MuCUBDataModule
from losses import DistillLoss
from model import LightningMuContrastive
from transforms import CUB_MU_INFERENCE, CUB_MU_STUDENT, CUB_MU_TEACHER

torch.set_float32_matmul_precision("high")


def main(args):
    model = LightningMuContrastive(
        args.name,
        out_dim=230,
        loss=DistillLoss(30, args.epochs),
        n_epochs=args.epochs,
        arch="vit",
    )

    dataset = MuCUBDataModule(
        "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
        CUB_MU_TEACHER,
        CUB_MU_STUDENT,
        CUB_MU_INFERENCE,
        batch_size=args.batch_size,
    )

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
    torch.save(model.teacher_backbone.state_dict(), f"model_weights/{args.name}_tb.pth")
    torch.save(model.teacher_head.state_dict(), f"model_weights/{args.name}_th.pth")
    torch.save(model.student_backbone.state_dict(), f"model_weights/{args.name}_sb.pth")
    torch.save(model.student_head.state_dict(), f"model_weights/{args.name}_sh.pth")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="selfsupcauchy_resnet18")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    start = time()
    main(args)
    print(f"Time taken: {time() - start}")
    