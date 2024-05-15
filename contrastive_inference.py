from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from transforms import INFERENCE_TRANSFORM
from data import Padding, inference_datapipe
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
        pretrained=True,  # overwritten by loading weights
        loss=None,
        n_epochs=0,
        use_head=args.head,
    )
    model.backbone.load_state_dict(
        torch.load(f"model_weights/{args.name}_backbone.pth")
    )
    model.projection_head.load_state_dict(
        torch.load(f"model_weights/{args.name}_head.pth")
    )

    # transforms and dataloaders
    datapipe_train = inference_datapipe(
        [
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        ],
        num_images=NUM_TRAIN,
        transforms=INFERENCE_TRANSFORM,
        padding=Padding.REFLECT,
    )
    datapipe_test = inference_datapipe(
        [
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
        ],
        num_images=NUM_TEST,
        transforms=INFERENCE_TRANSFORM,
        padding=Padding.REFLECT,
    )
    dataloader_train = DataLoader(
        datapipe_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.devices * 4,
    )
    dataloader_test = DataLoader(
        datapipe_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.devices * 4,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp",
    )
    out1 = trainer.predict(model, dataloader_train)
    out2 = trainer.predict(model, dataloader_test)

    output = np.concatenate(
        (
            torch.cat([o[0] for o in out1]).cpu().numpy(),
            torch.cat([o[0] for o in out2]).cpu().numpy(),
        )
    )
    labels = np.concatenate(
        (
            torch.cat([o[1] for o in out1]).cpu().numpy(),
            torch.cat([o[1] for o in out2]).cpu().numpy(),
        )
    )

    np.save(f"embeddings/output_{args.name}{'_head' if args.head else ''}.npy", output)
    np.save(f"embeddings/labels_{args.name}{'_head' if args.head else ''}.npy", labels)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="finetune_resnet18")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    main(args)
