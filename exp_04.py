"""
How much does contrastive fine tuning help?
"""

from time import time

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from lightly.loss import NTXentLoss
from PIL import Image
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities import grad_norm
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *
from losses import CombinedLoss, NTXentLossSupervised

torch.set_float32_matmul_precision("high")

# nearly a one-to-one copy from lightly examples
loss1 = NTXentLoss(temperature=1.0)
loss2 = NTXentLossSupervised(temperature=0.07)
loss_fn = CombinedLoss(loss1, loss2)


class SimCLR(L.LightningModule):
    def __init__(self):
        super(SimCLR, self).__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        # freeze all blocks except the last one
        for param in backbone.parameters():
            param.requires_grad_(False)
        for name, param in backbone.named_parameters():
            if "block" in name:
                block_num = int(name.split(".")[1])
                if block_num >= 11:
                    param.requires_grad_(True)

        projection = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 256),
        )
        self.backbone = backbone
        self.projection = projection
        self.criterion = loss_fn

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        (x0, x1), lbl = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1, lbl)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            decoupled_weight_decay=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0
        )
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(module=self, norm_type=2)  # returns dict
        self.log(
            "grad_norm", norms["grad_2.0_norm_total"], prog_bar=True
        )  # TODO: add clip if needed


# lets define the transforms for SimCLR training
def data_transform(img, label):
    # work around to maintain aspect ratio with albumentations
    with Image.open(img) as img:
        img = np.array(img)
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = A.LongestMaxSize(max_size=256)(image=img)["image"]
    img = A.PadIfNeeded(img.shape[1], img.shape[0], border_mode=4)(image=img)["image"]
    img = A.Resize(256, 256)(image=img)["image"]
    # if grayscale, convert to 3 channels
    if len(img.shape) == 2:
        img = A.ToRGB()(image=img)["image"]
    # stochastic augmentations
    augs = A.Compose(
        [
            A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
            A.HorizontalFlip(),
            A.ColorJitter(),
            A.GaussianBlur(),
            A.Normalize(),
        ]
    )
    # apply twice for two views
    img0 = augs(image=img)["image"]
    img1 = augs(image=img)["image"]
    img0 = torch.from_numpy(img0).permute(2, 0, 1)
    img1 = torch.from_numpy(img1).permute(2, 0, 1)
    label = torch.tensor(label, dtype=torch.long)
    return (img0, img1), label


def mask_label_transform(img, label):
    (img0, img1), label = data_transform(img, label)
    # set label to -1 to mask it
    label = -1 * torch.ones_like(label)
    return (img0, img1), label


# datasets to train on
datasets = [
    "CUB",
]
trials = 3
# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    for trial in range(trials):
        print(f"Trial {trial+1} for {dataset}")
        model = SimCLR()

        if dataset == "CUB":
            info = CUB_INFO
        elif dataset == "SCARS":
            info = SCARS_INFO
            # shift labels to start from 0
            for sample in info:
                sample["label"] -= 1
        elif dataset == "AIRCRAFT":
            info = AIRCRAFT_INFO
        elif dataset == "HERB19":
            info = HERB19_INFO
        elif dataset == "PLANKTON":
            info = PLANKTON_INFO

        trn_old_dataset = make_dataset(
            info, split_fit="train", split_cat="old", transform=data_transform
        )
        trn_new_dataset = make_dataset(
            info, split_fit="train", split_cat="new", transform=mask_label_transform
        )
        tst_old_dataset = make_dataset(
            info, split_fit="test", split_cat="old", transform=mask_label_transform
        )
        tst_new_dataset = make_dataset(
            info, split_fit="test", split_cat="new", transform=mask_label_transform
        )

        # merge all 4 datasets for contrastive fine tuning
        combined_dataset = torch.utils.data.ConcatDataset(
            [trn_old_dataset, trn_new_dataset, tst_old_dataset, tst_new_dataset]
        )

        def train_dataloader(dataset, batch_size):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                drop_last=True,
            )

        dl = train_dataloader(combined_dataset, 128)

        trainer = L.Trainer(
            accelerator="gpu",
            devices=4,
            num_nodes=1,
            max_epochs=100,
            strategy="ddp",
            sync_batchnorm=True,
            use_distributed_sampler=True,
        )

        t = time()
        trainer.fit(model, dl)
        print(f"Training time for {dataset}: {time()-t}")

        # save the model
        torch.save(model.state_dict(), f"outputs/exp_04_{dataset}_trial_{trial}.pt")
        print(f"Model saved for {dataset} trial {trial}")

        # delete the model, backbone and projection head - cuz I am paranoid
        del model
