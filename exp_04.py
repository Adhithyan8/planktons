"""
exp 03 - REFLECT padding improves performance on Planktons, but not others.
But we continue using it for all datasets for simplicity.

How much does contrastive fine tuning help?
"""

from time import time

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from lightly.loss import NTXentLoss
from PIL import Image
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *

# continuing with this precision setting
torch.set_float32_matmul_precision("medium")


# nearly a one-to-one copy from lightly examples
class SimCLR(L.LightningModule):
    def __init__(self, backbone, projection):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projection = projection
        self.criterion = NTXentLoss(
            gather_distributed=True
        )  # TODO: combine with supervised loss

    def forward(self, x):  # TODO: is it better to use projection head for prediction?
        x = self.backbone(x)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        z0 = self(x0)
        z1 = self(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4
        )  # TODO: try increasing lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=0
        )
        return [optimizer], [scheduler]


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
            A.GaussianBlur(),
            A.Normalize(),
        ]
    )  # TODO: try color / shiftscalerotate
    # apply twice for two views
    img0 = augs(image=img)["image"]
    img1 = augs(image=img)["image"]
    img0 = torch.from_numpy(img0).permute(2, 0, 1)
    img1 = torch.from_numpy(img1).permute(2, 0, 1)
    return (img0, img1), label


# datasets to train on
datasets = [
    "CUB",
    "SCARS",
    "AIRCRAFT",
    "HERB19",
    "PLANKTON",
]

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    # lets use ViT DINOv2 as backbone
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")

    # freeze all blocks except the last one
    for param in backbone.parameters():
        param.requires_grad_(False)
    for name, param in backbone.named_parameters():
        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num >= 11:
                param.requires_grad_(True)

    # 3 layer MLP projection head with ReLU and BN
    projection = torch.nn.Sequential(
        torch.nn.Linear(768, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 256),
    )  # TODO: do we need BN?

    model = SimCLR(
        backbone, projection
    )  # TODO: check the eRANK of features after inference

    if dataset == "CUB":
        info = CUB_INFO
    elif dataset == "SCARS":
        info = SCARS_INFO
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
        info, split_fit="train", split_cat="new", transform=data_transform
    )
    tst_old_dataset = make_dataset(
        info, split_fit="test", split_cat="old", transform=data_transform
    )
    tst_new_dataset = make_dataset(
        info, split_fit="test", split_cat="new", transform=data_transform
    )

    # merge all 4 datasets for contrastive fine tuning
    combined_dataset = torch.utils.data.ConcatDataset(
        [trn_old_dataset, trn_new_dataset, tst_old_dataset, tst_new_dataset]
    )

    def train_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    dl = train_dataloader(combined_dataset, 128)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        max_epochs=200,
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
    )

    t = time()
    trainer.fit(model, dl)
    print(f"Training time for {dataset}: {time()-t}")

    # save the model
    torch.save(model.state_dict(), f"outputs/exp_04_{dataset}.pt")
    print(f"Model saved for {dataset}")

    # delete the model, backbone and projection head - cuz I am paranoid
    del model
    del backbone
    del projection
