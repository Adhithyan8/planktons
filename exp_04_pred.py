"""
We did contrastive finetuning on all datasets but PLANKTON, which failed due to time limit
Let's predict the embeddings for the trained nets

Later we will evaluate them using kNN and k-means clustering
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

    # adding predict method to get embeddings
    def predict_step(self, batch, batch_idx):
        x, label = batch
        z = self.backbone(x)  # TODO: check if we need projection head
        return z, label


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
    img = A.CenterCrop(224, 224)(image=img)["image"]
    img = A.Normalize()(image=img)["image"]
    img = torch.tensor(img).permute(2, 0, 1)
    return img, label


# datasets to predict on
datasets = [
    "CUB",
    "SCARS",
    "AIRCRAFT",
    "HERB19",
]  # TODO: Add PLANKTON after training

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

    model = SimCLR(backbone, projection)

    # load the trained model
    model.load_state_dict(torch.load(f"outputs/exp_04_{dataset}.pt"))

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

    def predict_dataloader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    trn_old_dl = predict_dataloader(trn_old_dataset, 128)
    trn_new_dl = predict_dataloader(trn_new_dataset, 128)
    tst_old_dl = predict_dataloader(tst_old_dataset, 128)
    tst_new_dl = predict_dataloader(tst_new_dataset, 128)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
    )

    trn_old_outs = trainer.predict(model, trn_old_dl)
    trn_new_outs = trainer.predict(model, trn_new_dl)
    tst_old_outs = trainer.predict(model, tst_old_dl)
    tst_new_outs = trainer.predict(model, tst_new_dl)

    # concatenate the embeddings
    trn_old_out = torch.cat([out[0] for out in trn_old_outs]).cpu().numpy()
    trn_new_out = torch.cat([out[0] for out in trn_new_outs]).cpu().numpy()
    tst_old_out = torch.cat([out[0] for out in tst_old_outs]).cpu().numpy()
    tst_new_out = torch.cat([out[0] for out in tst_new_outs]).cpu().numpy()

    # labels
    trn_old_lbl = torch.cat([out[1] for out in trn_old_outs]).cpu().numpy()
    trn_new_lbl = torch.cat([out[1] for out in trn_new_outs]).cpu().numpy()
    tst_old_lbl = torch.cat([out[1] for out in tst_old_outs]).cpu().numpy()
    tst_new_lbl = torch.cat([out[1] for out in tst_new_outs]).cpu().numpy()

    # save everything
    np.save(f"outputs/SimCLRUnsup_{dataset}_trn_old_out.npy", trn_old_out)
    np.save(f"outputs/SimCLRUnsup_{dataset}_trn_new_out.npy", trn_new_out)
    np.save(f"outputs/SimCLRUnsup_{dataset}_tst_old_out.npy", tst_old_out)
    np.save(f"outputs/SimCLRUnsup_{dataset}_tst_new_out.npy", tst_new_out)
    np.save(f"outputs/SimCLRUnsup_{dataset}_trn_old_lbl.npy", trn_old_lbl)
    np.save(f"outputs/SimCLRUnsup_{dataset}_trn_new_lbl.npy", trn_new_lbl)
    np.save(f"outputs/SimCLRUnsup_{dataset}_tst_old_lbl.npy", tst_old_lbl)
    np.save(f"outputs/SimCLRUnsup_{dataset}_tst_new_lbl.npy", tst_new_lbl)
    print(f"Saved embeddings for {dataset}")
