"""
exp 01 results resonably match what is reported in the literature.

Lets change the pretrained model, as Sagar (2023) report massive gains from using DINOv2 trained nets.
We use the one with registers which is a small upgrade.
"""

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *

# continuing with this precision setting
torch.set_float32_matmul_precision("medium")

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")


# wrap in lightning module
class ViT(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        return out, label


ViT_model = ViT(model)


# lets define the transforms for the forward pass
def data_transform(img, label):
    # work around to maintain aspect ratio with albumentations
    with Image.open(img) as img:
        img = np.array(img)
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = A.LongestMaxSize(max_size=256)(image=img)["image"]
    img = A.PadIfNeeded(img.shape[1], img.shape[0], border_mode=0, value=0)(image=img)[
        "image"
    ]  # TODO: try mode 4
    img = A.Resize(256, 256)(image=img)["image"]
    img = A.CenterCrop(224, 224)(image=img)["image"]
    # if grayscale, convert to 3 channels
    if len(img.shape) == 2:
        img = A.ToRGB()(image=img)["image"]
    img = A.Normalize()(image=img)["image"]
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img, label


# datasets to evaluate on
datasets = [
    "CUB",
    "SCARS",
    "AIRCRAFT",
    "HERB19",
    "PLANKTON",
]

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
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

    trn_old_outs = trainer.predict(ViT_model, trn_old_dl)
    trn_new_outs = trainer.predict(ViT_model, trn_new_dl)
    tst_old_outs = trainer.predict(ViT_model, tst_old_dl)
    tst_new_outs = trainer.predict(ViT_model, tst_new_dl)

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
    np.save(f"outputs/ViT_DINO_{dataset}_trn_old_out.npy", trn_old_out)
    np.save(f"outputs/ViT_DINO_{dataset}_trn_new_out.npy", trn_new_out)
    np.save(f"outputs/ViT_DINO_{dataset}_tst_old_out.npy", tst_old_out)
    np.save(f"outputs/ViT_DINO_{dataset}_tst_new_out.npy", tst_new_out)
    np.save(f"outputs/ViT_DINO_{dataset}_trn_old_lbl.npy", trn_old_lbl)
    np.save(f"outputs/ViT_DINO_{dataset}_trn_new_lbl.npy", trn_new_lbl)
    np.save(f"outputs/ViT_DINO_{dataset}_tst_old_lbl.npy", tst_old_lbl)
    np.save(f"outputs/ViT_DINO_{dataset}_tst_new_lbl.npy", tst_new_lbl)
    print(f"Saved embeddings for {dataset}")
