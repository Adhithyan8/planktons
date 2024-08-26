"""
Predicting SimCLR embeddings
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
torch.set_float32_matmul_precision("high")


# nearly a one-to-one copy from lightly examples
class SimCLR(L.LightningModule):
    def __init__(self):
        super(SimCLR, self).__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        projection = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 256),
        )
        self.backbone = backbone
        self.projection = projection
        self.criterion = None

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x

    # adding predict method to get embeddings
    def predict_step(self, batch, batch_idx):
        x, label = batch
        z = self.backbone(x)
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
    label = torch.tensor(label, dtype=torch.long)
    return img, label


# datasets to predict on
datasets = [
    "AIRCRAFT",
]
trials = 3
# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    for trial in range(trials):
        print(f"Predicting for {dataset} trial {trial}")
        model = SimCLR()

        # load the trained model
        model.load_state_dict(torch.load(f"outputs/exp_04_{dataset}_trial_{trial}.pt"))

        # check for NaNs in model backbone weights
        for name, param in model.backbone.named_parameters():
            if torch.isnan(param).any():
                print(f"NaNs in {name} weights!")
        # check for NaNs in model projection weights
        for name, param in model.projection.named_parameters():
            if torch.isnan(param).any():
                print(f"NaNs in {name} weights!")

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
            info, split_fit="train", split_cat="new", transform=data_transform
        )
        tst_old_dataset = make_dataset(
            info, split_fit="test", split_cat="old", transform=data_transform
        )
        tst_new_dataset = make_dataset(
            info, split_fit="test", split_cat="new", transform=data_transform
        )

        def predict_dataloader(dataset, batch_size):
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=8
            )

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
        np.save(f"outputs/SimCLR_{dataset}_trn_old_out_{trial}.npy", trn_old_out)
        np.save(f"outputs/SimCLR_{dataset}_trn_new_out_{trial}.npy", trn_new_out)
        np.save(f"outputs/SimCLR_{dataset}_tst_old_out_{trial}.npy", tst_old_out)
        np.save(f"outputs/SimCLR_{dataset}_tst_new_out_{trial}.npy", tst_new_out)
        np.save(f"outputs/SimCLR_{dataset}_trn_old_lbl_{trial}.npy", trn_old_lbl)
        np.save(f"outputs/SimCLR_{dataset}_trn_new_lbl_{trial}.npy", trn_new_lbl)
        np.save(f"outputs/SimCLR_{dataset}_tst_old_lbl_{trial}.npy", tst_old_lbl)
        np.save(f"outputs/SimCLR_{dataset}_tst_new_lbl_{trial}.npy", tst_new_lbl)
        print(f"Saved embeddings for {dataset} trial {trial}")
