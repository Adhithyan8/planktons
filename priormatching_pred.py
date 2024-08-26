"""
Time to predict the embeddings.
"""

import copy

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *
from losses import DINOLoss
from model import CosineClassifier

# continuing with this precision setting
torch.set_float32_matmul_precision("high")


# nearly a one-to-one copy from lightly examples
class DINO(L.LightningModule):
    def __init__(self, output_dim=256):  # TODO: try fixed vs variable output_dim
        super(DINO, self).__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.student_backbone = backbone
        self.student_head = CosineClassifier(768, output_dim)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = CosineClassifier(768, output_dim)
        self.criterion = DINOLoss(output_dim=output_dim)

    def forward(self, x):
        y = self.student_backbone(x)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        z = self.teacher_head(y)
        return z

    def predict_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward_teacher(x)
        return z, y


# lets define the transforms
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
    "PLANKTON",
]
trials = 3

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    for trial in range(trials):
        if dataset == "CUB":
            info = CUB_INFO
            out_dim = 230
        elif dataset == "SCARS":
            info = SCARS_INFO
            out_dim = 230
            # shift labels to start from 0
            for sample in info:
                sample["label"] -= 1
        elif dataset == "AIRCRAFT":
            info = AIRCRAFT_INFO
            out_dim = 110
        elif dataset == "HERB19":
            info = HERB19_INFO
            out_dim = 700
        elif dataset == "PLANKTON":
            info = PLANKTON_INFO
            out_dim = 110

        model = DINO(output_dim=out_dim)

        # load the trained model
        model.load_state_dict(torch.load(f"outputs/exp_08_{dataset}_trial_{trial}.pt"))

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
        np.save(f"outputs/CTDINO_{dataset}_trn_old_out_{trial}.npy", trn_old_out)
        np.save(f"outputs/CTDINO_{dataset}_trn_new_out_{trial}.npy", trn_new_out)
        np.save(f"outputs/CTDINO_{dataset}_tst_old_out_{trial}.npy", tst_old_out)
        np.save(f"outputs/CTDINO_{dataset}_tst_new_out_{trial}.npy", tst_new_out)
        np.save(f"outputs/CTDINO_{dataset}_trn_old_lbl_{trial}.npy", trn_old_lbl)
        np.save(f"outputs/CTDINO_{dataset}_trn_new_lbl_{trial}.npy", trn_new_lbl)
        np.save(f"outputs/CTDINO_{dataset}_tst_old_lbl_{trial}.npy", tst_old_lbl)
        np.save(f"outputs/CTDINO_{dataset}_tst_new_lbl_{trial}.npy", tst_new_lbl)
        print(f"Predictions saved for {dataset} trial {trial}")
