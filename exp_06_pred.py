"""
exp 06 is DINO pretraining with labels.
We now predict with the projection head
"""

import copy
from time import time

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from PIL import Image
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *
from losses import DINOLoss

# continuing with this precision setting
torch.set_float32_matmul_precision("medium")


# nearly a one-to-one copy from lightly examples
class DINO(L.LightningModule):
    def __init__(self, output_dim=256):  # TODO: try fixed vs variable output_dim
        super(DINO, self).__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        # freeze all blocks except the last one
        for param in backbone.parameters():
            param.requires_grad_(False)
        for name, param in backbone.named_parameters():
            if "block" in name:
                block_num = int(name.split(".")[1])
                if block_num >= 11:
                    param.requires_grad_(True)

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim=768, hidden_dim=2048, bottleneck_dim=256, output_dim=output_dim
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim=768, hidden_dim=2048, bottleneck_dim=256, output_dim=output_dim
        )
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=output_dim,
            warmup_teacher_temp=0.07,
            teacher_temp=0.04,
            lambda0=0.65,
            lambda1=0.35,
            lambda2=0.0,
        )  # TODO: try decreasing teacher_temp

    def forward(self, x):
        y = self.student_backbone(x)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 100, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, momentum)
        update_momentum(self.student_head, self.teacher_head, momentum)
        views, labels = batch  # TODO: currently ignoring labels
        student_out = [self.forward(view) for view in views]
        teacher_out = [self.forward_teacher(view) for view in views]
        loss = self.criterion(
            teacher_out, student_out, labels, epoch=self.current_epoch
        )
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student_head.parameters(), lr=1e-4
        )  # TODO: try increasing lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0
        )
        return [optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx):
        img, labels = batch
        out = self.forward_teacher(img)
        return out, labels


# lets define the transforms for DINO training
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
    img = torch.from_numpy(img).permute(2, 0, 1)
    label = torch.tensor(label, dtype=torch.long)
    return img, label


def mask_label_transform(img, label):
    (img0, img1), label = data_transform(img, label)
    # set label to -1 to mask it
    label = -1 * torch.ones_like(label)
    return (img0, img1), label


# datasets to train on
datasets = [
    "CUB",
]

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    if dataset == "CUB":
        info = CUB_INFO
        out_dim = 230
    elif dataset == "SCARS":
        info = SCARS_INFO
        out_dim = 230
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
    model.load_state_dict(torch.load(f"outputs/exp_06_{dataset}.pt"))

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
    np.save(f"outputs/DINOSup_{dataset}_trn_old_out.npy", trn_old_out)
    np.save(f"outputs/DINOSup_{dataset}_trn_new_out.npy", trn_new_out)
    np.save(f"outputs/DINOSup_{dataset}_tst_old_out.npy", tst_old_out)
    np.save(f"outputs/DINOSup_{dataset}_tst_new_out.npy", tst_new_out)
    np.save(f"outputs/DINOSup_{dataset}_trn_old_lbl.npy", trn_old_lbl)
    np.save(f"outputs/DINOSup_{dataset}_trn_new_lbl.npy", trn_new_lbl)
    np.save(f"outputs/DINOSup_{dataset}_tst_old_lbl.npy", tst_old_lbl)
    np.save(f"outputs/DINOSup_{dataset}_tst_new_lbl.npy", tst_new_lbl)
    print(f"Saved embeddings for {dataset}")
