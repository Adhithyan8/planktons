"""
Time to predict the embeddings.
"""

import copy
from time import time

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from PIL import Image
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *

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
        )  # TODO: try increasing teacher_temp

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
        views, _ = batch  # TODO: currently ignoring labels
        student_out = [self.forward(view) for view in views]
        teacher_out = [self.forward_teacher(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
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
        x, y = batch
        z = self.teacher_backbone(x)
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
    return img, label


# datasets to predict on
datasets = [
    "CUB",
]

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    model = DINO(output_dim=1024)

    # load the trained model
    model.load_state_dict(torch.load(f"outputs/exp_05_{dataset}.pt"))

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
    np.save(f"outputs/DINOUnsup_{dataset}_trn_old_out.npy", trn_old_out)
    np.save(f"outputs/DINOUnsup_{dataset}_trn_new_out.npy", trn_new_out)
    np.save(f"outputs/DINOUnsup_{dataset}_tst_old_out.npy", tst_old_out)
    np.save(f"outputs/DINOUnsup_{dataset}_tst_new_out.npy", tst_new_out)
    np.save(f"outputs/DINOUnsup_{dataset}_trn_old_lbl.npy", trn_old_lbl)
    np.save(f"outputs/DINOUnsup_{dataset}_trn_new_lbl.npy", trn_new_lbl)
    np.save(f"outputs/DINOUnsup_{dataset}_tst_old_lbl.npy", tst_old_lbl)
    np.save(f"outputs/DINOUnsup_{dataset}_tst_new_lbl.npy", tst_new_lbl)
    print(f"Saved embeddings for {dataset}")
