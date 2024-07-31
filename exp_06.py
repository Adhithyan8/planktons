"""
exp 05 which was DINO pretraining without labels barely changed performance from the pretrained state
So will try using the label information now

Lets modify the loss for that
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
    def __init__(self, output_dim=256, target_dist=None):  # TODO: try fixed vs variable output_dim
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
            lambda2=2.0,
            center_momentum=0.9,
            target_dist=target_dist,
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
        views, labels = batch
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
            self.student_head.parameters(), lr=1e-4, weight_decay=1e-4,
        )  # TODO: try increasing lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300, eta_min=0
        )
        return [optimizer], [scheduler]


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
    # stochastic augmentations
    augs0 = A.Compose(
        [
            A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
            A.HorizontalFlip(),
            A.GaussianBlur(p=1.0),
            A.Normalize(),
        ]
    )  # TODO: try color / shiftscalerotate
    augs1 = A.Compose(
        [
            A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
            A.HorizontalFlip(),
            A.GaussianBlur(p=0.1),
            A.Solarize(p=0.2),
            A.Normalize(),
        ]
    )  # TODO: currently ignoring the local crops
    img0 = augs0(image=img)["image"]
    img1 = augs1(image=img)["image"]
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

def power_law_dist(out_dim, a=0.5):
    dist = torch.tensor([1 / (i + 1) ** a for i in range(out_dim)])  # unnormalized
    dist = dist / dist.sum()
    return dist

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    if dataset == "CUB":
        info = CUB_INFO
        out_dim = 230
        target_dist = torch.ones(out_dim) / out_dim  # uniform distribution
    elif dataset == "SCARS":
        info = SCARS_INFO
        out_dim = 230
        target_dist = torch.ones(out_dim) / out_dim  # uniform distribution
    elif dataset == "AIRCRAFT":
        info = AIRCRAFT_INFO
        out_dim = 110
        target_dist = torch.ones(out_dim) / out_dim  # uniform distribution
    elif dataset == "HERB19":
        info = HERB19_INFO
        out_dim = 700
        target_dist = power_law_dist(out_dim, a=2.0)
    elif dataset == "PLANKTON":
        info = PLANKTON_INFO
        out_dim = 110
        target_dist = power_law_dist(out_dim, a=2.0)

    model = DINO(output_dim=out_dim, target_dist=target_dist)

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
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    dl = train_dataloader(combined_dataset, 128)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=4,
        num_nodes=1,
        max_epochs=300,
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
    )

    t = time()
    trainer.fit(model, dl)
    print(f"Training time for {dataset}: {time()-t}")

    # save the model
    torch.save(model.state_dict(), f"outputs/exp_06_{dataset}.pt")
    print(f"Model saved for {dataset}")

    # delete the model - cuz I am paranoid
    del model
