"""
exp 04 indicates contrastive learning collapses features and degrades performance a bit.
Maybe instead of using this for representation learning, what if we directly did DINO training

For now we dont use labels, and instead evaluate features using kNN and kMeans clustering
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
            self.student_head.parameters(), lr=1e-1
        )  # TODO: try decreasing lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0
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
    augs = A.Compose(
        [
            A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
            A.HorizontalFlip(),
            A.GaussianBlur(),
            A.Normalize(),
        ]
    )  # TODO: try color / shiftscalerotate
    # TODO: currently ignoring the local crops
    img0 = augs(image=img)["image"]
    img1 = augs(image=img)["image"]
    img0 = torch.from_numpy(img0).permute(2, 0, 1)
    img1 = torch.from_numpy(img1).permute(2, 0, 1)
    # mask labels
    label = -1
    label = torch.tensor(label)
    return (img0, img1), label


# datasets to train on
datasets = [
    "CUB",
]

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    model = DINO(output_dim=1024)  # TODO: try different dims based on dataset

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
        max_epochs=100,
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,
    )

    t = time()
    trainer.fit(model, dl)
    print(f"Training time for {dataset}: {time()-t}")

    # save the model
    torch.save(model.state_dict(), f"outputs/exp_05_{dataset}.pt")
    print(f"Model saved for {dataset}")

    # delete the model - cuz I am paranoid
    del model
