"""
We do DINO training with target distribution
"""

import copy
from time import time

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from PIL import Image
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities import grad_norm
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *
from losses import DINOLoss  # modified from lightly to use labels
from model import CosineClassifier  # to avoid norm issue as per Sagar (2023)

# continuing with this precision setting
torch.set_float32_matmul_precision("high")


# nearly a one-to-one copy from lightly examples
class DINO(L.LightningModule):
    def __init__(
        self,
        output_dim=256,
        target_dist=None,
        weight_name=None,
        learning_rate=1e-4,
    ):
        super(DINO, self).__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        # load weights from the contrastive pre-training
        if weight_name is not None:
            backbone.load_state_dict(torch.load(weight_name), strict=False)
        # freeze all blocks except the last one
        for param in backbone.parameters():
            param.requires_grad_(False)
        for name, param in backbone.named_parameters():
            if "block" in name:
                block_num = int(name.split(".")[1])
                if block_num >= 11:
                    param.requires_grad_(True)
        head = CosineClassifier(768, output_dim)
        self.student_backbone = backbone
        self.student_head = head
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = copy.deepcopy(head)
        self.learning_rate = learning_rate
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        # centering to avoid collapse
        self.criterion = DINOLoss(
            output_dim=output_dim,
            center_momentum=0.9,  # TODO: also centering
            target_dist=target_dist,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            lambda0=0.65,
            lambda1=0.35,
            lambda2=2.0,
        )

    def forward(self, x):
        y = self.student_backbone(x)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 200, 0.996, 1.0)
        update_momentum(self.student_backbone, self.teacher_backbone, momentum)
        update_momentum(self.student_head, self.teacher_head, momentum)
        (x_teacher, x_student), labels = batch
        student_out = self.forward(x_student)
        teacher_out = self.forward_teacher(x_teacher)
        loss = self.criterion(
            teacher_out, student_out, labels, epoch=self.current_epoch
        )
        self.log("loss", loss, prog_bar=True)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_gradients(epoch=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-3,
            decoupled_weight_decay=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=0
        )
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(module=self, norm_type=2)  # returns dict
        self.log("grad_norm", norms["grad_2.0_norm_total"], prog_bar=True)


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
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.Normalize(),
        ]
    )
    augs1 = A.Compose(
        [
            A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
            A.HorizontalFlip(),
            A.Solarize(),
            A.GaussianBlur(),
            A.Normalize(),
        ]
    )
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
    "PLANKTON",
]
trials = 3

# given the info, split and transform, make_dataset should give us the dataset
for dataset in datasets:
    for trial in range(trials):
        if dataset == "CUB":
            info = CUB_INFO
            out_dim = 230
            target_dist = None
        elif dataset == "SCARS":
            info = SCARS_INFO
            out_dim = 230
            # shift labels to start from 0
            for sample in info:
                sample["label"] -= 1
            target_dist = None
        elif dataset == "AIRCRAFT":
            info = AIRCRAFT_INFO
            out_dim = 110
            target_dist = None
        elif dataset == "HERB19":
            info = HERB19_INFO
            out_dim = 700
            target_dist = torch.tensor(HERB19_DIST)
            # add trailing zeros to target_dist to match out_dim
            target_dist = torch.cat(
                [target_dist, torch.zeros(out_dim - len(target_dist))]
            )
        elif dataset == "PLANKTON":
            info = PLANKTON_INFO
            out_dim = 110
            target_dist = torch.tensor(PLANKTON_DIST)
            # add trailing zeros to target_dist to match out_dim
            target_dist = torch.cat(
                [target_dist, torch.zeros(out_dim - len(target_dist))]
            )

        model = DINO(
            output_dim=out_dim,
            target_dist=target_dist,
            weight_name=f"outputs/exp_04_{dataset}_trial_{trial}.pt",
        )

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
            max_epochs=200,
            strategy="ddp",
            sync_batchnorm=True,
            use_distributed_sampler=True,
        )

        t = time()
        trainer.fit(model, dl)
        print(f"Training time for {dataset}: {time()-t}")

        # save the model
        torch.save(model.state_dict(), f"outputs/exp_07_{dataset}_trial_{trial}.pt")
        print(f"Model saved for {dataset} trial {trial}")

        # delete the model - cuz I am paranoid
        del model
