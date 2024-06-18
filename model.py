import copy

import numpy as np
import pytorch_lightning as L
import torch
from torch import hub, nn, optim


class LightningContrastive(L.LightningModule):
    def __init__(
        self,
        head_dim: int,
        pretrained: bool,
        loss: nn.Module,
        n_epochs: int,
        use_head: bool = False,
        uuid: bool = False,
        arch: str = "resnet",
    ):
        super().__init__()
        if arch == "resnet":
            self.backbone = hub.load(
                "pytorch/vision:v0.9.0",
                "resnet18",
                pretrained=pretrained,
            )
            self.backbone.fc = nn.Identity()
            if pretrained:
                for param in self.backbone.parameters():
                    param.requires_grad_(True)
                for param in self.backbone.layer4.parameters():
                    param.requires_grad_(True)
            else:
                for param in self.backbone.parameters():
                    param.requires_grad_(True)

            self.projection_head = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, head_dim),
            )
        elif arch == "vit":
            self.backbone = hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14_reg",
            )
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            for name, param in self.backbone.named_parameters():
                if "block" in name:
                    block_num = int(name.split(".")[1])
                    if block_num >= 11:
                        param.requires_grad_(True)
            self.projection_head = nn.Sequential(
                nn.Linear(768, 1024),
                nn.ReLU(),
                nn.Linear(1024, head_dim),
            )
        self.loss = loss
        self.epochs = n_epochs
        self.use_head = use_head
        self.uuid = uuid

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x1, x2, id = batch
        x = torch.cat((x1, x2), dim=0)
        id = torch.cat((id, id), dim=0)
        out = self(x)
        loss = self.loss(out, id)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        if self.uuid:
            x, id, fname = batch
        else:
            x, id = batch
        if self.use_head:
            out = self(x)
        else:
            out = self.backbone(x)
        if self.uuid:
            return out, id, fname
        else:
            return out, id

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=0
        )
        return [optimizer], [scheduler]


class LightningTsimnce(L.LightningModule):
    def __init__(
        self,
        name: str,
        old_head_dim: int,
        new_head_dim: int,
        loss: nn.Module,
        n_epochs: int,
        phase: str = "readout",
        uuid: bool = False,
    ):
        super().__init__()
        if phase == "readout":
            self.backbone = hub.load(
                "pytorch/vision:v0.9.0",
                "resnet18",
                pretrained=True,
            )
            self.backbone.fc = nn.Identity()
            self.backbone.load_state_dict(torch.load(f"model_weights/{name}_bb.pth"))
            for param in self.backbone.parameters():
                param.requires_grad_(False)

            self.projection_head = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, old_head_dim),
            )
            self.projection_head.load_state_dict(
                torch.load(f"model_weights/{name}_ph.pth")
            )
            self.projection_head[-1] = nn.Linear(1024, new_head_dim)
            for param in self.projection_head.parameters():
                param.requires_grad_(False)
            for param in self.projection_head[-1].parameters():
                param.requires_grad_(True)

        elif phase == "finetune":
            self.backbone = hub.load(
                "pytorch/vision:v0.9.0",
                "resnet18",
                pretrained=True,
            )
            self.backbone.fc = nn.Identity()
            self.backbone.load_state_dict(
                torch.load(f"model_weights/read_{name}_bb.pth")
            )
            for param in self.backbone.parameters():
                param.requires_grad_(True)  # finetuning all layers
            for param in self.backbone.layer4.parameters():
                param.requires_grad_(True)

            self.projection_head = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, new_head_dim),
            )
            self.projection_head.load_state_dict(
                torch.load(f"model_weights/read_{name}_ph.pth")
            )
            for param in self.projection_head.parameters():
                param.requires_grad_(True)

        self.loss = loss
        self.epochs = n_epochs
        self.phase = phase
        self.uuid = uuid

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x1, x2, id = batch
        x = torch.cat((x1, x2), dim=0)
        id = torch.cat((id, id), dim=0)
        out = self(x)
        loss = self.loss(out, id)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        if self.uuid:
            x, id, fname = batch
            out = self(x)
            return out, id, fname
        else:
            x, id = batch
            out = self(x)
            return out, id

    def configure_optimizers(self):
        if self.phase == "readout":
            optimizer = optim.AdamW(self.parameters(), lr=0.1, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.12,
                total_steps=self.epochs,
                epochs=self.epochs,
                pct_start=0.05,
                div_factor=1e4,
                final_div_factor=1e4,
            )
            return [optimizer], [scheduler]
        elif self.phase == "finetune":
            optimizer = optim.AdamW(self.parameters(), lr=0.00012, weight_decay=5e-4)
            return optimizer


class LightningPretrained(L.LightningModule):
    def __init__(self, name: str):
        super().__init__()
        if name == "resnet18":
            self.model = hub.load(
                "pytorch/vision:v0.10.0",
                "resnet18",
                pretrained=True,
            )
            self.model.fc = nn.Identity()
        elif name == "resnet50":
            self.model = hub.load(
                "pytorch/vision:v0.10.0",
                "resnet50",
                pretrained=True,
            )
            self.model.fc = nn.Identity()
        elif name == "vitb14-dinov2":
            self.model = hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14_reg",
            )
        else:
            raise ValueError("Invalid model name")

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        x, id = batch
        out = self(x)
        return out, id


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, freeze_layer: int = -1):
        super().__init__()
        self.layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(in_dim, out_dim, bias=False)
        )
        self.layer.parametrizations.weight.original0.data.fill_(1)  # weight norm to 1
        self.layer.parametrizations.weight.original0.requires_grad = (
            False  # freeze weight norm
        )
        self.freeze_layer = freeze_layer

    def cancel_gradients(self, epoch: int):
        if epoch >= self.freeze_layer:
            return
        for param in self.layer.parameters():
            param.grad = None

    def forward(self, x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.layer(x)
        return x


@torch.no_grad()
def update_teacher(model_ema: nn.Module, model: nn.Module, m: float):
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
):
    return end_value - 0.5 * (1 - start_value) * (
        1 + np.cos(np.pi * step / (max_steps - 1))
    )


class LightningMuContrastive(L.LightningModule):
    def __init__(
        self,
        name: str,
        out_dim: int,
        loss: nn.Module,
        n_epochs: int,
        uuid: bool = False,
        arch: str = "resnet",
    ):
        super().__init__()
        if arch == "resnet":
            backbone = hub.load(
                "pytorch/vision:v0.9.0",
                "resnet18",
                pretrained=True,
            )
            backbone.fc = nn.Identity()
            backbone.load_state_dict(torch.load(f"model_weights/{name}_bb.pth"))
            cls_head = CosineClassifier(512, out_dim)
        elif arch == "vit":
            backbone = hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14_reg",
            )
            backbone.load_state_dict(torch.load(f"model_weights/{name}_bb.pth"))
            for param in backbone.parameters():
                param.requires_grad_(False)
            for name, param in backbone.named_parameters():
                if "block" in name:
                    block_num = int(name.split(".")[1])
                    if block_num >= 11:
                        param.requires_grad_(True)
            cls_head = CosineClassifier(768, out_dim)
        self.teacher_backbone = backbone
        self.student_backbone = copy.deepcopy(backbone)
        self.teacher_head = cls_head
        self.student_head = copy.deepcopy(cls_head)

        # freeze teacher, update using momentum
        for param in self.teacher_backbone.parameters():
            param.requires_grad_(False)
        for param in self.teacher_head.parameters():
            param.requires_grad_(False)

        self.loss = loss
        self.epochs = n_epochs
        self.uuid = uuid

    def forward_teacher(self, x):
        x = self.teacher_backbone(x)
        x = self.teacher_head(x)
        return x

    def forward(self, x):
        x = self.student_backbone(x)
        x = self.student_head(x)
        return x

    def training_step(self, batch, batch_idx):
        m = cosine_schedule(self.current_epoch, self.epochs, 0.999, 0.7)
        update_teacher(self.teacher_backbone, self.student_backbone, m)
        update_teacher(self.teacher_head, self.student_head, m)

        x_t, x_s, id = batch
        x_t = self.forward_teacher(x_t)
        x_s = self.forward(x_s)
        loss = self.loss(x_t, x_s, id, self.current_epoch)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_gradients(self.current_epoch)

    def predict_step(self, batch, batch_idx):
        if self.uuid:
            x, id, fname = batch
        else:
            x, id = batch
        out = self.forward_teacher(x)  # using teacher for inference
        if self.uuid:
            return out, id, fname
        else:
            return out, id

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-3,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=0
        )
        return [optimizer], [scheduler]
