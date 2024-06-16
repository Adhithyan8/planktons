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
