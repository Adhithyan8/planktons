import torch
from torch import hub, nn, optim
import pytorch_lightning as L

class LightningContrastive(L.LightningModule):
    def __init__(self, head_dim: int, pretrained: bool, loss: nn.Module, n_epochs: int, use_head: bool = False):
        super().__init__()
        self.backbone = hub.load(
            "pytorch/vision:v0.9.0",
            "resnet18",
            pretrained=pretrained,
        )
        self.backbone.fc = nn.Identity()
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
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
        self.loss = loss
        self.epochs = n_epochs
        self.use_head = use_head

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
        x, id = batch
        if self.use_head:
            out = self(x)
        else:
            out = self.backbone(x)
        return out, id

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), weight_decay=5e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.12,
            total_steps=self.epochs,
            epochs=self.epochs,
            pct_start=0.02,
            div_factor=1e4,
            final_div_factor=1e4,
        )
        return [optimizer], [scheduler]
