from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from torch import hub, nn, optim, utils
import pytorch_lightning as L

from config import CONTRASTIVE_TRANSFORM
from losses import InfoNCECosineSelfSupervised
from utils import Padding, contrastive_datapipe

torch.set_float32_matmul_precision("medium")

"""
Dataset sizes:
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676
NUM_TOTAL = NUM_TRAIN + NUM_TEST


class LightningContrastive(L.LightningModule):
    def __init__(self, head_dim: int, pretrained: bool, loss: nn.Module, n_epochs: int):
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
        out = self(x)
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


def main(args):
    model = LightningContrastive(
        head_dim=args.head_dim,
        pretrained=args.pretrained,
        loss=InfoNCECosineSelfSupervised(),
        n_epochs=args.epochs,
    )

    # transforms and dataloaders
    datapipe = contrastive_datapipe(
        [
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
            "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
        ],
        num_images=NUM_TOTAL,
        transforms=CONTRASTIVE_TRANSFORM,
        padding=Padding.REFLECT,
        ignore_mix=True,
        mask_label=True,
    )

    dataloader = utils.data.DataLoader(
        datapipe, batch_size=args.batch_size, shuffle=True, num_workers=args.devices * 4,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator="gpu", devices=args.devices, num_nodes=args.nodes, strategy="ddp",
    )
    trainer.fit(model, dataloader)

    # save the model
    torch.save(model.backbone.state_dict(), f"{args.name}_backbone.pth")
    torch.save(model.projection_head.state_dict(), f"{args.name}_head.pth")


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="selfsupcauchy_resnet18")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    main(args)
