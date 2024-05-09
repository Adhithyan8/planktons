from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from torch import cat, hub, nn, no_grad, optim, save, utils
import lightning as L

from config import CONTRASTIVE_TRANSFORM
from losses import InfoNCECosineSelfSupervised
from utils import Padding, contrastive_datapipe

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
        backbone_resnet = hub.load(
            "pytorch/vision:v0.9.0",
            "resnet18",
            pretrained=pretrained,
        )
        if pretrained:
            frozen_layers = list(backbone_resnet.children())[:-3]
            self.backbone_frozen = nn.Sequential(*frozen_layers)
            self.backbone_frozen.eval()

            trainable_layers = list(backbone_resnet.children())[-3:-1]
            self.backbone_trainable = nn.Sequential(*trainable_layers)
        else:
            self.backbone_frozen = nn.Identity()

            trainable_layers = list(backbone_resnet.children())[:-1]
            self.backbone_trainable = nn.Sequential(*trainable_layers)

        self.projection_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, head_dim),
        )
        self.loss = loss
        self.epochs = n_epochs

    def forward(self, x):
        with no_grad():
            x = self.backbone_frozen(x)
        x = self.backbone_trainable(x)
        x = self.projection_head(x)
        return x

    def training_step(self, batch, batch_idx):
        x1, x2, id = batch
        x = cat((x1, x2), dim=0)
        id = cat((id, id), dim=0)
        out = self(x)
        loss = self.loss(out, id)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, id = batch
        out = self(x)
        return out, id

    def configure_optimizers(self):
        trainable_params = list(self.backbone_trainable.parameters()) + list(
            self.projection_head.parameters()
        )
        optimizer = optim.AdamW(trainable_params, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.12,
            epochs=self.epochs,
            steps_per_epoch=96,
            pct_start=0.02,
            div_factor=1e4,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


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
        datapipe, batch_size=args.batch_size, shuffle=True, num_workers=16
    )

    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator="gpu", devices=args.devices, strategy="ddp"
    )
    trainer.fit(model, dataloader)

    # save the model
    save(model.backbone_trainable.state_dict(), f"{args.name}_backbone.pth")
    save(model.projection_head.state_dict(), f"{args.name}_head.pth")


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="selfsupcauchy_resnet18")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--devices", type=int, default=8)
    args = parser.parse_args()

    main(args)
