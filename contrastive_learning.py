from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import albumentations as A
import torch
from torch.utils.data import DataLoader

from utils import Padding, contrastive_datapipe
from losses import InfoNCECauchySelfSupervised, InfoNCECauchySupervised, InfoNCECauchySemiSupervised

# parse arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", default="selfcy_resnet18")
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--epochs", type=int, default=250)
parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
parser.add_argument("--freeze", action="store_true", help="Freeze early layers")
parser.add_argument("--head-dim", type=int, default=128)

args = vars(parser.parse_args())
name = args["name"]
batch_size = args["batch_size"]
n_epochs = args["epochs"]
pretrained = args["pretrained"]
freeze = args["freeze"]
head_dim = args["head_dim"]
padding = Padding.REFLECT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# transforms and dataloaders
contrastive_transform = A.Compose(
    [
        A.ShiftScaleRotate(p=0.5),
        A.Flip(p=0.5),
        A.CoarseDropout(fill_value=200),
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.AdvancedBlur(),
            ],
        ),
        A.ToRGB(),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0),
        A.RandomResizedCrop(128, 128, scale=(0.2, 1.0)),
    ]
)

datapipe = contrastive_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=contrastive_transform,
    padding=padding,
    ignore_mix=True,
    mask_label=True,
)

dataloader = DataLoader(datapipe, batch_size=batch_size, shuffle=True, num_workers=16)

# load model
backbone = torch.hub.load(
    "pytorch/vision:v0.9.0",
    "resnet18",
    pretrained=pretrained,
)
backbone.fc = torch.nn.Identity()

# freezing early layers
if freeze:
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.layer4.parameters():
        param.requires_grad = True
else:
    for param in backbone.parameters():
        param.requires_grad = True

projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, head_dim),
)

# combine the model and the projection head
model = torch.nn.Sequential(backbone, projection_head)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=5e-4)

# lr scheduler with linear warmup and cosine decay
max_lr = 0.03 * (batch_size / 256)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=n_epochs,
    steps_per_epoch=96,  # weird bug
    pct_start=0.02,
    div_factor=1e4,  # start close to 0
    final_div_factor=1e4,  # end close to 0
)

# loss
criterion = InfoNCECauchySemiSupervised(temperature=1.0)
criterion1 = InfoNCECauchySelfSupervised(temperature=0.5)
criterion2 = InfoNCECauchySupervised(temperature=0.5)
lamda = 0.5

print(f"Training {name} model")
# train the model
model.train().to(device)
for epoch in range(n_epochs):
    for i, (img1, img2, id) in enumerate(dataloader):
        img1, img2, id = img1.to(device), img2.to(device), id.to(device)
        optimizer.zero_grad()
        img = torch.cat((img1, img2), dim=0)
        id = torch.cat((id, id), dim=0)
        output = model(img)
        loss = criterion(output, id)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# save the model
torch.save(model[0].state_dict(), f"{name}_backbone.pth")
torch.save(model[1].state_dict(), f"{name}_head.pth")
