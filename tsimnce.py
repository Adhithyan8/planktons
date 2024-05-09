from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from torch.utils.data import DataLoader

from config import CONSTRASTIVE_TRANSFORM
from utils import Padding, contrastive_datapipe
from losses import (
    InfoNCECauchySelfSupervised,
    InfoNCECauchySupervised,
    InfoNCECauchySemiSupervised,
)

# parse arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", default="selfcy_resnet18")
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--readout-epochs", type=int, default=50)
parser.add_argument("--old-head-dim", type=int, default=128)
parser.add_argument("--new-head-dim", type=int, default=2)

args = vars(parser.parse_args())
name = args["name"]
batch_size = args["batch_size"]
readout_epochs = args["readout_epochs"]
old_head_dim = args["old_head_dim"]
new_head_dim = args["new_head_dim"]
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

datapipe = contrastive_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=CONSTRASTIVE_TRANSFORM,
    padding=padding,
    ignore_mix=True,
    mask_label=True,
)

dataloader = DataLoader(datapipe, batch_size=batch_size, shuffle=True, num_workers=16)

# load model
backbone = torch.hub.load(
    "pytorch/vision:v0.9.0",
    "resnet18",
    pretrained=True,
)
backbone.fc = torch.nn.Identity()

# freezing all layers
for param in backbone.parameters():
    param.requires_grad = False

projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, old_head_dim),
)

# load weights
backbone.load_state_dict(torch.load(f"model_weights/{name}_backbone.pth"))
projection_head.load_state_dict(torch.load(f"model_weights/{name}_head.pth"))

# change the last layer of the projection head to the new head dimension
projection_head[-1] = torch.nn.Linear(1024, new_head_dim)

# freeze the projection head except the last layer
for param in projection_head.parameters():
    param.requires_grad = False
for param in projection_head[-1].parameters():
    param.requires_grad = True

# combine the model and the projection head
model = torch.nn.Sequential(backbone, projection_head)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=5e-4)

# lr scheduler with linear warmup and cosine decay
max_lr = 0.03 * (batch_size / 256)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=readout_epochs,
    steps_per_epoch=96,  # weird bug
    pct_start=0.02,
    div_factor=1e4,  # start close to 0
    final_div_factor=1e4,  # end close to 0
)

# loss
criterion = InfoNCECauchySemiSupervised()

print(f"Training {name} model")
# train the model
model.train().to(device)
for epoch in range(readout_epochs):
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
            print(f"Epoch [{epoch+1}/{readout_epochs}], Loss: {loss.item():.4f}")

# save the model
torch.save(model[0].state_dict(), f"read_{name}_backbone.pth")
torch.save(model[1].state_dict(), f"read_{name}_head.pth")
