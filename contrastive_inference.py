from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import Padding, inference_datapipe

# parse arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", default="finetune_resnet18")
parser.add_argument("--padding", default="reflect")
parser.add_argument("--head", action="store_true", help="Use projection head")
parser.add_argument("--head-dim", type=int, default=128)

args = vars(parser.parse_args())
name = args["name"]
if args["pad"] == "constant":
    padding = Padding.CONSTANT
elif args["pad"] == "reflect":
    padding = Padding.REFLECT
else:
    padding = None
use_head = args["head"]
head_dim = args["head_dim"]

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
inference_transform = A.Compose(
    [
        A.ToRGB(),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0),
        A.Resize(256, 256),  # inference is at higher res than training
    ]
)

datapipe_train = inference_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    ],
    num_images=NUM_TRAIN,
    transforms=inference_transform,
    padding=padding,
)
datapipe_test = inference_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TEST,
    transforms=inference_transform,
    padding=padding,
)

dataloader_train = DataLoader(
    datapipe_train,
    batch_size=512,
    shuffle=False,
    num_workers=8,
)
dataloader_test = DataLoader(
    datapipe_test,
    batch_size=512,
    shuffle=False,
    num_workers=8,
)

# load model and weights
backbone = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
backbone.fc = torch.nn.Identity()
projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, head_dim),
)
backbone.load_state_dict(torch.load(f"model_weights/{name}_backbone.pth"))
projection_head.load_state_dict(torch.load(f"model_weights/{name}_head.pth"))

# configure model
if use_head:
    model = torch.nn.Sequential(backbone, projection_head)
else:
    model = backbone
model.eval()

# storing embeddings
if use_head:
    output = torch.empty((0, head_dim))
else:
    output = torch.empty((0, 512))
labels = torch.empty((0,))

# inference
model.to(device)
for images, labels_batch in dataloader_train:
    images = images.to(device)
    with torch.no_grad():
        output_batch = model(images)
    output = torch.cat((output, output_batch.cpu()))
    labels = torch.cat((labels, labels_batch))
for images, labels_batch in dataloader_test:
    images = images.to(device)
    with torch.no_grad():
        output_batch = model(images)
    output = torch.cat((output, output_batch.cpu()))
    labels = torch.cat((labels, labels_batch))

# to numpy
output = output.numpy()
labels = labels.numpy()

# save
np.save(f"embeddings/output_{name}{'_head' if use_head else ''}.npy", output)
np.save(f"embeddings/labels_{name}{'_head' if use_head else ''}.npy", labels)
