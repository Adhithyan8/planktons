from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import Padding, inference_datapipe

# parse arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", default="resnet18")
parser.add_argument("--padding", default="reflect")
args = vars(parser.parse_args())

name = args["name"]
if args["padding"] == "constant":
    padding = Padding.CONSTANT
elif args["padding"] == "reflect":
    padding = Padding.REFLECT
else:
    padding = None

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
        A.Resize(252, 252) if name == "vitb14-dinov2" else A.Resize(256, 256),
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
    datapipe_train, batch_size=512, shuffle=False, num_workers=12
)
dataloader_test = DataLoader(
    datapipe_test, batch_size=512, shuffle=False, num_workers=12
)

# create model
if name == "resnet18":
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    model.fc = torch.nn.Identity()
elif name == "resnet50":
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
    model.fc = torch.nn.Identity()
elif name == "vitb14-dinov2":
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
else:
    raise ValueError("Invalid model name")
model.eval()

# storing output
if name == "resnet18":
    output = torch.empty((0, 512))
elif name == "resnet50":
    output = torch.empty((0, 2048))
elif name == "vitb14-dinov2":
    output = torch.empty((0, 768))
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
np.save(f"embeddings/output_{name}.npy", output)
np.save(f"embeddings/labels_{name}.npy", labels)
