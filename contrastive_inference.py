from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import Padding, inference_datapipe

# parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--config", default="finetune", help="finetune or fulltrain")
parser.add_argument(
    "-p", "--pad", default="reflect", help="Image padding during inference"
)
parser.add_argument("--head", action="store_true", help="Use projection head")
args = vars(parser.parse_args())

config = args["config"]
if args["pad"] == "constant":
    padding = Padding.CONSTANT
elif args["pad"] == "reflect":
    padding = Padding.REFLECT
else:
    padding = None
head = args["head"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676
NUM_TOTAL = NUM_TRAIN + NUM_TEST

inference_transform = A.Compose(
    [
        A.ToRGB(),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0),
        A.Resize(256, 256),
    ]
)

datapipe_train = inference_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=inference_transform,
    padding=padding,
)
datapipe_test = inference_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=inference_transform,
    padding=padding,
)

dataloader_train = DataLoader(
    datapipe_train, batch_size=512, shuffle=False, num_workers=8
)
dataloader_test = DataLoader(
    datapipe_test, batch_size=512, shuffle=False, num_workers=8
)

# load model
backbone = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
backbone.fc = torch.nn.Identity()
projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 128),
)

# load state dict
backbone.load_state_dict(torch.load(f"model_weights/{config}_resnet18_backbone.pth"))
projection_head.load_state_dict(torch.load(f"model_weights/{config}_resnet18_head.pth"))

if head:
    model = torch.nn.Sequential(backbone, projection_head)
else:
    model = backbone
model.eval()

# store output
if head:
    output = torch.empty((0, 128))
else:
    output = torch.empty((0, 512))
labels = torch.empty((0,))

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
np.save(f"embeddings/output_{config}_resnet18{'_head' if head else ''}.npy", output)
np.save(f"embeddings/labels_{config}_resnet18{'_head' if head else ''}.npy", labels)
