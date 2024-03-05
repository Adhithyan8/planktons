from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import get_datapipe

# parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--config", default="finetune", help="finetune or fulltrain")
parser.add_argument("-h", "--head", action="store_true", help="Use projection head")
args = vars(parser.parse_args())

config = args["config"]
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

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TRAIN,
    transforms=transform,
    ignore_mix=True,
    padding=True,
)
test_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    num_images=NUM_TEST,
    transforms=transform,
    ignore_mix=True,
    padding=True,
)

train_dataloader = DataLoader(
    train_datapipe, batch_size=512, shuffle=False, num_workers=8
)
test_dataloader = DataLoader(
    test_datapipe, batch_size=512, shuffle=False, num_workers=8
)

# load model
backbone = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", weights="DEFAULT")
backbone.fc = torch.nn.Identity()
projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 128),
)


# load state dict
backbone.load_state_dict(torch.load(f"{config}_resnet18_backbone.pth"))
projection_head.load_state_dict(torch.load(f"{config}_resnet18_head.pth"))

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
for images, labels_batch in train_dataloader:
    images = images.to(device)
    with torch.no_grad():
        output_batch = model(images)
    output = torch.cat((output, output_batch.cpu()))
    labels = torch.cat((labels, labels_batch))
for images, labels_batch in test_dataloader:
    images = images.to(device)
    with torch.no_grad():
        output_batch = model(images)
    output = torch.cat((output, output_batch.cpu()))
    labels = torch.cat((labels, labels_batch))

# to numpy
output = output.numpy()
labels = labels.numpy()

# save
np.save(f"embeddings/output_{config}{'_head' if head else ''}.npy", output)
np.save(f"embeddings/labels.npy", labels)
