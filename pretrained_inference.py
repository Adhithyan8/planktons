import numpy as np
import torch
import torch.hub
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import get_datapipe

model_name = "resnet50"
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

train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TRAIN,
    transforms=train_transform,
    ignore_mix=True,
)
test_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TEST,
    transforms=test_transform,
    ignore_mix=True,
)
train_dataloader = DataLoader(
    train_datapipe, batch_size=1024, shuffle=False, num_workers=12
)
test_dataloader = DataLoader(
    test_datapipe, batch_size=1024, shuffle=False, num_workers=12
)

# model (pick one)
if model_name == "resnet18":
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
elif model_name == "resnet50":
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
    model.fc = torch.nn.Identity()
    print(model)
    model.eval()
elif model_name == "dinov2":
    model = torch.hub.load("facebookresearch/dino:main", "dino_v2_8")
    model = model.eval()
else:
    raise ValueError("Invalid model name")

# store output
if model_name == "resnet18":
    output = torch.empty((0, 512))
elif model_name == "resnet50":
    output = torch.empty((0, 2048))
elif model_name == "dinov2":
    output = torch.empty((0, 384))
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
np.save(f"embeddings/output_{model_name}.npy", output)
np.save(f"embeddings/labels.npy", labels)
