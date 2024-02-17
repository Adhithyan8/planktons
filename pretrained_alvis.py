import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import get_datapipe

model_name = "resnet18"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
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
    num_images=115951,
    transforms=train_transform,
    ignore_mix=True,
)
test_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=63676,
    transforms=test_transform,
    ignore_mix=True,
)
train_dataloader = DataLoader(
    train_datapipe, batch_size=512, shuffle=False, num_workers=4
)
test_dataloader = DataLoader(
    test_datapipe, batch_size=512, shuffle=False, num_workers=4
)

# model (pick one)
if model_name == "resnet18":
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
