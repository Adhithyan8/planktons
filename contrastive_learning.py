import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import get_datapipe, InfoNCECosine

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

model_name = "resnet18"

train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

datapipe_1 = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TRAIN,
    transforms=train_transform,
    ignore_mix=True,
    padding=False,
)
datapipe_2 = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    num_images=NUM_TEST,
    transforms=train_transform,
    ignore_mix=True,
    padding=False,
)
# combine the two datapipe
datapipe = datapipe_1.concat(datapipe_2)

# create a dataloader
train_dataloader = DataLoader(datapipe, batch_size=512, shuffle=True, num_workers=12)

if model_name == "resnet18":
    model = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
    model.fc = torch.nn.Identity()
else:
    raise ValueError(f"Model {model_name} not supported")

# projection head which will be removed after training
projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 128),
)

print(model)
