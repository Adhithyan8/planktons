import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import get_datapipe

# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676

pad_transform = transforms.Compose(
    [
        transforms.Resize(224),
    ]
)

no_pad_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
)

datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TRAIN,
    transforms=pad_transform,
    ignore_mix=True,
    padding=True,
)
dataloader = DataLoader(datapipe, batch_size=1, shuffle=True, num_workers=8)

# visualize the data
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 4, figsize=(12, 12))

for i, (img, label) in enumerate(dataloader):
    if i == 16:
        break
    ax[i // 4, i % 4].imshow(img.squeeze().permute(1, 2, 0))
    ax[i // 4, i % 4].set_title(label.item())
    ax[i // 4, i % 4].axis("off")

# save the plot
plt.savefig("test.png")
