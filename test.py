import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import get_datapipe, contrastive_datapipe

# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 636764
NUM_TOTAL = NUM_TRAIN + NUM_TEST

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
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.5, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
    ]
)

# datapipe = get_datapipe(
#     "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
#     num_images=NUM_TRAIN,
#     transforms=pad_transform,
#     ignore_mix=True,
#     padding=True,
# )
contrastivepipe = contrastive_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=train_transform,
    ignore_mix=True,
)

# dataloader = DataLoader(datapipe, batch_size=1, shuffle=True, num_workers=8)
train_dataloader = DataLoader(
    contrastivepipe, batch_size=1, shuffle=True, num_workers=8
)

# visualize the data
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(4, 4, figsize=(12, 12))

# for i, (img, label) in enumerate(dataloader):
#     if i == 16:
#         break
#     ax[i // 4, i % 4].imshow(img.squeeze().permute(1, 2, 0))
#     ax[i // 4, i % 4].set_title(label.item())
#     ax[i // 4, i % 4].axis("off")

fig, ax = plt.subplots(3, 10, figsize=(24, 8))

for i, (imgs, imgs_1, imgs_2, labels) in enumerate(train_dataloader):
    if i == 10:
        break
    ax[0, i].imshow(imgs.squeeze().permute(1, 2, 0))
    ax[0, i].set_title(labels.item())
    ax[0, i].axis("off")
    ax[1, i].imshow(imgs_1.squeeze().permute(1, 2, 0))
    ax[1, i].set_title(f"{labels.item()} - 1")
    ax[1, i].axis("off")
    ax[2, i].imshow(imgs_2.squeeze().permute(1, 2, 0))
    ax[2, i].set_title(f"{labels.item()} - 2")
    ax[2, i].axis("off")

# save the plot
plt.savefig("test.png")
