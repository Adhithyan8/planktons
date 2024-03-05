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
            scale=(0.1, 1.0),
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
    contrastivepipe, batch_size=8, shuffle=True, num_workers=8
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

fig, ax = plt.subplots(4, 4, figsize=(12, 12))

for imgs_1, imgs_2, labels in train_dataloader:
    for i in range(16):
        # show img_1s, img_2s, and labels
        ax[i // 4, i % 4].imshow(imgs_1[i].squeeze().permute(1, 2, 0))
        ax[i // 4, i % 4].set_title(f"{labels[i].item()} (1)")
        ax[i // 4, i % 4].axis("off")
        ax[i // 4, i % 4 + 4].imshow(imgs_2[i].squeeze().permute(1, 2, 0))
        ax[i // 4, i % 4 + 4].set_title(f"{labels[i].item()} (2)")
        ax[i // 4, i % 4 + 4].axis("off")
    break

# save the plot
plt.savefig("test.png")
