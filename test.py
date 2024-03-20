import albumentations as A
from torch.utils.data import DataLoader

from utils import Padding, contrastive_datapipe, inference_datapipe

# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 636764
NUM_TOTAL = NUM_TRAIN + NUM_TEST

contrastive_transform = A.Compose(
    [
        # shape augmentation
        A.ShiftScaleRotate(p=0.5),
        A.Flip(p=0.5),
        # cutout
        A.CoarseDropout(fill_value=200),
        # color augmentation
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.AdvancedBlur(),
            ],
        ),
        # below are always applied
        # A.ToRGB(),
        # A.ToFloat(max_value=255),
        # A.Normalize(),
        A.RandomResizedCrop(128, 128, scale=(0.2, 1.0)),
    ]
)
inference_transform = A.Compose(
    [
        # A.ToRGB(),
        # A.ToFloat(max_value=255),
        # A.Normalize(),
        A.Resize(256, 256),
    ]
)

datapipe = inference_datapipe(
    ["/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip"],
    num_images=NUM_TRAIN,
    transforms=inference_transform,
    padding=Padding.CONSTANT,
)

repr_learning_pipe = contrastive_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=contrastive_transform,
    padding=Padding.REFLECT,
)

dataloader = DataLoader(datapipe, batch_size=1, shuffle=True, num_workers=8)

contrastive_loader = DataLoader(
    repr_learning_pipe, batch_size=1, shuffle=True, num_workers=8
)

# visualize the data
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1)
for imgs, _, _, _ in contrastive_loader:
    axs.imshow(
        A.Resize(256, 256)(image=imgs.numpy().squeeze())["image"],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axs.axis("off")
    break

# save the plot
plt.tight_layout()
plt.savefig("test.png")
plt.close()

fig, axs = plt.subplots(2, 3)
for i in range(2):
    for j in range(3):
        axs[i, j].imshow(
            contrastive_transform(image=imgs.numpy().squeeze())["image"],
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axs[i, j].axis("off")

# save the plot
plt.tight_layout()
plt.savefig("augs.png")
plt.close()
