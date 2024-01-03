import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

train_dataset = load_dataset(
    "/local_storage/users/adhkal/planktons_dataset", "2013-14", split="train"
)

train_dataset = train_dataset.with_format("torch")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# create a dataframe to store some features for data exploration
# 1. image label
# 2. image shape
# 3. image area
# 4. image aspect ratio
# 5. image mean
# 6. image std

train_stats = pd.DataFrame(
    columns=[
        "label",
        "shape",
        "area",
        "aspect_ratio",
        "mean",
        "std",
    ]
)

for batch in train_loader:
    image = batch["image"].double()
    train_stats = train_stats.append(
        {
            "label": batch["label"].item(),
            "shape": (image.shape[1], image.shape[2]),
            "area": image.shape[1] * image.shape[2],
            "aspect_ratio": image.shape[1] / image.shape[2],
            "mean": image.mean().item(),
            "std": image.std().item(),
        },
        ignore_index=True,
    )

# save the dataframe
train_stats.to_csv("train_stats.csv", index=False)
