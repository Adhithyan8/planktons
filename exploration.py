import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

train_dataset = load_dataset(
    "/local_storage/users/adhkal/planktons_dataset", "2013-14", split="train"
)

train_dataset = train_dataset.with_format("torch")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# images have single channel and different sizes
# TODO: resize images to a fixed size
# TODO: normalize images, make 3 channels


# given index, show image
def show_image(index):
    image = train_dataset[index]["image"]
    label = train_dataset[index]["label"]
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.title(f"Label: {label}")
    plt.savefig(f"image_{index}.png")
