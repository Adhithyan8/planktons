import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image
from tsimcne.tsimcne import TSimCNE
from transformers import AutoImageProcessor, Dinov2Model

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")


# gets both the train and test datasets combined
class PlanktonsPT(Dataset):
    def __init__(self, annotations_file):
        self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        with open(img_path, "rb") as f:
            image = Image.open(f)
            image = T.Resize((224, 224))(image)  # to speed up training
            image = processor(image)["pixel_values"][0]
        label = self.img_labels.iloc[idx, 1]
        return image, label


# both the train and test datasets combined
planktons_dataset = PlanktonsPT(
    annotations_file="/local_storage/users/adhkal/planktons_pytorch/annotations.csv"
)

backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
# mlp with one hidden layer as the projection head
projection_head = torch.nn.Sequential(
    torch.nn.Linear(768, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 128),
)

# tsimcne object (using their default rand initialized resnet18 as backbone)
tsimcne = TSimCNE(
    backbone=backbone,
    projection_head=projection_head,
    total_epochs=[1, 1, 1],
    batch_size=2,
)

# train
tsimcne.fit(planktons_dataset)

# map the images to 2D vectors
vecs = tsimcne.transform(planktons_dataset)

# save the output vectors
np.save("embeddings/vec_tsimcne_resnet18.npy", vecs)
