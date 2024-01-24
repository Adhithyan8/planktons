import time

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image
from tsimcne.tsimcne import TSimCNE

t = time.time()


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
        label = self.img_labels.iloc[idx, 1]
        return image, label


# both the train and test datasets combined
planktons_dataset = PlanktonsPT(
    annotations_file="/local_storage/users/adhkal/planktons_pytorch/annotations.csv"
)  # not sure if reshaping is needed

# tsimcne object (using their default rand initialized resnet18 as backbone)
tsimcne = TSimCNE(
    data_transform=None,  # Change this later
    total_epochs=[1, 1, 1],  # for testing
    batch_size=32,
)

# train
tsimcne.fit(planktons_dataset)

print(f"Time taken to train the model: {time.time()-t}")

# map the images to 2D vectors
vecs = tsimcne.transform(planktons_dataset)

print(f"Time taken to get the vectors: {time.time()-t}")

# save the output vectors
np.save("embeddings/vec_tsimcne_resnet18.npy", vecs)
