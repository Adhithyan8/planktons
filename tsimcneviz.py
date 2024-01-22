import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models import resnet18
from tsimcne.tsimcne import TSimCNE
import time

t = time.time()
# dataset
train_dataset = load_dataset(
    "/local_storage/users/adhkal/planktons_dataset",
    "2013-14",
    split="train",
)
test_dataset = load_dataset(
    "/local_storage/users/adhkal/planktons_dataset",
    "2013-14",
    split="test",
)

# combine train and test
full_dataset = concatenate_datasets([train_dataset, test_dataset])
# preprocess the dataset
full_dataset = full_dataset.map(
    lambda x: {
        "image": x["image"].convert("RGB"),
        "label": x["label"],
    }
)
full_dataset = full_dataset.map(
    lambda x: {
        "image": T.Resize((224, 224))(x["image"]),
        "label": x["label"],
    }
)

print(f"Time taken to preprocess the dataset: {time.time()-t}")
t = time.time()


# tsimcne only takes in a torch dataset
# HACKY FIX: convert the huggingface dataset to torch dataset
class TorchDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        lbl = self.dataset[idx]["label"]
        return img, lbl

    def __len__(self):
        return len(self.dataset)


full_dataset = TorchDataset(full_dataset)


# tsimcne object (using their default backbone resnet18)
tsimcne = TSimCNE(
    data_transform=None,  # Change this later
    total_epochs=[5, 1, 2],  # Low for testing
    batch_size=16,
    image_size=(224, 224),
)

# train
tsimcne.fit(full_dataset)

print(f"Time taken to train the model: {time.time()-t}")

# map the images to 2D vectors
vecs = tsimcne.transform(full_dataset)

print(f"Time taken to get the vectors: {time.time()-t}")

# save the output vectors
torch.save(vecs, "embeddings/vecs_tsimcne_resnet18.pt")
