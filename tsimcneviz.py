import torch
from tsimcne.tsimcne import TSimCNE
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    Dinov2Model,
    ResNetModel,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
full_dataset = full_dataset.with_format("torch")
# convert to 3 channels
full_dataset = full_dataset.map(
    lambda x: {"image": x["image"].repeat(3, 1, 1), "label": x["label"]},
)

# load pretrained model (pick one)
feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
# feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

# preprocess
full_dataset = full_dataset.map(
    lambda x: {"image": feature_extractor(x["image"]), "label": x["label"]},
)
# format to contain img, label tuples
full_dataset = full_dataset.map(
    lambda x: (x["image"]["pixel_values"].squeeze(1), x["label"]),
    remove_columns=["image", "label"],
)

# model
model = ResNetModel.from_pretrained("microsoft/resnet-18")
print(model.config)
# model = Dinov2Model.from_pretrained("facebook/dinov2-base")

# tsimcne object
tsimcne = TSimCNE(
    backbone=model,
    data_transform=None,  # Change this later
    total_epochs=[500, 50, 250],
    batch_size=256,
)

# train
tsimcne.fit(full_dataset)

# map the images to 2D vectors
vecs = tsimcne.transform(full_dataset)

# save the output vectors
torch.save(vecs, "embeddings/vecs_tsimcne_resnet18.pt")
