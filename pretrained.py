import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import AutoImageProcessor, ResNetModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
train_dataset = load_dataset(
    "/local_storage/users/adhkal/planktons_dataset", "2013-14", split="train"
)
# torch format
train_dataset = train_dataset.with_format("torch")
# convert to 3 channels
train_dataset = train_dataset.map(
    lambda x: {"image": x["image"].repeat(3, 1, 1), "label": x["label"]},
)

# load pretrained model
feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-152")
# preprocess
train_dataset = train_dataset.map(
    lambda x: {"image": feature_extractor(x["image"]), "label": x["label"]},
)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

# save the output vectors
vecs = torch.empty((0, 2048))
tags = torch.empty((0,))

# model
model = ResNetModel.from_pretrained("microsoft/resnet-152")
model.to(device).eval()

for batch in train_loader:
    images = batch["image"]["pixel_values"].squeeze(1)
    tags = torch.cat((tags, batch["label"]), dim=0)
    with torch.no_grad():
        outputs = model(images.to(device)).pooler_output.squeeze()  # (B, 2048)
        vecs = torch.cat((vecs, outputs.cpu()), dim=0)

# save the vectors
torch.save(vecs, "tensors/resnet_vecs.pt")
torch.save(tags, "tensors/tags.pt")
