import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    CLIPVisionModelWithProjection,
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
train_dataset = train_dataset.with_format("torch")

# convert to 3 channels
train_dataset = train_dataset.map(
    lambda x: {"image": x["image"].repeat(3, 1, 1), "label": x["label"]},
)

# load pretrained model (pick one)
# feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-152")
# feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
feature_extractor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# preprocess (CLIP is different)
# train_dataset = train_dataset.map(
#     lambda x: {"image": feature_extractor(x["image"]), "label": x["label"]},
# )
train_dataset = train_dataset.map(
    lambda x: {"image": feature_extractor(images=x["image"]), "label": x["label"]},
)  # CLIPVisionModelWithProjection

# dataloader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# save the output vectors
# vecs = torch.empty((0, 2048))  # ResNetModel
vecs = torch.empty((0, 512))  # CLIPVisionModelWithProjection
# vecs = torch.empty((0, 768))  # other models
tags = torch.empty((0,))

# model (pick one)
# model = ResNetModel.from_pretrained("microsoft/resnet-152")
# model = Dinov2Model.from_pretrained("facebook/dinov2-base")
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")

model.to(device).eval()
for batch in train_loader:
    images = batch["image"]["pixel_values"].squeeze(1)
    tags = torch.cat((tags, batch["label"]), dim=0)
    with torch.no_grad():
        # outputs = model(images.to(device)).pooler_output.squeeze()  # (B, 2048)
        outputs = model(images.to(device)).image_embeds.squeeze()  # (B, 768) CLIP
        vecs = torch.cat((vecs, outputs.cpu()), dim=0)

# save the vectors
torch.save(tags, "embeddings/tags.pt")
# torch.save(vecs, "embeddings/vecs_resnet.pt")
# torch.save(vecs, "embeddings/vecs_dinov2.pt")
torch.save(vecs, "embeddings/vecs_clip.pt")
