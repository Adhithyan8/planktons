import matplotlib.pyplot as plt
import numpy as np
import openTSNE
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    CLIPVisionModelWithProjection,
    Dinov2Model,
    ResNetModel,
)

"""
Embed the images using pretrained models, so entirely unsupervised.
"""

model_name = "resnet50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}; Model: {model_name}")

# datasets
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

train_dataset = train_dataset.with_format("torch")
test_dataset = test_dataset.with_format("torch")

print("Dataset loaded!")

# # load pretrained model (pick one)
# if model_name == "resnet50":
#     feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# elif model_name == "dinov2":
#     feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# elif model_name == "clip":
#     feature_extractor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# else:
#     raise ValueError("Invalid model name")

# # preprocess (CLIP is different)
# if model_name == "resnet50" or model_name == "dinov2":
#     train_dataset = train_dataset.map(
#         lambda x: {
#             "image": feature_extractor(x["image"])["pixel_values"][0],
#             "label": x["label"],
#         },
#     )
#     test_dataset = test_dataset.map(
#         lambda x: {
#             "image": feature_extractor(x["image"])["pixel_values"][0],
#             "label": x["label"],
#         },
#     )
# elif model_name == "clip":
#     train_dataset = train_dataset.map(
#         lambda x: {"image": feature_extractor(images=x["image"]), "label": x["label"]},
#     )
#     test_dataset = test_dataset.map(
#         lambda x: {"image": feature_extractor(images=x["image"]), "label": x["label"]},
#     )
# print("Preprocessing done!")

# # dataloader
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# # model (pick one)
# if model_name == "resnet50":
#     model = ResNetModel.from_pretrained("microsoft/resnet-50")
# elif model_name == "dinov2":
#     model = Dinov2Model.from_pretrained("facebook/dinov2-base")
# elif model_name == "clip":
#     model = CLIPVisionModelWithProjection.from_pretrained(
#         "openai/clip-vit-base-patch32"
#     )
# else:
#     raise ValueError("Invalid model name")

# # save the output vectors
# if model_name == "resnet50":
#     vecs = torch.empty((0, 2048))
# elif model_name == "dinov2":
#     vecs = torch.empty((0, 512))
# elif model_name == "clip":
#     vecs = torch.empty((0, 512))
# tags = torch.empty((0,))

# # embed the images
# model.to(device).eval()
# if model_name == "resnet50" or model_name == "dinov2":
#     for batch in train_loader:
#         tags = torch.cat((tags, batch["label"]), dim=0)
#         with torch.no_grad():
#             outputs = model(batch["image"].to(device)).pooler_output.squeeze()
#             vecs = torch.cat((vecs, outputs.cpu()), dim=0)
#     for batch in test_loader:
#         tags = torch.cat((tags, batch["label"]), dim=0)
#         with torch.no_grad():
#             outputs = model(batch["image"].to(device)).pooler_output.squeeze()
#             vecs = torch.cat((vecs, outputs.cpu()), dim=0)
# elif model_name == "clip":
#     for batch in train_loader:
#         images = batch["image"]["pixel_values"].squeeze(1)
#         tags = torch.cat((tags, batch["label"]), dim=0)
#         with torch.no_grad():
#             outputs = model(images.to(device)).image_embeds.squeeze()
#             vecs = torch.cat((vecs, outputs.cpu()), dim=0)
#     for batch in test_loader:
#         images = batch["image"]["pixel_values"].squeeze(1)
#         tags = torch.cat((tags, batch["label"]), dim=0)
#         with torch.no_grad():
#             outputs = model(images.to(device)).image_embeds.squeeze()
#             vecs = torch.cat((vecs, outputs.cpu()), dim=0)

# # save the vectors
# np.save(f"embeddings/vec_{model_name}.npy", vecs.numpy())
# np.save(f"embeddings/tags.npy", tags.numpy())

# vecs = np.load(f"embeddings/vec_{model_name}.npy")
# tags = np.load("embeddings/tags.npy")

# # tsne embedding
# # perform tsne with multiscale embedding
# affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
#     vecs,
#     perplexities=[50, 500],
#     metric="cosine",
#     n_jobs=8,
#     random_state=3,
# )
# init = openTSNE.initialization.pca(vecs, random_state=42)
# embedding = openTSNE.TSNE(n_jobs=8).fit(
#     affinities=affinities_multiscale_mixture,
#     initialization=init,
# )

# # save the tsne embedding
# np.save(f"embeddings/tsne_{model_name}.npy", embedding)

# # plot the embedding (only train set)
# embed_train = embedding[: len(train_dataset)]
# plt.figure(figsize=(10, 10))
# plt.scatter(
#     embed_train[:, 0],
#     embed_train[:, 1],
#     c=tags[: len(train_dataset)],
#     cmap="Spectral",
#     s=0.1,
# )
# plt.axis("off")
# plt.savefig(f"figures/tsne_train_{model_name}.png", dpi=300)
# plt.close()

embedding = np.load(f"embeddings/tsne_{model_name}.npy")
tags = np.load("embeddings/tags.npy")

embed_train = embedding[: len(train_dataset)]
embed_test = embedding[len(train_dataset) :]

neighbours = [1, 3, 5, 15]

# knn classifier
for n in neighbours:
    knn = KNeighborsClassifier(n_neighbors=n, n_jobs=8)
    knn.fit(embed_train, tags[: len(train_dataset)])

    # test the classifier (class wise F1 score and overall F1 score)
    preds = knn.predict(embed_test)
    f1 = f1_score(
        tags[len(train_dataset) :], preds, average="macro"
    )  # unweighted average
    acc = accuracy_score(tags[len(train_dataset) :], preds)

    print(f"Accuracy: {acc}")
    print(f"F1 score: {f1}")

# # class wise F1 score
# for i in range(int(max(tags)) + 1):
#     mask = tags[len(train_dataset) :] == i
#     f1 = f1_score(tags[len(train_dataset) :][mask], preds[mask], average="macro")
#     print(f"F1 score for class {i}: {f1}")
