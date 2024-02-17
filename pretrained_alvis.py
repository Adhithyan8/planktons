import matplotlib.pyplot as plt
import openTSNE
import torch
import torch.hub
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import get_datapipe

model_name = "resnet18"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676

train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TRAIN,
    transforms=train_transform,
    ignore_mix=True,
)
test_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TEST,
    transforms=test_transform,
    ignore_mix=True,
)
train_dataloader = DataLoader(
    train_datapipe, batch_size=512, shuffle=False, num_workers=8
)
test_dataloader = DataLoader(
    test_datapipe, batch_size=512, shuffle=False, num_workers=8
)

# model (pick one)
if model_name == "resnet18":
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
elif model_name == "resnet50":
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
elif model_name == "dinov2":
    model = torch.hub.load("facebookresearch/dino:main", "dino_v2_8")
    model = model.eval()
else:
    raise ValueError("Invalid model name")

# store output
if model_name == "resnet18":
    output = torch.empty((0, 512))
elif model_name == "resnet50":
    output = torch.empty((0, 2048))
elif model_name == "dinov2":
    output = torch.empty((0, 384))
labels = torch.empty((0,))

model.to(device)
for images, labels_batch in train_dataloader:
    images = images.to(device)
    with torch.no_grad():
        output_batch = model(images)
    output = torch.cat((output, output_batch.cpu()))
    labels = torch.cat((labels, labels_batch))
for images, labels_batch in test_dataloader:
    images = images.to(device)
    with torch.no_grad():
        output_batch = model(images)
    output = torch.cat((output, output_batch.cpu()))
    labels = torch.cat((labels, labels_batch))

# to numpy
output = output.numpy()
labels = labels.numpy()

# tsne
affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    output,
    perplexities=[50, 500],
    metric="cosine",
    n_jobs=8,
    random_state=3,
)
init = openTSNE.initialization.pca(output, random_state=42)
embedding = openTSNE.TSNE(n_jobs=8).fit(
    affinities=affinities_multiscale_mixture,
    initialization=init,
)

# plot the embedding (only train set)
embed_train = embedding[:NUM_TRAIN]
plt.figure(figsize=(10, 10))
plt.scatter(
    embed_train[:, 0],
    embed_train[:, 1],
    c=labels[:NUM_TRAIN],
    cmap="Spectral",
    s=0.1,
)
plt.axis("off")
plt.savefig(f"figures/tsne_train_{model_name}.png", dpi=300)
plt.close()

embed_test = embedding[NUM_TRAIN:]
# knn
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=8)
knn.fit(embed_train, labels[:NUM_TRAIN])

# test the classifier (class wise F1 score and overall F1 score)
preds = knn.predict(embed_test)
f1 = f1_score(labels[NUM_TRAIN:], preds, average="macro")  # unweighted average
acc = accuracy_score(labels[NUM_TRAIN:], preds)

print(f"Accuracy: {acc}")
print(f"F1 score: {f1}")

# class wise F1 score
for i in range(int(max(labels)) + 1):
    mask = labels[NUM_TRAIN:] == i
    f1 = f1_score(labels[NUM_TRAIN:][mask], preds[mask], average="macro")
    print(f"F1 score for class {i}: {f1}")
