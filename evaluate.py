import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from utils import get_datapipe

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

test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TRAIN,
    transforms=test_transform,
    ignore_mix=True,
    padding=True,
)
test_datapipe = get_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    num_images=NUM_TEST,
    transforms=test_transform,
    ignore_mix=True,
    padding=True,
)
train_dataloader = DataLoader(
    train_datapipe, batch_size=512, shuffle=False, num_workers=8
)
test_dataloader = DataLoader(
    test_datapipe, batch_size=512, shuffle=False, num_workers=8
)

backbone = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=False)
backbone.fc = torch.nn.Identity()

projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 128),
)

for config in ["finetune", "fulltrain"]:
    for outspace in ["backbone", "head"]:
        # load state dict
        backbone.load_state_dict(torch.load(f"{config}_resnet18_backbone.pth"))
        projection_head.load_state_dict(torch.load(f"{config}_resnet18_head.pth"))

        if outspace == "backbone":
            model = backbone
        else:
            model = torch.nn.Sequential(backbone, projection_head)
        model.eval()

        # store output
        output = torch.empty((0, 512))
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

        output_train = output[:NUM_TRAIN]
        output_test = output[NUM_TRAIN:]

        # knn
        if outspace == "backbone":
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=8, metric="cosine")
        else:
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=8, metric="minkowski")
        knn.fit(output_train, labels[:NUM_TRAIN])

        # test
        preds = knn.predict(output_test)
        f1 = f1_score(labels[NUM_TRAIN:], preds, average="macro")
        acc = accuracy_score(labels[NUM_TRAIN:], preds)

        print(f"{config}_{outspace}")

        print(f"Accuracy: {acc}")
        print(f"F1 score: {f1}")

        # class wise F1 score
        for i in range(int(max(labels)) + 1):
            mask = labels[NUM_TRAIN:] == i
            f1 = f1_score(labels[NUM_TRAIN:][mask], preds[mask], average="macro")
            print(f"F1 score for class {i}: {f1}")
