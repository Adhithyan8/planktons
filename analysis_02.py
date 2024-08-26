import copy

import openTSNE
import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *
from losses import DINOLoss
from model import CosineClassifier
import matplotlib.pyplot as plt
import pandas as pd

# continuing with this precision setting
torch.set_float32_matmul_precision("high")


# nearly a one-to-one copy from lightly examples
class DINO(L.LightningModule):
    def __init__(self, output_dim=256):  # TODO: try fixed vs variable output_dim
        super(DINO, self).__init__()
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.student_backbone = backbone
        self.student_head = CosineClassifier(768, output_dim)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = CosineClassifier(768, output_dim)
        self.criterion = DINOLoss(output_dim=output_dim)

    def forward(self, x):
        y = self.student_backbone(x)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x)
        z = self.teacher_head(y)
        return z

    def predict_step(self, batch, batch_idx):
        x, y, n = batch
        z = self.teacher_backbone(x)
        return z, y, n


# lets define the transforms
def data_transform(img, label):
    # work around to maintain aspect ratio with albumentations
    with Image.open(img) as img:
        img = np.array(img)
    if img.shape[0] > 256 or img.shape[1] > 256:
        img = A.LongestMaxSize(max_size=256)(image=img)["image"]
    img = A.PadIfNeeded(img.shape[1], img.shape[0], border_mode=4)(image=img)["image"]
    img = A.Resize(256, 256)(image=img)["image"]
    # if grayscale, convert to 3 channels
    if len(img.shape) == 2:
        img = A.ToRGB()(image=img)["image"]
    img = A.CenterCrop(224, 224)(image=img)["image"]
    img = A.Normalize()(image=img)["image"]
    img = torch.tensor(img).permute(2, 0, 1)
    label = torch.tensor(label, dtype=torch.long)
    return img, label


# datasets to predict on
dataset = "PLANKTON"
trial = 0

# given the info, split and transform, make_dataset should give us the dataset
if dataset == "CUB":
    info = CUB_INFO
    out_dim = 230
elif dataset == "SCARS":
    info = SCARS_INFO
    out_dim = 230
    # shift labels to start from 0
    for sample in info:
        sample["label"] -= 1
elif dataset == "AIRCRAFT":
    info = AIRCRAFT_INFO
    out_dim = 110
elif dataset == "HERB19":
    info = HERB19_INFO
    out_dim = 700
    dist = HERB19_DIST
elif dataset == "PLANKTON":
    info = PLANKTON_INFO
    out_dim = 110
    dist = PLANKTON_DIST

model = DINO(output_dim=out_dim)

# load the trained model
model.load_state_dict(torch.load(f"outputs/exp_07_{dataset}_trial_{trial}.pt"))

trn_old_dataset = make_dataset(
    info, split_fit="train", split_cat="old", transform=data_transform
)
trn_new_dataset = make_dataset(
    info, split_fit="train", split_cat="new", transform=data_transform
)
tst_old_dataset = make_dataset(
    info, split_fit="test", split_cat="old", transform=data_transform
)
tst_new_dataset = make_dataset(
    info, split_fit="test", split_cat="new", transform=data_transform
)


def predict_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)


trn_old_dl = predict_dataloader(trn_old_dataset, 128)
trn_new_dl = predict_dataloader(trn_new_dataset, 128)
tst_old_dl = predict_dataloader(tst_old_dataset, 128)
tst_new_dl = predict_dataloader(tst_new_dataset, 128)

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    num_nodes=1,
)

trn_old_outs = trainer.predict(model, trn_old_dl)
trn_new_outs = trainer.predict(model, trn_new_dl)
tst_old_outs = trainer.predict(model, tst_old_dl)
tst_new_outs = trainer.predict(model, tst_new_dl)

# concatenate the embeddings
trn_old_out = torch.cat([out[0] for out in trn_old_outs]).cpu().numpy()
trn_new_out = torch.cat([out[0] for out in trn_new_outs]).cpu().numpy()
tst_old_out = torch.cat([out[0] for out in tst_old_outs]).cpu().numpy()
tst_new_out = torch.cat([out[0] for out in tst_new_outs]).cpu().numpy()

# labels
trn_old_lbl = torch.cat([out[1] for out in trn_old_outs]).cpu().numpy()
trn_new_lbl = torch.cat([out[1] for out in trn_new_outs]).cpu().numpy()
tst_old_lbl = torch.cat([out[1] for out in tst_old_outs]).cpu().numpy()
tst_new_lbl = torch.cat([out[1] for out in tst_new_outs]).cpu().numpy()

trn_old_name = [item for out in trn_old_outs for item in out[2]]
trn_new_name = [item for out in trn_new_outs for item in out[2]]
tst_old_name = [item for out in tst_old_outs for item in out[2]]
tst_new_name = [item for out in tst_new_outs for item in out[2]]

out = np.concatenate([trn_old_out, trn_new_out, tst_old_out, tst_new_out], axis=0)
lbl = np.concatenate([trn_old_lbl, trn_new_lbl, tst_old_lbl, tst_new_lbl], axis=0)
name = trn_old_name + trn_new_name + tst_old_name + tst_new_name

# top 6 large classes based on dist
top_classes = np.argsort(dist)[::-1][:6]

# tsne
affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    out,
    perplexities=[50, 500],
    metric="cosine",
    n_jobs=8,
    random_state=3,
)
init = openTSNE.initialization.pca(out, random_state=42)
embedding = openTSNE.TSNE(n_jobs=8).fit(
    affinities=affinities_multiscale_mixture,
    initialization=init,
)

df = pd.DataFrame(
    {
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "label": lbl,
        "pathname": name,
    }
)

# filter out the top 6 classes using lbl
df_top = df[df["label"].isin(top_classes)]
df_top.to_csv(f"outputs/tsne_{dataset}_top.csv", index=False)

label2labelname = {}
for sample in info:
    if sample["label"] in top_classes:
        label2labelname[sample["label"]] = sample["label_name"]

# plotting
plt.figure(figsize=(10, 10))
for i, (label, label_name) in enumerate(label2labelname.items()):
    plt.scatter(
        df_top[df_top["label"] == label]["x"],
        df_top[df_top["label"] == label]["y"],
        label=label_name,
        s=1,
        alpha=0.6,
        marker="o",
        c=f"C{i}",
    )
plt.legend()
plt.title(f"t-SNE: {dataset}")
plt.savefig(f"figures/tsne_{dataset}_top.png", dpi=600)
plt.close()
