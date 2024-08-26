import copy

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from data import make_dataset
from datasheet import *
from losses import DINOLoss
from model import CosineClassifier

# continuing with this precision setting
torch.set_float32_matmul_precision("high")


# nearly a one-to-one copy from lightly examples
class DINO(L.LightningModule):
    def __init__(self, output_dim=256):
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
        x, y = batch
        z = self.forward_teacher(x)
        return z, y

    def get_teacher_head(self):
        # get the linear layer of the teacher head
        weight = self.teacher_head.layer.weight
        # L2 normalize the weights
        weight = nn.functional.normalize(weight, p=2, dim=-1)
        return weight


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


def eRANK(embeddings):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    corr = np.dot(embeddings.T, embeddings) / embeddings.shape[0]
    eigvals = np.linalg.eigvals(corr)
    eigvals = (eigvals + 1e-6) / np.sum(eigvals)  # add 1e-6 to avoid log(0)
    entropy = -np.sum(eigvals * np.log(eigvals))
    eRANK = np.exp(entropy)
    return eRANK


# datasets to predict on
datasets = ["CUB", "HERB19", "PLANKTON"]
exp = "exp_07"
trial = 1

# 1 row and 3 columns for the subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for j, dataset in enumerate(datasets):
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
    elif dataset == "PLANKTON":
        info = PLANKTON_INFO
        out_dim = 110
    num_classes = NUM_CLASSES[dataset]

    model = DINO(output_dim=out_dim)

    # load the trained model
    model.load_state_dict(torch.load(f"outputs/{exp}_{dataset}_trial_{trial}.pt"))

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

    out = np.concatenate([trn_old_out, trn_new_out, tst_old_out, tst_new_out], axis=0)
    lbl = np.concatenate([trn_old_lbl, trn_new_lbl, tst_old_lbl, tst_new_lbl], axis=0)

    # out contains the logits
    prd = np.argmax(out, axis=1)

    # optimal assignment
    D = max(num_classes, out_dim)
    cost = np.zeros((D, D))
    for i in range(prd.shape[0]):
        cost[int(prd[i]), int(lbl[i])] += 1
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

    proto_2_lbl = {i: col_ind[i] for i in range(out_dim)}
    lbl_2_proto = {col_ind[i]: i for i in range(out_dim)}

    # pairwise cosine similarity of teacher head weights
    teacher_head = model.get_teacher_head().detach().cpu().numpy()
    # already L2 normalized
    teacher_head_sim = np.dot(teacher_head, teacher_head.T)
    # eRANK of teacher head weights
    eRANK_teacher_head = eRANK(teacher_head.T)

    # reorder the indices of sim - put seen proto first
    seen_lbl = []
    for sample in info:
        if sample["old"] == 1:
            if sample["label"] not in seen_lbl:
                seen_lbl.append(sample["label"])
    unseen_lbl = [lbl for lbl in range(out_dim) if lbl not in seen_lbl]

    # opt prd gives the class labels, while prd gives the prototype index
    seen_proto = [lbl_2_proto[lbl] for lbl in seen_lbl]
    unseen_proto = [lbl_2_proto[lbl] for lbl in unseen_lbl]

    seen_proto.sort()
    unseen_proto.sort()

    # reorder the similarity matrix
    reorder = seen_proto + unseen_proto
    teacher_head_sim = teacher_head_sim[reorder][:, reorder]

    print(f"eRANK: {eRANK_teacher_head:.4f}")

    # plot the similarity matrix in the subplot
    ax = axs[j]
    ax.imshow(teacher_head_sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"{dataset}")
    if j == 2:
        ax.colorbar()

plt.suptitle(f"Pairwise cosine similarity of prototypes")
plt.savefig(f"figures/head_sim_{exp}.png")
plt.close()