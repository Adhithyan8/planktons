import json
from enum import Enum
from fnmatch import fnmatch
from functools import partial
from time import time

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torchdata.datapipes.iter import FileOpener
from torch.utils.data import DataLoader


class Padding(Enum):
    CONSTANT = 1
    REFLECT = 2


CONTRASTIVE_TRANSFORM = A.Compose(
    [
        A.ShiftScaleRotate(p=0.5),
        A.RandomResizedCrop(128, 128, scale=(0.2, 1.0)),
        A.Flip(p=0.5),
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.AdvancedBlur(),
            ],
        ),
        A.ToRGB(),
        A.Normalize(),
    ]
)


def img_filter(data, ignore_mix: bool):
    fname, _ = data
    if ignore_mix:
        return fnmatch(fname, "*.png") and fname.split("/")[-2] != "mix"
    else:
        return fnmatch(fname, "*.png")


def parse_contrastive(
    data,
    label2id: dict,
    labeled: list,
    transforms,
    padding: Padding,
):
    fname, fcontent = data
    id = label2id[fname.split("/")[-2]]
    if id not in labeled or fname.split("/")[-3] == "2014":
        id = -1
    with Image.open(fcontent) as img:
        img_array = np.array(img)
    if img_array.shape[0] > 256 or img_array.shape[1] > 256:
        img_array = A.LongestMaxSize(max_size=256)(image=img_array)["image"]
    if padding == Padding.CONSTANT:
        img_array = A.PadIfNeeded(
            img_array.shape[1],
            img_array.shape[0],
            border_mode=0,
            value=200,
        )(image=img_array)["image"]
    elif padding == Padding.REFLECT:
        img_array = A.PadIfNeeded(
            img_array.shape[1],
            img_array.shape[0],
            border_mode=4,
        )(image=img_array)["image"]
    img_array1 = transforms(image=img_array)["image"]
    img_array2 = transforms(image=img_array)["image"]
    img_array1 = torch.from_numpy(img_array1).permute(2, 0, 1)
    img_array2 = torch.from_numpy(img_array2).permute(2, 0, 1)
    return img_array1, img_array2, id


def parse_just_load(
    data,
    label2id: dict,
    labeled: list,
    padding: Padding,
):
    fname, fcontent = data
    id = label2id[fname.split("/")[-2]]
    if id not in labeled or fname.split("/")[-3] == "2014":
        id = -1
    with Image.open(fcontent) as img:
        img_array = np.array(img)
    if img_array.shape[0] > 256 or img_array.shape[1] > 256:
        img_array = A.LongestMaxSize(max_size=256)(image=img_array)["image"]
    if padding == Padding.CONSTANT:
        img_array = A.PadIfNeeded(
            img_array.shape[1],
            img_array.shape[0],
            border_mode=0,
            value=200,
        )(image=img_array)["image"]
    elif padding == Padding.REFLECT:
        img_array = A.PadIfNeeded(
            img_array.shape[1],
            img_array.shape[0],
            border_mode=4,
        )(image=img_array)["image"]
    return img_array, id


def contrastive_datapipe(
    paths,
    transforms,
    padding,
    ignore_mix=True,
):
    fileopener = FileOpener(paths, mode="b")
    datapipe = fileopener.load_from_zip()
    datapipe = datapipe.filter(partial(img_filter, ignore_mix=ignore_mix))
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id.json") as f:
        label2id = json.load(f)
    with open("labeled_classes.json") as f:
        labeled = json.load(f)

    datapipe = datapipe.map(
        partial(
            parse_contrastive,
            label2id=label2id,
            labeled=labeled,
            transforms=transforms,
            padding=padding,
        )
    )
    return datapipe


def just_load_to_ram(
    paths,
    padding,
    ignore_mix=True,
):
    fileopener = FileOpener(paths, mode="b")
    datapipe = fileopener.load_from_zip()
    datapipe = datapipe.filter(partial(img_filter, ignore_mix=ignore_mix))
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id.json") as f:
        label2id = json.load(f)
    with open("labeled_classes.json") as f:
        labeled = json.load(f)

    datapipe = datapipe.map(
        partial(
            parse_just_load,
            label2id=label2id,
            labeled=labeled,
            padding=padding,
        )
    )
    return datapipe


class JustLoad(torch.utils.data.Dataset):
    def __init__(self, paths, padding, transforms, ignore_mix=True):
        self.datapipe = just_load_to_ram(paths, padding, ignore_mix)
        self.transforms = transforms
        self.data = dict()
        idx = 0
        for img, id in self.datapipe:
            self.data[idx] = (img, id)
            idx += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, id = self.data[idx]
        img1 = self.transforms(image=img)["image"]
        img2 = self.transforms(image=img)["image"]
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        return img1, img2, id


# just load
dataset = JustLoad(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    padding=Padding.REFLECT,
    transforms=CONTRASTIVE_TRANSFORM,
    ignore_mix=True,
)


# # transforms and dataloaders
# datapipe = contrastive_datapipe(
#     [
#         "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
#         "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
#     ],
#     transforms=CONTRASTIVE_TRANSFORM,
#     padding=Padding.REFLECT,
#     ignore_mix=True,
# )

for w in [2, 4, 8, 16, 32]:
    dataloader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=w,
    )

    start = time()
    for img1, img2, id in dataloader:
        data = (img1.shape, img2.shape, id.shape)

    print(f"Workers: {w}, Time: {time() - start}s")
