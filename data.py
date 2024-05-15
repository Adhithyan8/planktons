import json
from enum import Enum
from fnmatch import fnmatch
from functools import partial

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torchdata.datapipes.iter import FileOpener
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L


class Padding(Enum):
    CONSTANT = 1
    REFLECT = 2


def img_filter(data, ignore_mix: bool):
    fname, _ = data
    if ignore_mix:
        return fnmatch(fname, "*.png") and fname.split("/")[-2] != "mix"
    else:
        return fnmatch(fname, "*.png")


def parse_datamodule(
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


def datapipe_datamodule(
    paths: list,
    padding: Padding,
    ignore_mix: bool = True,
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
            parse_datamodule,
            label2id=label2id,
            labeled=labeled,
            padding=padding,
        )
    )
    return datapipe


def make_data(
    paths: list,
    padding: Padding,
    ignore_mix: bool = True,
):
    datapipe = datapipe_datamodule(
        paths,
        padding,
        ignore_mix,
    )
    data = dict()
    idx = 0
    for img, id in datapipe:
        data[idx] = (img, id)
        idx += 1
    return data


class TrainDatasetDataModule(Dataset):
    def __init__(
        self,
        data: dict,
        transforms,
    ):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, id = self.data[idx]
        img1 = self.transforms(image=img)["image"]
        img2 = self.transforms(image=img)["image"]
        img1 = torch.from_numpy(img1).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        return img1, img2, id


class TestDatasetDataModule(Dataset):
    def __init__(
        self,
        data: dict,
        transforms,
    ):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, id = self.data[idx]
        img = self.transforms(image=img)["image"]
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, id


class PlanktonDataModule(L.LightningDataModule):
    def __init__(
        self,
        paths: list,
        train_transforms,
        test_transforms,
        padding: Padding,
        ignore_mix: bool = True,
        batch_size: int = 2048,
    ):
        super().__init__()
        self.paths = paths
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.padding = padding
        self.ignore_mix = ignore_mix
        self.batch_size = batch_size

    def prepare_data(self):
        self.data = make_data(
            self.paths,
            self.padding,
            self.ignore_mix,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TrainDatasetDataModule(
                self.data,
                self.train_transforms,
            )
        if stage == "test" or stage is None:
            self.test_dataset = TestDatasetDataModule(
                self.data,
                self.test_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
        )


"""
BELOW ARE NOT USED IN THE FINAL IMPLEMENTATION
"""


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


def datapipe_contrastive(
    paths: list,
    transforms,
    padding: Padding,
    ignore_mix: bool = True,
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


def parse_inference(
    data,
    label2id: dict,
    transforms,
    padding: Padding,
):
    fname, fcontent = data
    id = label2id[fname.split("/")[-2]]
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
    img_array = transforms(image=img_array)["image"]
    img_array = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_array, id


def datapipe_inference(
    paths: list,
    transforms,
    padding: Padding,
    ignore_mix: bool = True,
):
    fileopener = FileOpener(paths, mode="b")
    datapipe = fileopener.load_from_zip()
    datapipe = datapipe.filter(partial(img_filter, ignore_mix=ignore_mix))
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id.json") as f:
        label2id = json.load(f)

    datapipe = datapipe.map(
        partial(
            parse_inference,
            label2id=label2id,
            transforms=transforms,
            padding=padding,
        )
    )
    return datapipe
