import json
from enum import Enum
from fnmatch import fnmatch
from functools import partial

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchdata.datapipes.iter import FileOpener


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
    mask: bool = True,
    uuid: bool = False,
):
    fname, fcontent = data
    id = label2id[fname.split("/")[-2]]
    if mask:
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
    if uuid:
        return img_array, id, fname
    else:
        return img_array, id


def datapipe_datamodule(
    paths: list,
    padding: Padding,
    ignore_mix: bool = True,
    mask: bool = True,
    uuid: bool = False,
):
    fileopener = FileOpener(paths, mode="b")
    datapipe = fileopener.load_from_zip()
    datapipe = datapipe.filter(partial(img_filter, ignore_mix=ignore_mix))
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id_whoi.json") as f:
        label2id = json.load(f)
    with open("labeled_classes_whoi.json") as f:
        labeled = json.load(f)

    datapipe = datapipe.map(
        partial(
            parse_datamodule,
            label2id=label2id,
            labeled=labeled,
            padding=padding,
            mask=mask,
            uuid=uuid,
        )
    )
    return datapipe


def make_data(
    paths: list,
    padding: Padding,
    ignore_mix: bool = True,
    mask: bool = True,
    uuid: bool = False,
):
    datapipe = datapipe_datamodule(
        paths,
        padding,
        ignore_mix,
        mask,
        uuid,
    )
    data = dict()
    idx = 0
    if uuid:
        for img, id, fname in datapipe:
            data[idx] = (img, id, fname)
            idx += 1
    else:
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
        uuid: bool = False,
    ):
        self.data = data
        self.transforms = transforms
        self.uuid = uuid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.uuid:
            img, id, fname = self.data[idx]
            img = self.transforms(image=img)["image"]
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img, id, fname
        else:
            img, id = self.data[idx]
            img = self.transforms(image=img)["image"]
            img = torch.from_numpy(img).permute(2, 0, 1)
        return img, id


class PlanktonDataModule(L.LightningDataModule):
    def __init__(
        self,
        data: dict,
        train_transforms,
        test_transforms,
        batch_size: int = 2048,
        uuid: bool = False,
    ):
        super().__init__()
        self.data = data
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.uuid = uuid

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = TrainDatasetDataModule(
                self.data,
                self.train_transforms,
            )
        if stage == "predict" or stage is None:
            self.test_dataset = TestDatasetDataModule(
                self.data,
                self.test_transforms,
                self.uuid,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
        )


class CUBDataset(Dataset):
    def __init__(
        self,
        path: str,
        transforms,
        uuid: bool = False,
        split: str = "train",
        mode: str = "train",
    ):
        self.path = path
        self.transforms = transforms
        self.uuid = uuid
        self.split = split
        self.mode = mode
        with open(f"{path}/classes.txt") as f:
            self.label2id = {
                line.split(" ")[1].strip(): int(line.split(" ")[0]) for line in f
            }
        with open(f"{path}/images.txt") as f:
            self.images = {
                int(line.split(" ")[0]): line.split(" ")[1].strip() for line in f
            }
        with open(f"{path}/image_class_labels.txt") as f:
            self.labels = {
                int(line.split(" ")[0]): int(line.split(" ")[1]) for line in f
            }
        with open(f"{path}/train_test_split.txt") as f:
            self.labeled = np.array([int(line.split(" ")[1]) for line in f])
        with open(f"labeled_classes_cub.json") as f:
            self.labeled_classes = json.load(f)

    def __len__(self):
        if self.split == "train":
            return len(self.labeled[self.labeled == 1])
        elif self.split == "test":
            return len(self.labeled[self.labeled == 0])
        else:
            return len(self.labeled)

    def __getitem__(self, idx):
        if self.split == "train":
            idx = np.where(self.labeled == 1)[0][idx]
        elif self.split == "test":
            idx = np.where(self.labeled == 0)[0][idx]
        else:
            idx = idx

        image = self.images[idx + 1]
        label = self.labels[idx + 1]
        with Image.open(f"{self.path}/images/{image}") as img:
            img_array = np.array(img)
        if img_array.shape[0] > 256 or img_array.shape[1] > 256:
            img_array = A.LongestMaxSize(max_size=256)(image=img_array)["image"]
        img_array = A.PadIfNeeded(
            img_array.shape[1],
            img_array.shape[0],
            border_mode=4,
        )(image=img_array)["image"]
        if len(img_array.shape) == 2:
            img_array = A.ToRGB()(image=img_array)["image"]
        if self.mode == "train":
            img_array1 = self.transforms(image=img_array)["image"]
            img_array2 = self.transforms(image=img_array)["image"]
            img_array1 = torch.from_numpy(img_array1).permute(2, 0, 1)
            img_array2 = torch.from_numpy(img_array2).permute(2, 0, 1)
            if self.labeled[idx] == 1 and label in self.labeled_classes:
                return img_array1, img_array2, label
            else:
                return img_array1, img_array2, -1
        else:
            img_array = self.transforms(image=img_array)["image"]
            img_array = torch.from_numpy(img_array).permute(2, 0, 1)
            if self.uuid:
                return img_array, label, image
            else:
                return img_array, label


class CUBDataModule(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        train_transforms,
        test_transforms,
        batch_size: int = 2048,
        uuid: bool = False,
        split: str = "all",
    ):
        super().__init__()
        self.path = path
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.uuid = uuid
        self.split = split

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CUBDataset(
                self.path,
                self.train_transforms,
                False,
                self.split,
                mode="train",
            )
        if stage == "predict" or stage is None:
            self.test_dataset = CUBDataset(
                self.path,
                self.test_transforms,
                self.uuid,
                self.split,
                mode="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
        )


class MuCUBDataset(Dataset):
    def __init__(
        self,
        path: str,
        transforms_teacher,
        transforms_student,
        uuid: bool = False,
        split: str = "train",
        mode: str = "train",
    ):
        self.path = path
        self.transforms_teacher = transforms_teacher
        self.transforms_student = transforms_student
        self.uuid = uuid
        self.split = split
        self.mode = mode
        with open(f"{path}/classes.txt") as f:
            self.label2id = {
                line.split(" ")[1].strip(): int(line.split(" ")[0]) for line in f
            }
        with open(f"{path}/images.txt") as f:
            self.images = {
                int(line.split(" ")[0]): line.split(" ")[1].strip() for line in f
            }
        with open(f"{path}/image_class_labels.txt") as f:
            self.labels = {
                int(line.split(" ")[0]): int(line.split(" ")[1]) for line in f
            }
        with open(f"{path}/train_test_split.txt") as f:
            self.labeled = np.array([int(line.split(" ")[1]) for line in f])
        with open(f"labeled_classes_cub.json") as f:
            self.labeled_classes = json.load(f)

    def __len__(self):
        if self.split == "train":
            return len(self.labeled[self.labeled == 1])
        elif self.split == "test":
            return len(self.labeled[self.labeled == 0])
        else:
            return len(self.labeled)

    def __getitem__(self, idx):
        if self.split == "train":
            idx = np.where(self.labeled == 1)[0][idx]
        elif self.split == "test":
            idx = np.where(self.labeled == 0)[0][idx]
        else:
            idx = idx

        image = self.images[idx + 1]
        label = self.labels[idx + 1]
        with Image.open(f"{self.path}/images/{image}") as img:
            img_array = np.array(img)
        if img_array.shape[0] > 256 or img_array.shape[1] > 256:
            img_array = A.LongestMaxSize(max_size=256)(image=img_array)["image"]
        img_array = A.PadIfNeeded(
            img_array.shape[1],
            img_array.shape[0],
            border_mode=4,
        )(image=img_array)["image"]
        if len(img_array.shape) == 2:
            img_array = A.ToRGB()(image=img_array)["image"]
        if self.mode == "train":
            img_array1 = self.transforms_teacher(image=img_array)["image"]
            img_array2 = self.transforms_student(image=img_array)["image"]
            img_array1 = torch.from_numpy(img_array1).permute(2, 0, 1)
            img_array2 = torch.from_numpy(img_array2).permute(2, 0, 1)
            if self.labeled[idx] == 1 and label in self.labeled_classes:
                return img_array1, img_array2, label
            else:
                return img_array1, img_array2, -1
        else:
            img_array = self.transforms_teacher(image=img_array)["image"]
            img_array = torch.from_numpy(img_array).permute(2, 0, 1)
            if self.uuid:
                return img_array, label, image
            else:
                return img_array, label


class MuCUBDataModule(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        train_transforms_teacher,
        train_transforms_student,
        test_transforms,
        batch_size: int = 2048,
        uuid: bool = False,
        split: str = "all",
    ):
        super().__init__()
        self.path = path
        self.train_transforms_teacher = train_transforms_teacher
        self.train_transforms_student = train_transforms_student
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.uuid = uuid
        self.split = split

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MuCUBDataset(
                self.path,
                self.train_transforms_teacher,
                self.train_transforms_student,
                False,
                self.split,
                mode="train",
            )
        if stage == "predict" or stage is None:
            self.test_dataset = MuCUBDataset(
                self.path,
                self.test_transforms,
                self.test_transforms,
                self.uuid,
                self.split,
                mode="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )

    def predict_dataloader(self):
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
