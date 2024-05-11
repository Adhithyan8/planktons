import json
from enum import Enum
from fnmatch import fnmatch
from functools import partial

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torchdata.datapipes.iter import FileOpener


class Padding(Enum):
    CONSTANT = 1
    REFLECT = 2


def image_filter(data, ignore_mix):
    file_name, _ = data
    if ignore_mix:
        return fnmatch(file_name, "*.png") and file_name.split("/")[-2] != "mix"
    else:
        return fnmatch(file_name, "*.png")


def parse_data_inference(data, label2id, transforms, padding):
    file_name, file_content = data
    id = label2id[file_name.split("/")[-2]]
    img_array = np.array(Image.open(file_content))
    if padding == Padding.CONSTANT:
        img_array = A.PadIfNeeded(
            img_array.shape[1], img_array.shape[0], border_mode=0, value=200
        )(image=img_array)["image"]
    elif padding == Padding.REFLECT:
        img_array = A.PadIfNeeded(
            img_array.shape[1], img_array.shape[0], border_mode=4
        )(image=img_array)["image"]
    img_array = transforms(image=img_array)["image"]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img_tensor, id


def parse_data_contrastive(
    data, label2id, labelled_classes, transforms, padding, mask_label=False
):
    file_name, file_content = data
    if not mask_label:
        id = label2id[file_name.split("/")[-2]]
    else:
        if (
            label2id[file_name.split("/")[-2]] in labelled_classes
            and file_name.split("/")[-3] == "2013"
        ):
            id = label2id[file_name.split("/")[-2]]
        else:
            id = -1
    img_array = np.array(Image.open(file_content))
    if padding == Padding.CONSTANT:
        img_array = A.PadIfNeeded(
            img_array.shape[1], img_array.shape[0], border_mode=0, value=200
        )(image=img_array)["image"]
    elif padding == Padding.REFLECT:
        img_array = A.PadIfNeeded(
            img_array.shape[1], img_array.shape[0], border_mode=4
        )(image=img_array)["image"]

    img_array1 = transforms(image=img_array)["image"]
    img_array2 = transforms(image=img_array)["image"]
    img_tensor1 = torch.from_numpy(img_array1).permute(2, 0, 1)
    img_tensor2 = torch.from_numpy(img_array2).permute(2, 0, 1)
    return img_tensor1, img_tensor2, id


def inference_datapipe(path, num_images, transforms, padding, ignore_mix=True):
    fileopener = FileOpener(path, mode="b")
    datapipe = fileopener.load_from_zip()  # recommended to load from zip
    datapipe = datapipe.filter(partial(image_filter, ignore_mix=ignore_mix))
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id.json") as f:
        label2id = json.load(f)  # dictionary mapping class names to ids

    datapipe = datapipe.map(
        partial(
            parse_data_inference,
            label2id=label2id,
            transforms=transforms,
            padding=padding,
        )
    )
    return datapipe


def contrastive_datapipe(
    paths, num_images, transforms, padding, ignore_mix=True, mask_label=False
):
    fileopener = FileOpener(paths, mode="b")
    datapipe = fileopener.load_from_zip()
    datapipe = datapipe.filter(partial(image_filter, ignore_mix=ignore_mix))
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id.json") as f:
        label2id = json.load(f)  # dictionary mapping class names to ids
    with open("labelled_classes.json") as f:
        labelled_classes = json.load(f)  # list of labelled class ids

    datapipe = datapipe.map(
        partial(
            parse_data_contrastive,
            label2id=label2id,
            labelled_classes=labelled_classes,
            transforms=transforms,
            padding=padding,
            mask_label=mask_label,
        )
    )
    return datapipe
