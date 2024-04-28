import json
from enum import Enum
from fnmatch import fnmatch

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileOpener, IterDataPipe


class Padding(Enum):
    CONSTANT = 1
    REFLECT = 2


@functional_datapipe("set_length")
class LengthSetterIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, length: int) -> None:
        self.source_datapipe = source_datapipe
        assert length >= 0
        self.length = length

    def __iter__(self) -> IterDataPipe:  # type: ignore
        yield from self.source_datapipe

    def __len__(self) -> int:
        return self.length


def inference_datapipe(path, num_images, transforms, padding, ignore_mix=True):
    fileopener = FileOpener(path, mode="b")
    datapipe = fileopener.load_from_zip()  # recommended to load from zip

    def image_filter(data):
        file_name, _ = data
        if ignore_mix:
            return fnmatch(file_name, "*.png") and file_name.split("/")[-2] != "mix"
        else:
            return fnmatch(file_name, "*.png")

    datapipe = datapipe.filter(image_filter)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id.json") as f:
        label2id = json.load(f)  # dictionary mapping class names to ids

    def parse_data(data):
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

    datapipe = datapipe.map(parse_data)
    if not hasattr(datapipe, "set_length"):

        @functional_datapipe("set_length")
        class LengthSetterIterDataPipe(IterDataPipe):
            def __init__(self, source_datapipe: IterDataPipe, length: int) -> None:
                self.source_datapipe = source_datapipe
                assert length >= 0
                self.length = length

            def __iter__(self) -> IterDataPipe:  # type: ignore
                yield from self.source_datapipe

            def __len__(self) -> int:
                return self.length

    datapipe = datapipe.set_length(num_images)
    return datapipe


def contrastive_datapipe(
    paths, num_images, transforms, padding, ignore_mix=True, mask_label=False
):
    fileopener = FileOpener(paths, mode="b")
    datapipe = fileopener.load_from_zip()

    def image_filter(data):
        file_name, _ = data
        if ignore_mix:
            return fnmatch(file_name, "*.png") and file_name.split("/")[-2] != "mix"
        else:
            return fnmatch(file_name, "*.png")

    datapipe = datapipe.filter(image_filter)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()

    with open("label2id.json") as f:
        label2id = json.load(f)  # dictionary mapping class names to ids
    with open("labelled_classes.json") as f:
        labelled_classes = json.load(f)  # list of labelled class ids

    def parse_data(data):
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

    datapipe = datapipe.map(parse_data)
    if not hasattr(datapipe, "set_length"):

        @functional_datapipe("set_length")
        class LengthSetterIterDataPipe(IterDataPipe):
            def __init__(self, source_datapipe: IterDataPipe, length: int) -> None:
                self.source_datapipe = source_datapipe
                assert length >= 0
                self.length = length

            def __iter__(self) -> IterDataPipe:  # type: ignore
                yield from self.source_datapipe

            def __len__(self) -> int:
                return self.length

    datapipe = datapipe.set_length(num_images)
    return datapipe
