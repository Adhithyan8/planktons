from fnmatch import fnmatch
from PIL import Image
import json
import numpy as np
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileOpener, IterDataPipe
from torchvision.transforms import PILToTensor
import albumentations as A


def expand2square(pil_img, background_color):
    """
    To resize the images to 224x224, and convert to RGB
    """
    # resize to thumbnail (224, 224)
    pil_img.thumbnail((224, 224), Image.Resampling.BILINEAR)
    # create new image of desired size and color
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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


def get_datapipe(path, num_images, transforms, ignore_mix=True, padding="none"):
    dataset_path = path
    fileopener = FileOpener([dataset_path], mode="b")
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

    # get label dictionary
    with open("labels.json") as f:
        label2id = json.load(f)

    def parse_data(data):
        file_name, file_content = data
        id = label2id[file_name.split("/")[-2]]
        if padding=="constant":
            img_pil = Image.open(file_content)
            img_pil = img_pil.convert("RGB")
            img_pil = expand2square(img_pil, (200, 200, 200))
            img_tensor = PILToTensor()(img_pil).float()
        elif padding=="reflect":
            img_array = np.array(Image.open(file_content).convert("RGB"))
            img_array = A.PadIfNeeded(img_array.shape[1], img_array.shape[0])(
                image=img_array
            )["image"]
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
        else:
            img_array = np.array(Image.open(file_content))
            if img_array.ndim < 3:
                img_array = np.repeat(img_array[..., np.newaxis], 3, -1)

            img_tensor = torch.from_numpy(img_array).float()
            img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.div(255)
        img_tensor = transforms(img_tensor)
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


def contrastive_datapipe(paths, num_images, transforms, ignore_mix=True):
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

    # get label dictionary
    with open("labels.json") as f:
        label2id = json.load(f)

    def parse_data(data):
        file_name, file_content = data
        id = label2id[file_name.split("/")[-2]]
        img_array = np.array(Image.open(file_content).convert("RGB"))
        img_array = A.PadIfNeeded(img_array.shape[1], img_array.shape[0])(
            image=img_array
        )["image"]
        img_tensor = torch.from_numpy(img_array).float().div(255).permute(2, 0, 1)
        img_tensor_1 = transforms(img_tensor)
        img_tensor_2 = transforms(img_tensor)
        return img_tensor_1, img_tensor_2, id

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


class InfoNCECosine(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCECosine, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        batch_size = features.shape[0] // 2

        z1 = torch.nn.functional.normalize(features[:batch_size], dim=1)
        z2 = torch.nn.functional.normalize(features[batch_size:], dim=1)

        cos_z1_z1 = z1 @ z1.T / self.temperature
        cos_z2_z2 = z2 @ z2.T / self.temperature
        cos_z1_z2 = z1 @ z2.T / self.temperature

        pos = cos_z1_z2.trace() / batch_size

        # mask out the diagonal elements with float(-inf)
        mask = torch.eye(batch_size, device=features.device).bool()
        cos_z1_z1 = cos_z1_z1.masked_fill_(mask, float("-inf"))
        cos_z2_z2 = cos_z2_z2.masked_fill_(mask, float("-inf"))

        logsumexp_1 = torch.hstack((cos_z1_z1, cos_z1_z2)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_z1_z2.T, cos_z2_z2)).logsumexp(dim=1).mean()

        neg = (logsumexp_1 + logsumexp_2) / 2
        loss = -(pos - neg)
        return loss
