from fnmatch import fnmatch
from PIL import Image
import json
import numpy as np
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileOpener, IterDataPipe
from torchvision.transforms import PILToTensor


def expand2square(pil_img, background_color):
    """
    To resize the images to 224x224, and convert to RGB
    """
    # convert pil_img to "rgb"
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    # resize to thumbnail (224, 224)
    pil_img.thumbnail((224, 224), Image.Resampling.LANCZOS)
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

    def __iter__(self) -> IterDataPipe:
        yield from self.source_datapipe

    def __len__(self) -> int:
        return self.length


def get_datapipe(path, num_images, transforms, ignore_mix=True, padding=False):
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
        if padding:
            img_pil = Image.open(file_content)
            img_pil = img_pil.convert("RGB")
            img_pil = expand2square(img_pil, (200, 200, 200))
            img_tensor = PILToTensor()(img_pil).float()
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

            def __iter__(self) -> IterDataPipe:
                yield from self.source_datapipe

            def __len__(self) -> int:
                return self.length

    datapipe = datapipe.set_length(num_images)

    return datapipe
