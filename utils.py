from fnmatch import fnmatch
from PIL import Image
import json
import numpy as np
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileOpener, IterDataPipe


@functional_datapipe('set_length')
class LengthSetterIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, length: int) -> None:
        self.source_datapipe = source_datapipe
        assert length >= 0
        self.length = length

    def __iter__(self) -> IterDataPipe:
        yield from self.source_datapipe

    def __len__(self) -> int:
        return self.length
        

def get_datapipe(path, num_images, transforms, ignore_mix=True):
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

        img_array = np.array(Image.open(file_content))
        if img_array.ndim < 3:
            img_array = np.repeat(img_array[..., np.newaxis], 3, -1)

        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = transforms(img_tensor)
        return img_tensor.float(), id

    datapipe = datapipe.map(parse_data)
    if not hasattr(datapipe, 'set_length'):
        @functional_datapipe('set_length')
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
