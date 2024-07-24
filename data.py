from PIL import Image
from torch.utils.data import Dataset


class make_dataset(Dataset):
    def __init__(self, info, split_fit="train", split_cat="old", transform=None):
        self.info = info
        self.split_fit = split_fit
        self.split_cat = split_cat
        self.data = []
        for i in range(len(info)):
            if info[i]["train"] == 1 and split_fit == "train":
                if info[i]["old"] == 1 and split_cat == "old":
                    self.data.append(info[i])
                elif info[i]["old"] == 0 and split_cat == "new":
                    self.data.append(info[i])
            elif info[i]["train"] == 0 and split_fit == "test":
                if info[i]["old"] == 1 and split_cat == "old":
                    self.data.append(info[i])
                elif info[i]["old"] == 0 and split_cat == "new":
                    self.data.append(info[i])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["path"]
        label = self.data[idx]["label"]
        if self.transform:
            img, label = self.transform(img, label)
        return img, label
