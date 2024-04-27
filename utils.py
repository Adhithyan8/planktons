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


# @functional_datapipe("set_length")
# class LengthSetterIterDataPipe(IterDataPipe):
#     def __init__(self, source_datapipe: IterDataPipe, length: int) -> None:
#         self.source_datapipe = source_datapipe
#         assert length >= 0
#         self.length = length

#     def __iter__(self) -> IterDataPipe:  # type: ignore
#         yield from self.source_datapipe

#     def __len__(self) -> int:
#         return self.length


def inference_datapipe(path, num_images, transforms, padding, ignore_mix=True):
    fileopener = FileOpener(path, mode="b")
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

    # get label dictionary
    with open("labels.json") as f:
        label2id = json.load(f)
    with open("labelled_classes.json") as f:
        labelled_classes = json.load(f)

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


def std_of_l2_normalized(z):
    """
    From lightly-ai repo
    https://github.com/lightly-ai/lightly/blob/master/lightly/utils/debug.py
    """
    z_norm = torch.nn.functional.normalize(z, dim=1)
    return torch.std(z_norm, dim=0).mean()


class SupervisedContrastive(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastive, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # find and drop unlabelled samples
        mask = labels != -1
        features = features[mask]
        labels = labels[mask]

        # square mask that matches labels
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.to(features.device)
        # ignore self similarity
        diag = torch.eye(features.shape[0], device=features.device).bool()
        mask = mask.masked_fill_(diag, bool(0))

        # get the cosine similarity matrix
        features = torch.nn.functional.normalize(features, dim=1)
        cos = features @ features.T / self.temperature

        # where the mask is true
        pos_sum = ((mask.float() * cos).sum(dim=1)) / (mask.sum(dim=1).float())
        pos = pos_sum.mean()

        # logsumexp over all but the diagonal
        logsumexp = cos.masked_fill_(diag, float("-inf")).logsumexp(dim=1)
        neg = logsumexp.mean()

        loss = -(pos - neg)
        return loss


class SemiSupervisedContrastive(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SemiSupervisedContrastive, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # find the unlabelled samples
        mask = labels == -1
        features_u = features[mask]
        labels_u = labels[mask]  # not used
        size_u = features_u.shape[0] // 2

        # find and drop unlabelled samples
        mask = labels != -1
        features_l = features[mask]
        labels_l = labels[mask]

        # mask matching positive pairs
        m11 = (
            torch.cat(
                (
                    torch.cat((torch.zeros(size_u, size_u), torch.eye(size_u)), dim=1),
                    torch.cat((torch.eye(size_u), torch.zeros(size_u, size_u)), dim=1),
                ),
                dim=0,
            )
            .bool()
            .to(features.device)
        )
        m22 = labels_l.unsqueeze(0) == labels_l.unsqueeze(1)
        # set diagonal to false
        m22 = m22.masked_fill_(
            torch.eye(m22.shape[0], device=features.device).bool(), bool(0)
        )
        m22.to(features.device)
        m12 = torch.zeros(
            features_u.shape[0], features_l.shape[0], device=features.device
        ).bool()
        m21 = torch.zeros(
            features_l.shape[0], features_u.shape[0], device=features.device
        ).bool()
        mask = torch.cat(
            (torch.cat((m11, m12), dim=1), torch.cat((m21, m22), dim=1)), dim=0
        )

        features = torch.cat((features_u, features_l), dim=0)
        # get the cosine similarity matrix
        features = torch.nn.functional.normalize(features, dim=1)
        cos = features @ features.T / self.temperature

        # where the mask is true
        pos_sum = ((mask.float() * cos).sum(dim=1)) / (mask.sum(dim=1).float())
        pos = pos_sum.mean()

        # logsumexp over all but the diagonal
        logsumexp = cos.masked_fill_(
            torch.eye(cos.shape[0], device=features.device).bool(), float("-inf")
        ).logsumexp(dim=1)
        neg = logsumexp.mean()

        loss = -(pos - neg)
        return loss


class InfoCNECauchy(torch.nn.Module):
    def __init__(self, temperature=0.07, mode="selfsupervised"):
        super(InfoCNECauchy, self).__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(self, features, labels):
        if self.mode == "selfsupervised":
            size = features.shape[0] // 2
            mask = torch.cat(
                    (
                        torch.cat((torch.zeros(size, size), torch.eye(size)), dim=1),
                        torch.cat((torch.eye(size), torch.zeros(size, size)), dim=1),
                    ),
                    dim=0,
                ).bool().to(features.device)

        elif self.mode == "supervised":
            # find and drop unlabelled samples
            mask = labels != -1
            features = features[mask]
            labels = labels[mask]

            # square mask that matches labels
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask.to(features.device)
            # ignore self similarity
            mask = mask.masked_fill_(
                torch.eye(features.shape[0], device=features.device).bool(), bool(0)
            )

        elif self.mode == "semisupervised":
            # find the unlabelled samples
            mask = labels == -1
            features_u = features[mask]
            labels_u = labels[mask]
            size_u = features_u.shape[0] // 2

            # find and drop unlabelled samples
            mask = labels != -1
            features_l = features[mask]
            labels_l = labels[mask]

            # mask matching positive pairs
            m11 = (
                torch.cat(
                    (
                        torch.cat(
                            (torch.zeros(size_u, size_u), torch.eye(size_u)), dim=1
                        ),
                        torch.cat(
                            (torch.eye(size_u), torch.zeros(size_u, size_u)), dim=1
                        ),
                    ),
                    dim=0,
                )
                .bool()
                .to(features.device)
            )
            m22 = labels_l.unsqueeze(0) == labels_l.unsqueeze(1)
            m22.to(features.device)
            # set diagonal to false
            m22 = m22.masked_fill_(
                torch.eye(m22.shape[0], device=features.device).bool(), bool(0)
            )
            m12 = torch.zeros(
                features_u.shape[0], features_l.shape[0], device=features.device
            ).bool()
            m21 = torch.zeros(
                features_l.shape[0], features_u.shape[0], device=features.device
            ).bool()
            mask = torch.cat(
                (torch.cat((m11, m12), dim=1), torch.cat((m21, m22), dim=1)), dim=0
            )

            features = torch.cat((features_u, features_l), dim=0)

        # cauchy similarity
        sim = 1 / (
            torch.cdist(features, features, p=2) / self.temperature
        ).square().add(1)

        # where the mask is true - log(sim[mask])
        pos = (mask.float() * torch.log(sim)).sum(dim=1) / (mask.sum(dim=1).float())
        pos = pos.mean()

        # log sum over all but the diagonal
        neg = (sim - torch.eye(sim.shape[0], device=features.device)).sum(dim=1).log_().mean()

        loss = -(pos - neg)
        return loss
