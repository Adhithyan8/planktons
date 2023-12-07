import os
import datasets
import random


# dataset info
_DESCRIPTION = """\
WHOI Planktons dataset consisting of annotated plankton images for classification. 
There are 103 classes ('mix' contains the unlabeled images).
"""

# get label names from folder names
_NAMES = os.listdir("planktons_dataset/data/2014")

"""
TRAIN: data/2013
TEST: data/2014 excluding "mix", as they are unlabeled

Consider three training configurations
1. exclude "mix"; 
    50% classes unlabeled; 
    remaining classes: 50% images labeled, 50% images unlabeled
2. exclude "mix"; 
    40% classes unlabeled;
    40% classes: 50% images labeled, 50% images unlabeled;
    remaining 20% classes: unused
3. include "mix";
    50% classes unlabeled;
    remaining classes: 50% images labeled, 50% images unlabeled
"""


# get the data files for the given years and classes to exclude
def get_files(years, exclude):
    """Get the data files for the given years and classes to exclude."""
    files = []
    for year in years:
        for folder in os.listdir("planktons_dataset/data/" + year):
            if folder not in exclude:
                files += [
                    "planktons_dataset/data/" + year + "/" + folder + "/" + file
                    for file in os.listdir(
                        "planktons_dataset/data/" + year + "/" + folder
                    )
                ]
    return files


class PlanktonsConfig(datasets.BuilderConfig):
    """BuilderConfig for Planktons."""

    def __init__(self, files_config, **kwargs):
        """BuilderConfig for Planktons.
        Args:
          files_config: `dict`
          **kwargs: keyword arguments forwarded to super
        """
        super(PlanktonsConfig, self).__init__(
            version=datasets.Version("1.0.1"), **kwargs
        )
        self.files_config = files_config


class Planktons(datasets.GeneratorBasedBuilder):
    """Planktons dataset."""

    BUILDER_CONFIGS = [
        PlanktonsConfig(
            name="2013-14",
            files_config={
                "train": ["2013"],
                "test": ["2014"],
            },
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # train, validataion, test splits
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files_config": self.config.files_config,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "files_config": self.config.files_config,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, files_config, split):
        """Yields examples."""
        files = get_files(
            files_config[split], exclude=["mix"] if split == "test" else ["mix"]
        )

        for id_, file in enumerate(files):
            # get the label from the file path
            label_name = file.split("/")[-2]
            # get the label id
            label = _NAMES.index(label_name)
            # read images that are in .png format
            yield id_, {
                "image": {"path": file, "bytes": open(file, "rb").read()},
                "label": label,
            }
