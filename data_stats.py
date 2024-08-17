import numpy
from datasheet import *
import matplotlib.pyplot as plt

datasets = ["CUB", "SCARS", "AIRCRAFT", "HERB19", "PLANKTON"]

for dataset in datasets:
    # record the following statistics
    train_old_samples = 0
    train_old_classes = set()
    train_new_samples = 0
    train_new_classes = set()
    test_old_samples = 0
    test_old_classes = set()
    test_new_samples = 0
    test_new_classes = set()

    # data of samples as dicts in this list
    if dataset == "CUB":
        info = CUB_INFO
    elif dataset == "SCARS":
        info = SCARS_INFO
        # shift labels to start from 0
        for sample in info:
            sample["label"] -= 1
    elif dataset == "AIRCRAFT":
        info = AIRCRAFT_INFO
    elif dataset == "HERB19":
        info = HERB19_INFO
    elif dataset == "PLANKTON":
        info = PLANKTON_INFO

    min_label = min([sample["label"] for sample in info])
    max_label = max([sample["label"] for sample in info])
    print(f"min label: {min_label}, max label: {max_label}")

    class_samples = {i: 0 for i in range(NUM_CLASSES[dataset])}

    # calculate the statistics
    for sample in info:
        # class in "label"
        # train/test in "train"
        # old/new in "old"
        if sample["train"] == 1:
            if sample["old"] == 1:
                train_old_samples += 1
                train_old_classes.add(sample["label"])
            else:
                train_new_samples += 1
                train_new_classes.add(sample["label"])
        else:
            if sample["old"] == 1:
                test_old_samples += 1
                test_old_classes.add(sample["label"])
            else:
                test_new_samples += 1
                test_new_classes.add(sample["label"])
        class_samples[sample["label"]] += 1

    # print the statistics
    print(f"Dataset: {dataset}")
    print(f"Labelled samples: {train_old_samples}")
    print(f"Labelled classes: {len(train_old_classes)}")
    print(
        f"Unlabelled samples: {train_new_samples + test_old_samples + test_new_samples}"
    )
    print(
        f"Unlabelled classes: {len(train_new_classes.union(test_old_classes).union(test_new_classes))}"
    )
    print(f"Unlabelled old samples: {test_old_samples}")
    print(f"Unlabelled new samples: {train_new_samples + test_new_samples}")
    print(f"min samples per class: {min(class_samples.values())}")
    print(f"max samples per class: {max(class_samples.values())}")
    total_samples = (
        train_old_samples + train_new_samples + test_old_samples + test_new_samples
    )
    class_freq = {
        i: class_samples[i] / total_samples for i in range(NUM_CLASSES[dataset])
    }
    if dataset == "HERB19":
        herb_class_freq = class_freq
        herb_total_samples = total_samples
    elif dataset == "PLANKTON":
        plankton_class_freq = class_freq
        plankton_total_samples = total_samples

# plot the distribution of sample frequency vs sorted class index (log - log)
plt.figure()
plt.plot(
    sorted(herb_class_freq.values(), reverse=True), label="Herbarium-19", color="C0"
)
plt.plot(
    sorted(plankton_class_freq.values(), reverse=True),
    label="WHOI-Plankton",
    color="C1",
)
plt.yscale("log")
plt.xlabel("Sorted Class index")
plt.ylabel("Sample Frequency")
plt.title(f"Class distribution")
plt.legend()
plt.grid()
plt.savefig(f"class_freq.png", dpi=300)
plt.close()
