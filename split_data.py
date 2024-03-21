import json
import random
from fnmatch import fnmatch

from torchdata.datapipes.iter import FileOpener

# 2013 is considered as the training set
paths = ["/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip"]
datapipe = FileOpener(paths, mode="b").load_from_zip()


# ignore mix folder for now
def image_filter(data):
    file_name, _ = data
    return fnmatch(file_name, "*.png") and file_name.split("/")[-2] != "mix"


# filter images
datapipe = datapipe.filter(image_filter)

# load labels from json
with open("labels.json") as f:
    label2id = json.load(f)
num_labels = len(label2id)

# get count of each class
class_count = {i: 0 for i in range(num_labels)}
for file_name, _ in datapipe:
    class_count[label2id[file_name.split("/")[-2]]] += 1

# sort by count (descending)
sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)

# dropping random 50 classes to be the labelled data
SEED = 42
random.seed(SEED)
random.shuffle(sorted_class_count)
# drop 50 classes
drop_classes = [label for label, _ in sorted_class_count[:50]]
print(f"Dropping classes: {drop_classes}")

# save dropped classes
with open("labelled_classes.json", "w") as f:
    json.dump(drop_classes, f)

import matplotlib.pyplot as plt

# highlight dropped classes in red color
plt.figure(figsize=(10, 5))

# sort by class count (descending)
sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
# map the class label and position
class_label_map = {label: i for i, (label, _) in enumerate(sorted_class_count)}
# scatter plot
plt.scatter(range(num_labels), [count for _, count in sorted_class_count])
for label in drop_classes:
    plt.scatter(class_label_map[label], class_count[label], color="red")
plt.xlabel("Class")
plt.ylabel("Count")
plt.yscale("log")
plt.legend(["Unlabelled", "Labelled"])
plt.title("Class distribution (2013)")
plt.savefig("class_distribution_dropped.png")
plt.close()

# how many images to drop
num_images = sum(class_count.values())
drop_images = sum(class_count[label] for label in drop_classes)

print(f"Total images: {num_images}")
print(f"Dropping images: {drop_images}")
print(f"Remaining images: {num_images - drop_images}")
