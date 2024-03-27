import numpy as np

labels = np.load("embeddings/labels.npy")

# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676

labels_train = labels[:NUM_TRAIN]
labels_test = labels[NUM_TRAIN:]

for class_id in range(103):
    print(
        f"Class {class_id}, Train: {np.sum(labels_train == class_id)}, Test: {np.sum(labels_test == class_id)}"
    )
