from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use("Agg")

# parse arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", default="resnet18")
parser.add_argument("--visualize", action="store_true", help="tSNE")
parser.add_argument("--metric", default="cosine")
parser.add_argument("--knn-neighbors", type=int, default=5)
args = vars(parser.parse_args())

name = args["name"]
visualize = args["visualize"]
metric = args["metric"]
neighbors = args["knn_neighbors"]
output = np.load(f"embeddings/output_{name}.npy")
labels = np.load(f"embeddings/labels_{name}.npy")

"""
Dataset sizes:
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676

if visualize:
    affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
        output,
        perplexities=[50, 500],
        metric=metric,
        n_jobs=8,
        random_state=3,
    )
    init = openTSNE.initialization.pca(output, random_state=42)
    embedding = openTSNE.TSNE(n_jobs=8).fit(
        affinities=affinities_multiscale_mixture,
        initialization=init,
    )

    # plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=0.1, alpha=0.8
    )
    plt.axis("off")
    plt.savefig(f"figures/tsne_{name}.png", dpi=600)
    plt.close()

    np.save(f"embeddings/embeds_{name}.npy", embedding)

output_train = output[:NUM_TRAIN]
output_test = output[NUM_TRAIN:]
labels_train = labels[:NUM_TRAIN]
labels_test = labels[NUM_TRAIN:]

# knn fit on training data
knn = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=8, metric=metric)
knn.fit(output_train, labels_train)
preds = knn.predict(output_test)

# metrics
f1 = f1_score(labels_test, preds, labels=list(range(103)), average="macro")
acc = accuracy_score(labels_test, preds)

print(f"{name}")
print(f"Accuracy: {acc}")
print(f"Macro F1: {f1}")
print("\n")

# class wise macro f1
f1 = f1_score(labels_test, preds, labels=list(range(103)), average=None)
for i in range(103):
    print(f"class {i}: {f1[i]}")
