from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")

# parse arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", default="resnet18")
parser.add_argument("--visualize", action="store_true", help="tSNE")
parser.add_argument("--normalize", action="store_true", help="normalize embeddings")
parser.add_argument("--metric", default="cosine")
args = vars(parser.parse_args())

name = args["name"]
visualize = args["visualize"]
normalize = args["normalize"]
metric = args["metric"]
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

# normalize outputs
if normalize:
    output /= np.linalg.norm(output, axis=1, keepdims=True)

output_train = output[:NUM_TRAIN]
output_test = output[NUM_TRAIN:]
labels_train = labels[:NUM_TRAIN]
labels_test = labels[NUM_TRAIN:]

# kmeans
kmeans = KMeans(n_clusters=103, random_state=0).fit(output)  # assuming 103 classes
preds = kmeans.predict(output_test)

# optimal assignment to maximize accuracy
cost = np.zeros((103, 103))
for i in range(preds.shape[0]):
    cost[int(preds[i]), int(labels_test[i])] += 1
row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

optimal_preds = np.zeros_like(preds)
for i in range(preds.shape[0]):
    optimal_preds[i] = col_ind[int(preds[i])]

# metrics
acc = accuracy_score(labels_test, optimal_preds)
f1 = f1_score(labels_test, optimal_preds, labels=list(range(103)), average="macro")

print(f"{name}")
print(f"Accuracy: {acc}")
print(f"Macro F1: {f1}")
print("\n")

# classwise macro f1
f1 = f1_score(labels_test, optimal_preds, labels=list(range(103)), average=None)
for i in range(103):
    print(f"class {i}: {f1[i]}")

if visualize:
    centroids = kmeans.cluster_centers_
    if output.shape[1] > 2:
        affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
            np.concatenate((output, centroids), axis=0),
            perplexities=[50, 500],
            metric=metric,
            n_jobs=8,
            random_state=3,
        )
        init = openTSNE.initialization.pca(
            np.concatenate((output, centroids), axis=0), random_state=42
        )
        embedding = openTSNE.TSNE(n_jobs=8).fit(
            affinities=affinities_multiscale_mixture,
            initialization=init,
        )
    else:
        embedding = np.concatenate((output, centroids), axis=0)

    # plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(
        embedding[: output.shape[0], 0],
        embedding[: output.shape[0], 1],
        c=labels,
        cmap="Spectral",
        s=0.1,
        alpha=0.8,
    )
    plt.scatter(
        embedding[output.shape[0] :, 0],
        embedding[output.shape[0] :, 1],
        c="black",
        s=10,
        alpha=1,
    )
    plt.axis("off")
    plt.savefig(f"figures/tsne_{name}.png", dpi=600)
    plt.close()

    np.save(f"embeddings/embeds_{name}.npy", embedding)
