from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")

# parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", default="resnet18", help="Model architecture")
parser.add_argument(
    "-v", "--visualize", action="store_true", help="Visualize 2D embedding with tSNE"
)
args = vars(parser.parse_args())

# load the embeddings
model_name = args["model"]
visualize = args["visualize"]
output = np.load(f"embeddings/output_{model_name}.npy")
labels = np.load(f"embeddings/labels_{model_name}.npy")

"""
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676

# normalize output to unit length
output /= np.linalg.norm(output, axis=1, keepdims=True)

output_train = output[:NUM_TRAIN]
output_test = output[NUM_TRAIN:]

# kmeans
kmeans = KMeans(n_clusters=103, random_state=0).fit(output) # assuming 103 classes, TODO: estimate classes
preds = kmeans.predict(output_test)

# optimal assignment using linear sum assignment (cost is accuracy)
# preds and labels are [0, 102] and [0, 102] respectively
cost_matrix = np.zeros((103, 103))
for i in range(preds.shape[0]):
    cost_matrix[int(preds[i]), int(labels[NUM_TRAIN + i])] += 1

row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
optimal_preds = np.zeros_like(preds)
for i in range(preds.shape[0]):
    optimal_preds[i] = col_ind[int(preds[i])]

# calculate accuracy and f1 score
acc = accuracy_score(labels[NUM_TRAIN:], optimal_preds)
f1 = f1_score(labels[NUM_TRAIN:], optimal_preds, labels=list(range(103)), average="macro")

print(f"{model_name}")
print(f"Accuracy: {acc}")
print(f"F1 score: {f1}")

# classwise macro f1
f1 = f1_score(labels[NUM_TRAIN:], optimal_preds, labels=list(range(103)), average=None)
for i in range(103):
    print(f"Class {i}: {f1[i]}")

if visualize:
    # plot the centroids
    centroids = kmeans.cluster_centers_
    # tsne
    affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
        np.concatenate((output, centroids), axis=0),
        perplexities=[50, 500],
        metric="cosine",
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

    # plot the embedding
    plt.figure(figsize=(10, 10))
    plt.scatter(
        embedding[:output.shape[0], 0],
        embedding[:output.shape[0], 1],
        c=labels, 
        cmap="Spectral",
        s=0.1,
        alpha=0.8,
    )
    plt.scatter(
        embedding[output.shape[0]:, 0],
        embedding[output.shape[0]:, 1],
        c="black",
        s=10,
        alpha=1,
    )
    plt.axis("off")
    plt.savefig(f"figures/tsne_{model_name}.png", dpi=600)
    plt.close()

    # save embedding
    np.save(f"embeddings/embeds_{model_name}.npy", embedding)
