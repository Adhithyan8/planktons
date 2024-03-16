from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

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
labels = np.load("embeddings/labels.npy")

"""
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676

if visualize:
    # tsne
    affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
        output,
        perplexities=[50, 500],
        metric="cosine",
        n_jobs=8,
        random_state=3,
    )
    init = openTSNE.initialization.pca(output, random_state=42)
    embedding = openTSNE.TSNE(n_jobs=8).fit(
        affinities=affinities_multiscale_mixture,
        initialization=init,
    )

    # plot the embedding (only train set)
    embed_train = embedding[:NUM_TRAIN]
    plt.figure(figsize=(10, 10))
    plt.scatter(
        embed_train[:, 0],
        embed_train[:, 1],
        c=labels[:NUM_TRAIN],
        cmap="Spectral",
        s=0.1,
    )
    plt.axis("off")
    plt.savefig(f"figures/tsne_train_{model_name}.png", dpi=300)
    plt.close()

output_train = output[:NUM_TRAIN]
output_test = output[NUM_TRAIN:]

# knn
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=8, metric="cosine")
knn.fit(output_train, labels[:NUM_TRAIN])

# test
preds = knn.predict(output_test)
f1 = f1_score(labels[NUM_TRAIN:], preds, average="macro")
acc = accuracy_score(labels[NUM_TRAIN:], preds)

print(f"{model_name}")
print(f"Accuracy: {acc}")
print(f"F1 score: {f1}")
