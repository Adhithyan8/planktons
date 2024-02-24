import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# load the embeddings
model_name = "vitb14-dinov2"
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

# tsne
affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    output,
    perplexities=[50, 500],
    metric="cosine",
    n_jobs=12,
    random_state=3,
)
init = openTSNE.initialization.pca(output, random_state=42)
embedding = openTSNE.TSNE(n_jobs=12).fit(
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

embed_test = embedding[NUM_TRAIN:]
# knn
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=12)
knn.fit(embed_train, labels[:NUM_TRAIN])

# test the classifier (class wise F1 score and overall F1 score)
preds = knn.predict(embed_test)
f1 = f1_score(labels[NUM_TRAIN:], preds, average="macro")  # unweighted average
acc = accuracy_score(labels[NUM_TRAIN:], preds)

print(f"Accuracy: {acc}")
print(f"F1 score: {f1}")

# class wise F1 score
for i in range(int(max(labels)) + 1):
    mask = labels[NUM_TRAIN:] == i
    f1 = f1_score(labels[NUM_TRAIN:][mask], preds[mask], average="macro")
    print(f"F1 score for class {i}: {f1}")
