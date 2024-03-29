import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676

embeddings = np.load("embeddings/embeds_ft500_resnet18.npy")
labels = np.load("embeddings/labels_ft500_resnet18.npy").astype('int')

embeds_train = embeddings[:NUM_TRAIN]
embeds_test = embeddings[NUM_TRAIN:]
labels_train = labels[:NUM_TRAIN]
labels_test = labels[NUM_TRAIN:]

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=8, metric="cosine")
knn.fit(embeds_train, labels_train)

preds = knn.predict(embeds_test)

# error analysis
data = pd.DataFrame(
    columns=["pred", "label", "knn1", "knn2", "knn3", "knn4", "knn5",
             "dist1", "dist2", "dist3", "dist4", "dist5"])
data["pred"] = preds
data["label"] = labels_test
# the 5 nearest neighbors
distances, indices = knn.kneighbors(embeds_test)
data["knn1"] = indices[:, 0]
data["knn2"] = indices[:, 1]
data["knn3"] = indices[:, 2]
data["knn4"] = indices[:, 3]
data["knn5"] = indices[:, 4]
# the 5 distances
data["dist1"] = distances[:, 0]
data["dist2"] = distances[:, 1]
data["dist3"] = distances[:, 2]
data["dist4"] = distances[:, 3]
data["dist5"] = distances[:, 4]

# save the data
data.to_csv("error_analysis.csv", index=False)