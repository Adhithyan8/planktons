import numpy as np
import openTSNE
import torch
from sklearn.neighbors import KNeighborsClassifier

model = "dinov2"

# load the vectors
train_vecs = torch.load(f"embeddings/vecs_{model}.pt").numpy()  # (N, 2048)
test_vecs = torch.load(f"embeddings/vecs_test_{model}.pt").numpy()  # (N, 2048)
train_tags = torch.load("embeddings/tags.pt").numpy()  # (N,)
test_tags = torch.load("embeddings/tags_test.pt").numpy()  # (N,)

# join the train and test vectors
vecs = np.concatenate((train_vecs, test_vecs), axis=0)

# perform tsne with multiscale embedding
affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    vecs,
    perplexities=[50, 500],
    metric="cosine",
    n_jobs=8,
    random_state=3,
)
init = openTSNE.initialization.pca(vecs, random_state=42)
embedding = openTSNE.TSNE(n_jobs=8).fit(
    affinities=affinities_multiscale_mixture,
    initialization=init,
)

# save the tsne embedding
np.save(f"embeddings/tsne_joint_{model}.npy", embedding)

# train a knn classifier
train_embedding = embedding[: train_vecs.shape[0]]
test_embedding = embedding[train_vecs.shape[0] :]

knn = KNeighborsClassifier(n_neighbors=15, n_jobs=8)
knn.fit(train_embedding, train_tags)
print(knn.score(test_embedding, test_tags))
