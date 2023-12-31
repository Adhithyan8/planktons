import torch
import numpy as np
import matplotlib.pyplot as plt
import openTSNE

# load the vectors
vecs = torch.load("tensors/resnet_vecs.pt").numpy()  # (N, 2048)
tags = torch.load("tensors/tags.pt").numpy()  # (N,)

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

# save the embedding
np.save("embedding.npy", embedding)

# # load the embedding
# embedding = np.load("embedding.npy")

# plot the embedding
plt.figure(figsize=(10, 10))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=tags,
    cmap="Spectral",
    s=0.1,
)
plt.axis("off")
plt.savefig("embedding.png", dpi=300)
plt.close()
