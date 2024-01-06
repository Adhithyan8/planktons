import torch
import numpy as np
import matplotlib.pyplot as plt
import openTSNE

# load the vectors
# vecs = torch.load("embeddings/vecs_resnet.pt").numpy()  # (N, 2048)
# vecs = torch.load("embeddings/vecs_dinov2.pt").numpy()  # (N, 768)
vecs = torch.load("embeddings/vecs_clip.pt").numpy()  # (N, 512)
tags = torch.load("embeddings/tags.pt").numpy()  # (N,)

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
# np.save("embeddings/tsne_resnet.npy", embedding)
# np.save("embeddings/tsne_dinov2.npy", embedding)
np.save("embeddings/tsne_clip.npy", embedding)

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
plt.savefig("figures/tsne_clip.png", dpi=300)
plt.close()
