from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openTSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use("Agg")


def main(args):
    output = np.load(f"embeddings/output_{args.name}.npy")
    labels = np.load(f"embeddings/labels_{args.name}.npy")

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

    if args.tsne:
        affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
            output,
            perplexities=[50, 500],
            metric=args.metric,
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
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            cmap="tab10",
            s=0.1,
            alpha=0.8,
        )
        plt.axis("off")
        plt.savefig(f"figures/tsne_{args.name}.png", dpi=600)
        plt.close()

    out_trn = output[:NUM_TRAIN]
    out_tst = output[NUM_TRAIN:]
    lbl_trn = labels[:NUM_TRAIN]
    lbl_tst = labels[NUM_TRAIN:]

    knn = KNeighborsClassifier(
        n_neighbors=args.knn_neighbors,
        n_jobs=8,
        metric=args.metric,
    )
    knn.fit(out_trn, lbl_trn)
    prd = knn.predict(out_tst)

    f1 = f1_score(
        lbl_tst,
        prd,
        labels=list(range(103)),
        average="macro",
    )
    acc = accuracy_score(lbl_tst, prd)

    print(f"{args.name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="resnet18")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--knn-neighbors", type=int, default=5)
    args = parser.parse_args()

    main(args)
