from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN
from sklearn.metrics import accuracy_score, f1_score

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

    out_trn = output[:NUM_TRAIN]
    out_tst = output[NUM_TRAIN:]
    lbl_trn = labels[:NUM_TRAIN]
    lbl_tst = labels[NUM_TRAIN:]

    # clustering
    model = HDBSCAN(min_cluster_size=50, min_samples=10)
    prd = model.fit_predict(out_tst)

    # shift labels by 1 to avoid -1
    prd += 1
    print(f"Number of clusters: {len(np.unique(prd))}")

    if args.viz:
        # plotting
        plt.figure(figsize=(10, 10))
        for i in range(len(np.unique(prd))):
            if i == 0:
                plt.scatter(
                    out_tst[prd == i, 0],
                    out_tst[prd == i, 1],
                    s=1.0,
                    alpha=0.5,
                    label=i,
                    marker="x",
                )
            else:
                plt.scatter(
                    out_tst[prd == i, 0],
                    out_tst[prd == i, 1],
                    s=1.0,
                    alpha=0.5,
                    label=i,
                    marker="o",
                )
        plt.axis("off")
        plt.savefig(f"figures/hdbscan_{args.name}.png", dpi=600)
        plt.close()

    if args.one2one:
        # optimal assignment to maximize accuracy
        cst = np.zeros((len(np.unique(prd)), 103))
        for i in range(prd.shape[0]):
            cst[int(prd[i]), int(lbl_tst[i])] += 1
        r_ind, c_ind = linear_sum_assignment(cst, maximize=True)

        opt_prd = np.zeros_like(prd)
        for i in range(prd.shape[0]):
            opt_prd[i] = c_ind[int(prd[i])]
    else:
        # assign cluster to the most frequent label
        cst = np.zeros((len(np.unique(prd)), 103))
        for i in range(prd.shape[0]):
            cst[int(prd[i]), int(lbl_tst[i])] += 1
        opt_prd = np.argmax(cst, axis=1)[prd]

    f1 = f1_score(
        lbl_tst,
        opt_prd,
        labels=list(range(103)),
        average="macro",
    )
    acc = accuracy_score(lbl_tst, opt_prd)

    print(f"{args.name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="resnet18")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--one2one", action="store_true")
    args = parser.parse_args()

    main(args)