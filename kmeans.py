from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
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

    if args.viz_large:
        large_class_labels = [88, 49, 95, 8, 90, 19, 65, 5, 66, 38]
        large_class_names = [
            "detritus",
            "Leptocylindrus",
            "mix_elongated",
            "Chaetoceros",
            "dino30",
            "Cylindrotheca",
            "Rhizosolenia",
            "Cerataulina",
            "Skeletonema",
            "Guinardia_delicatula",
        ]

        # plotting
        plt.figure(figsize=(10, 10))
        for i, label in enumerate(large_class_labels):
            plt.scatter(
                output[labels == label, 0],
                output[labels == label, 1],
                c=f"C{i}",
                s=1.0,
                alpha=0.5,
                label=large_class_names[i],
            )
        plt.axis("off")
        plt.legend()
        plt.savefig(f"figures/tsim_{args.name}.png", dpi=600)
        plt.close()

    if args.normalize:
        output /= np.linalg.norm(output, axis=1, keepdims=True)

    out_trn = output[:NUM_TRAIN]
    out_tst = output[NUM_TRAIN:]
    lbl_trn = labels[:NUM_TRAIN]
    lbl_tst = labels[NUM_TRAIN:]

    kmeans = KMeans(
        n_clusters=103,
        random_state=0,
    ).fit(output)
    prd = kmeans.predict(out_tst)

    # optimal assignment to maximize accuracy
    cst = np.zeros((103, 103))
    for i in range(prd.shape[0]):
        cst[int(prd[i]), int(lbl_tst[i])] += 1
    r_ind, c_ind = linear_sum_assignment(cst, maximize=True)

    opt_prd = np.zeros_like(prd)
    for i in range(prd.shape[0]):
        opt_prd[i] = c_ind[int(prd[i])]

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
    parser.add_argument("--viz-large", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    main(args)
