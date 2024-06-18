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

    if args.data == "whoi_plankton":
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
    elif args.data == "cub":
        NUM_TRAIN = 5994
        NUM_TEST = 5794

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
        n_clusters=args.k,
        random_state=0,
    ).fit(output)
    prd = kmeans.predict(out_tst)

    if args.data == "whoi_plankton":
        num_classes = 103
    elif args.data == "cub":
        num_classes = 200

    if args.one2one:
        # optimal assignment to maximize accuracy
        D = max(num_classes, args.k)
        cst = np.zeros((D, D))
        for i in range(prd.shape[0]):
            cst[int(prd[i]), int(lbl_tst[i])] += 1
        r_ind, c_ind = linear_sum_assignment(cst, maximize=True)

        opt_prd = np.zeros_like(prd)
        for i in range(prd.shape[0]):
            opt_prd[i] = c_ind[int(prd[i])]
    else:
        # assign cluster to the most frequent label
        cst = np.zeros((args.k, num_classes))
        for i in range(prd.shape[0]):
            cst[int(prd[i]), int(lbl_tst[i])] += 1
        opt_prd = np.argmax(cst, axis=1)[prd]

    f1 = f1_score(
        lbl_tst,
        opt_prd,
        labels=list(range(num_classes)),
        average="macro",
    )
    acc = accuracy_score(lbl_tst, opt_prd)

    print(f"{args.name}, kmeans:{args.k}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="resnet18")
    parser.add_argument("--data", default="whoi_plankton")
    parser.add_argument("--k", type=int, default=103)
    parser.add_argument("--viz-large", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--one2one", action="store_true")
    args = parser.parse_args()

    main(args)
