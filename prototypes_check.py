import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    p0 = np.load(f"embeddings/prototypes_{args.name}_t0.npy")
    p1 = np.load(f"embeddings/prototypes_{args.name}_t1.npy")

    p1 = p1 / np.linalg.norm(p1, axis=1)[:, None]
    if args.data == "cub":
        with open(f"labeled_classes_cub.json", "r") as f:
            labeled = json.load(f)

        labeled = np.array(sorted(labeled)) - 1
        unlabeled = np.array([i for i in range(p1.shape[0]) if i not in labeled])
    else:
        with open(f"labeled_classes_whoi.json", "r") as f:
            labeled = json.load(f)

        labeled = np.array(sorted(labeled))
        unlabeled = np.array([i for i in range(p1.shape[0]) if i not in labeled])

    p1_sorted = np.concatenate([p1[labeled], p1[unlabeled]])
    sim = np.dot(p1_sorted, p1_sorted.T)

    # plot the similarity matrix with diverging colormap
    plt.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig(f"figures/prototypes_{args.name}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    main(args)
