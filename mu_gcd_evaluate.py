from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score


def main(args):
    output = np.load(f"embeddings/output_{args.name}.npy")
    labels = np.load(f"embeddings/labels_{args.name}.npy")

    # magic numbers
    NUM_TRAIN = 5994
    NUM_TEST = 5794

    preds = np.argmax(output, axis=1)
    prd_trn = preds[:NUM_TRAIN]
    prd_tst = preds[NUM_TRAIN:]
    lbl_trn = labels[:NUM_TRAIN]
    lbl_tst = labels[NUM_TRAIN:]

    # optimal assignment to maximize accuracy
    num_classes = 200
    D = max(num_classes, output.shape[1])
    cst = np.zeros((D, D))
    for i in range(prd_tst.shape[0]):
        cst[int(prd_tst[i]), int(lbl_tst[i])] += 1
    r_ind, c_ind = linear_sum_assignment(cst, maximize=True)

    opt_prd = np.zeros_like(prd_tst)
    for i in range(prd_tst.shape[0]):
        opt_prd[i] = c_ind[int(prd_tst[i])]

    f1 = f1_score(
        lbl_tst,
        opt_prd,
        labels=list(range(num_classes)),
        average="macro",
    )
    acc = accuracy_score(lbl_tst, opt_prd)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="resnet18")
    args = parser.parse_args()

    main(args)
