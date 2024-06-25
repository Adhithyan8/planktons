import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score


def main(args):
    output = np.load(f"embeddings/output_{args.name}.npy")
    labels = np.load(f"embeddings/labels_{args.name}.npy")

    # magic numbers
    if args.data == "cub":
        NUM_TRAIN = 5994
        NUM_TEST = 5794
        num_classes = 200
    elif args.data == "whoi_plankton":
        NUM_TRAIN = 115951
        NUM_TEST = 63676
        num_classes = 103

    preds = np.argmax(output, axis=1)
    prd_trn = preds[:NUM_TRAIN]
    prd_tst = preds[NUM_TRAIN:]
    lbl_trn = labels[:NUM_TRAIN]
    lbl_tst = labels[NUM_TRAIN:]

    # optimal assignment to maximize accuracy
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

    if args.verbose:
        index_to_id_map = {}
        for i in range(output.shape[1]):
            index_to_id_map[i] = c_ind[i]
        id_to_index_map = {v: k for k, v in index_to_id_map.items()}

        debug_info = {}
        for i in range(prd_tst.shape[0]):
            if lbl_tst[i] not in debug_info:
                debug_info[lbl_tst[i]] = {
                    "num_samples": 0,
                    "num_preds": 0,
                    "num_correct": 0,
                    "logit_at_index": [],
                    "logit_at_id": [],
                }
            if opt_prd[i] not in debug_info:
                debug_info[opt_prd[i]] = {
                    "num_samples": 0,
                    "num_preds": 0,
                    "num_correct": 0,
                    "logit_at_index": [],
                    "logit_at_id": [],
                }
            debug_info[lbl_tst[i]]["num_samples"] += 1
            debug_info[opt_prd[i]]["num_preds"] += 1
            if lbl_tst[i] == opt_prd[i]:
                debug_info[lbl_tst[i]]["num_correct"] += 1
            debug_info[lbl_tst[i]]["logit_at_index"].append(
                output[NUM_TRAIN + i][np.argmax(output[NUM_TRAIN + i])]
            )
            debug_info[lbl_tst[i]]["logit_at_id"].append(
                output[NUM_TRAIN + i][id_to_index_map[lbl_tst[i]]]
            )

        for i in num_classes:
            if i not in debug_info:
                debug_info[i] = {
                    "num_samples": 0,
                    "num_preds": 0,
                    "num_correct": 0,
                    "logit_at_index": [],
                    "logit_at_id": [],
                }
            print(
                f"{i}, {debug_info[i]['num_samples']}, {debug_info[i]['num_preds']}, {debug_info[i]['num_correct']/debug_info[i]['num_samples']:.4f}, {np.mean(debug_info[i]['logit_at_id']):.4f}, {np.var(debug_info[i]['logit_at_id']):.4f}, {np.mean(debug_info[i]['logit_at_index']):.4f}, {np.var(debug_info[i]['logit_at_index']):.4f}"
            )

        for i in range(num_classes):
            if i not in debug_info:
                continue
            plt.hist(debug_info[i]["logit_at_id"], range=[-1, 1], bins=100)
            plt.savefig(f"hist1_{i}.png")
            plt.close()
            plt.hist(debug_info[i]["logit_at_index"], range=[-1, 1], bins=100)
            plt.savefig(f"hist2_{i}.png")
            plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", default="resnet18")
    parser.add_argument("--data", default="cub")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
