import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
from datasheet import *


def eRANK(embeddings):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    corr = np.dot(embeddings.T, embeddings) / embeddings.shape[0]
    eigvals = np.linalg.eigvals(corr)
    eigvals = (eigvals + 1e-6) / np.sum(eigvals)  # add 1e-6 to avoid log(0)
    entropy = -np.sum(eigvals * np.log(eigvals))
    eRANK = np.exp(entropy)
    return eRANK


def main(args):
    data = "CUB"
    trials = 3
    num_classes = NUM_CLASSES[data]

    if data == "PLANKTON":
        n = 110
    elif data == "HERB19":
        n = 700
    elif data == "SCARS":
        n = 230
    elif data == "AIRCRAFT":
        n = 110
    elif data == "CUB":
        n = 230

    old_accs = []
    new_accs = []
    t_accs = []
    for trial in range(trials):
        trn_old_out = np.load(f"outputs/{args.name}_{data}_trn_old_out_{trial}.npy")
        trn_old_lbl = np.load(f"outputs/{args.name}_{data}_trn_old_lbl_{trial}.npy")
        trn_new_out = np.load(f"outputs/{args.name}_{data}_trn_new_out_{trial}.npy")
        trn_new_lbl = np.load(f"outputs/{args.name}_{data}_trn_new_lbl_{trial}.npy")
        tst_old_out = np.load(f"outputs/{args.name}_{data}_tst_old_out_{trial}.npy")
        tst_old_lbl = np.load(f"outputs/{args.name}_{data}_tst_old_lbl_{trial}.npy")
        tst_new_out = np.load(f"outputs/{args.name}_{data}_tst_new_out_{trial}.npy")
        tst_new_lbl = np.load(f"outputs/{args.name}_{data}_tst_new_lbl_{trial}.npy")

        if args.knn:
            trn_out = np.concatenate([trn_old_out, trn_new_out], axis=0)
            trn_lbl = np.concatenate([trn_old_lbl, trn_new_lbl], axis=0)

            out = np.concatenate([trn_out, tst_old_out, tst_new_out], axis=0)
            lbl = np.concatenate([trn_lbl, tst_old_lbl, tst_new_lbl], axis=0)

            knn = KNeighborsClassifier(n_neighbors=args.k, n_jobs=8, metric=args.metric)

            # here we assume trn is fully labeled, i.e., both old and new
            knn.fit(trn_out, trn_lbl)

            tst_old_prd = knn.predict(tst_old_out)
            tst_new_prd = knn.predict(tst_new_out)
            t_prd = knn.predict(trn_old_out)

            tst_old_acc = accuracy_score(tst_old_lbl, tst_old_prd)
            tst_new_acc = accuracy_score(tst_new_lbl, tst_new_prd)
            t_acc = accuracy_score(trn_old_lbl, t_prd)

            old_accs.append(tst_old_acc)
            new_accs.append(tst_new_acc)
            t_accs.append(t_acc)

        elif args.kmeans:
            if args.metric == "cosine":
                trn_old_out = trn_old_out / np.linalg.norm(trn_old_out, axis=1)[:, None]
                trn_new_out = trn_new_out / np.linalg.norm(trn_new_out, axis=1)[:, None]
                tst_old_out = tst_old_out / np.linalg.norm(tst_old_out, axis=1)[:, None]
                tst_new_out = tst_new_out / np.linalg.norm(tst_new_out, axis=1)[:, None]

            trn_out = np.concatenate([trn_old_out, trn_new_out], axis=0)
            trn_lbl = np.concatenate([trn_old_lbl, trn_new_lbl], axis=0)

            out = np.concatenate([trn_out, tst_old_out, tst_new_out], axis=0)
            lbl = np.concatenate([trn_lbl, tst_old_lbl, tst_new_lbl], axis=0)

            if args.init_lbl:
                # initialize some centers with the true labels of trn_old
                # and the rest with kmeans++
                centers, _ = kmeans_plusplus(out, n)
                for label in np.unique(trn_old_lbl):
                    centers[label] = np.mean(trn_old_out[trn_old_lbl == label], axis=0)
            else:
                centers, _ = kmeans_plusplus(out, n)

            kmeans = KMeans(n_clusters=n, init=centers, n_init=1)
            kmeans.fit(out)

            prd = kmeans.predict(out)

            # optimal assignment
            D = max(num_classes, n)
            cost = np.zeros((D, D))
            for i in range(prd.shape[0]):
                cost[int(prd[i]), int(lbl[i])] += 1
            row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

            opt_prd = np.zeros_like(prd)
            for i in range(prd.shape[0]):
                opt_prd[i] = col_ind[int(prd[i])]

            trn_old_prd = opt_prd[: trn_old_out.shape[0]]
            trn_new_prd = opt_prd[
                trn_old_out.shape[0] : trn_old_out.shape[0] + trn_new_out.shape[0]
            ]
            tst_old_prd = opt_prd[
                -tst_old_out.shape[0] - tst_new_out.shape[0] : -tst_new_out.shape[0]
            ]
            tst_new_prd = opt_prd[-tst_new_out.shape[0] :]

            t_prd = trn_old_prd
            old_prd = tst_old_prd
            new_prd = np.concatenate([trn_new_prd, tst_new_prd], axis=0)
            t_lbl = trn_old_lbl
            old_lbl = tst_old_lbl
            new_lbl = np.concatenate([trn_new_lbl, tst_new_lbl], axis=0)

            old_acc = accuracy_score(old_lbl, old_prd)
            new_acc = accuracy_score(new_lbl, new_prd)
            t_acc = accuracy_score(t_lbl, t_prd)

            old_accs.append(old_acc)
            new_accs.append(new_acc)
            t_accs.append(t_acc)

        elif args.logits:
            trn_out = np.concatenate([trn_old_out, trn_new_out], axis=0)
            trn_lbl = np.concatenate([trn_old_lbl, trn_new_lbl], axis=0)

            out = np.concatenate([trn_out, tst_old_out, tst_new_out], axis=0)
            lbl = np.concatenate([trn_lbl, tst_old_lbl, tst_new_lbl], axis=0)

            # out contains the logits
            prd = np.argmax(out, axis=1)

            # optimal assignment
            D = max(num_classes, n)
            cost = np.zeros((D, D))
            for i in range(prd.shape[0]):
                cost[int(prd[i]), int(lbl[i])] += 1
            row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

            opt_prd = np.zeros_like(prd)
            for i in range(prd.shape[0]):
                opt_prd[i] = col_ind[int(prd[i])]

            trn_old_prd = opt_prd[: trn_old_out.shape[0]]
            trn_new_prd = opt_prd[
                trn_old_out.shape[0] : trn_old_out.shape[0] + trn_new_out.shape[0]
            ]
            tst_old_prd = opt_prd[
                -tst_old_out.shape[0] - tst_new_out.shape[0] : -tst_new_out.shape[0]
            ]
            tst_new_prd = opt_prd[-tst_new_out.shape[0] :]

            t_prd = trn_old_prd
            old_prd = tst_old_prd
            new_prd = np.concatenate([trn_new_prd, tst_new_prd], axis=0)
            t_lbl = trn_old_lbl
            old_lbl = tst_old_lbl
            new_lbl = np.concatenate([trn_new_lbl, tst_new_lbl], axis=0)

            old_acc = accuracy_score(old_lbl, old_prd)
            new_acc = accuracy_score(new_lbl, new_prd)
            t_acc = accuracy_score(t_lbl, t_prd)

            old_accs.append(old_acc)
            new_accs.append(new_acc)
            t_accs.append(t_acc)

    old_acc = np.mean(old_accs)
    new_acc = np.mean(new_accs)
    t_acc = np.mean(t_accs)

    std_old_acc = np.std(old_accs)
    std_new_acc = np.std(new_accs)
    std_t_acc = np.std(t_accs)

    if args.knn:
        print(f"Dataset: {data}")
        print(f"method: knn, k: {args.k}, metric: {args.metric}")
        print(f"Old Test Accuracy: {old_acc:.4f} ± {std_old_acc:.4f}")
        print(f"New Test Accuracy: {new_acc:.4f} ± {std_new_acc:.4f}")
        print(f"Train Accuracy: {t_acc:.4f} ± {std_t_acc:.4f}")

    elif args.kmeans:
        print(f"Dataset: {data}")
        print(f"method: kmeans, n: {n}, metric: {args.metric}")
        print(f"Old Test Accuracy: {old_acc:.4f} ± {std_old_acc:.4f}")
        print(f"New Test Accuracy: {new_acc:.4f} ± {std_new_acc:.4f}")
        print(f"Train Accuracy: {t_acc:.4f} ± {std_t_acc:.4f}")

    elif args.logits:
        print(f"Dataset: {data}")
        print(f"method: logits")
        print(f"Old Test Accuracy: {old_acc:.4f} ± {std_old_acc:.4f}")
        print(f"New Test Accuracy: {new_acc:.4f} ± {std_new_acc:.4f}")
        print(f"Train Accuracy: {t_acc:.4f} ± {std_t_acc:.4f}")

    # compute effective rank
    eRANK_val = eRANK(out)
    print(f"Effective Rank: {eRANK_val:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ViT_DINO")
    parser.add_argument("--knn", action="store_true")
    parser.add_argument("--kmeans", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--init_lbl", action="store_true")
    parser.add_argument("--logits", action="store_true")
    args = parser.parse_args()

    main(args)
