import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus


def main(args):
    output = np.load(f"embeddings/output_{args.name}.npy")

    if args.mode == "1-datum":
        # pick one output embedding randomly
        idx = np.random.randint(output.shape[0])
        emb = output[idx]

        # tile emb out-dim times
        prototypes = np.tile(emb, (args.out_dim, 1))
        # add noise
        noise = np.random.normal(0, 0.1, prototypes.shape)
        prototypes += noise
        prototypes = torch.tensor(prototypes).float()

    elif args.mode == "k++":
        prototypes = kmeans_plusplus(output, args.out_dim)[0]
        prototypes = torch.tensor(prototypes).float()

    # save the prototypes
    torch.save(prototypes, f"embeddings/prototypes_init_{args.name}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--out-dim", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    main(args)
