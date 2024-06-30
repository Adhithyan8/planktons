import numpy as np


def main(args):
    output = np.load(f"embeddings/output_{args.name}.npy")
    labels = np.load(f"embeddings/labels_{args.name}.npy")

    # normalize output
    output = output / np.linalg.norm(output, axis=1)[:, None]

    centers = np.zeros((args.out_dim, output.shape[1]))
    for i in range(args.out_dim):
        if output[labels == i].shape[0] == 0:
            centers[i] = np.random.rand(output.shape[1])
        else:
            centers[i] = np.mean(output[labels == i], axis=0)

    # normalize centers
    centers = centers / np.linalg.norm(centers, axis=1)[:, None]

    # save centers
    np.save(f"embeddings/centers_{args.name}.npy", centers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="semicosineWHOIvit")
    parser.add_argument("--out_dim", type=int, default=110)
    args = parser.parse_args()
    main(args)
