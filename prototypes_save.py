import argparse

import numpy as np
import torch

from model import LightningMuContrastive


def main(args):
    model = LightningMuContrastive(
        args.name,
        out_dim=args.out_dim,
        loss=None,
        n_epochs=0,
        uuid=False,
        arch="vit",
    )
    model.teacher_backbone.load_state_dict(
        torch.load(f"model_weights/{args.name}_tb.pth")
    )
    model.teacher_head.load_state_dict(torch.load(f"model_weights/{args.name}_th.pth"))
    model.student_backbone.load_state_dict(
        torch.load(f"model_weights/{args.name}_sb.pth")
    )
    model.student_head.load_state_dict(torch.load(f"model_weights/{args.name}_sh.pth"))

    for i, param in enumerate(model.teacher_head.parameters()):
        # save the teacher prototypes in numpy format
        np.save(
            f"embeddings/prototypes_{args.name}_t{i}.npy", param.cpu().detach().numpy()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--out-dim", type=int, required=True)
    args = parser.parse_args()

    main(args)
