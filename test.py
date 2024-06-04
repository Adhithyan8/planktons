from data import CUBDataModule
from transforms import CONTRASTIVE_TRANSFORM, INFERENCE_TRANSFORM
import matplotlib.pyplot as plt

dm = CUBDataModule(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/CUB/CUB_200_2011",
    CONTRASTIVE_TRANSFORM,
    INFERENCE_TRANSFORM,
    batch_size=16,
    uuid=False,
)

train_loader = dm.train_dataloader()
for i, (x, y) in enumerate(train_loader):
    print(x.shape, y.shape)
    print(y)
    plt.imshow(x[0].permute(1, 2, 0))
    plt.savefig(f"test_{i}.png")
    if i == 0:
        break
