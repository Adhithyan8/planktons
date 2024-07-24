from torch import nn


class CosineClassifier(nn.Module):
    """
    Returns the cosine similarity between the prototypes and the input.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        freeze_layer: int = -1,
    ):
        super().__init__()
        self.layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(in_dim, out_dim, bias=False)
        )
        self.layer.parametrizations.weight.original0.data.fill_(1)
        self.layer.parametrizations.weight.original0.requires_grad = False
        self.freeze_layer = freeze_layer

    def cancel_gradients(self, epoch: int):
        if epoch >= self.freeze_layer:
            return
        for param in self.layer.parameters():
            param.grad = None

    def forward(self, x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.layer(x)
        return x
