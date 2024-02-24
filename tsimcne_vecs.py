import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import AutoImageProcessor, Dinov2Model
from tsimcne.tsimcne import TSimCNE

# gets both the train and test datasets combined



# edit the forward method to return the pooler output
class Dinov2ModelModified(Dinov2Model):
    def forward(self, x):
        output = super().forward(x)
        return output.pooler_output.squeeze()


backbone = Dinov2ModelModified.from_pretrained("facebook/dinov2-base")
# freeze the backbone
for name, param in backbone.named_parameters():
    param.requires_grad = False


# mlp with one hidden layer as the projection head
# need layers as attributes of the model for the tsimcne object
class FC(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


projection_head = FC(768, 1024, 128)

# tsimcne object (try different augmentation strategies)
tsimcne = TSimCNE(
    backbone=backbone,
    projection_head=projection_head,
    total_epochs=[500, 50, 250],  # resnet was run with [500, 50, 250] and 32x32 images
    batch_size=256,  # cannot increase more than 256
)

# train
tsimcne.fit(planktons_dataset)

# map the images to 2D vectors
vecs = tsimcne.transform(planktons_dataset)

# save the output vectors
np.save("embeddings/vec_tsimcne_dinov2.npy", vecs)
