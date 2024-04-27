import albumentations as A
import torch
from torch.utils.data import DataLoader

from utils import InfoNCECosine, SemiSupervisedContrastive, SupervisedContrastive, InfoCNECauchy, Padding, contrastive_datapipe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
2013: 421238
2013: 115951 (ignore mix)
2014: 329832
2014: 63676 (ignore mix)
"""
# magic numbers
NUM_TRAIN = 115951
NUM_TEST = 63676
NUM_TOTAL = NUM_TRAIN + NUM_TEST
batch_size = 2048
n_epochs = 250
model_name = "resnet18"
pad = Padding.REFLECT

contrastive_transform = A.Compose(
    [
        # shape augmentation
        A.ShiftScaleRotate(p=0.5),
        A.Flip(p=0.5),
        # cutout
        A.CoarseDropout(fill_value=200),
        # color augmentation
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.AdvancedBlur(),
            ],
        ),
        # below are always applied
        A.ToRGB(),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0),
        A.RandomResizedCrop(128, 128, scale=(0.2, 1.0)),
    ]
)

datapipe = contrastive_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=contrastive_transform,
    padding=pad,
    ignore_mix=True,
    mask_label=True, # for semi-supervised learning
)

# create a dataloader
dataloader = DataLoader(datapipe, batch_size=batch_size, shuffle=True, num_workers=16)

if model_name == "resnet18":
    backbone = torch.hub.load(
        "pytorch/vision:v0.9.0",
        "resnet18",
        pretrained=True,
    )
    backbone.fc = torch.nn.Identity()
else:
    raise ValueError(f"Model {model_name} not supported")

# freeze early layers
if model_name == "resnet18":
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.layer4.parameters():
        param.requires_grad = True
else:
    raise ValueError(f"Model {model_name} not supported")

# projection head which will be removed after training
if model_name == "resnet18":
    projection_head = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 128),
    )
else:
    raise ValueError(f"Model {model_name} not supported")

# combine the model and the projection head
model = torch.nn.Sequential(backbone, projection_head)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# lr scheduler with linear warmup and cosine decay
lr = 0.03 * (batch_size / 256) * 0.1
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=n_epochs,
    steps_per_epoch=96,  # weird bug
    pct_start=0.02,
    div_factor=1e4,  # start close to 0
    final_div_factor=1e4,  # end close to 0
)

# loss
criterion1 = InfoNCECosine(temperature=0.5)
criterion2 = SupervisedContrastive(temperature=0.5)
criterion = InfoCNECauchy(temperature=1.0, mode="selfsupervised")
lamda = 0.5

print(f"Model: {model_name}")

# train the model
model.train().to(device)
for epoch in range(n_epochs):
    for i, (img1, img2, id) in enumerate(dataloader):
        img1, img2, id = img1.to(device), img2.to(device), id.to(device)
        optimizer.zero_grad()
        img = torch.cat((img1, img2), dim=0)
        id = torch.cat((id, id), dim=0)
        output = model(img)
        # unsupervised_loss = criterion1(output)
        # supervised_loss = criterion2(output, id)
        # loss = (1 - lamda) * unsupervised_loss + lamda * supervised_loss
        loss = criterion(output, id)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            # print(f"Epoch [{epoch+1}/{n_epochs}], Unsup_loss: {unsupervised_loss.item():.4f}, Sup_loss: {supervised_loss.item():.4f}, Loss: {loss.item():.4f}")
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

# save the model
torch.save(model[0].state_dict(), f"selfcy_{model_name}_backbone.pth")
torch.save(model[1].state_dict(), f"selfcy_{model_name}_head.pth")
