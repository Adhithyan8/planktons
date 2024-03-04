import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import contrastive_datapipe, InfoNCECosine

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
batch_size = 1024
n_epochs = 250

model_name = "vitb14-dinov2"

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.1, 1.0),
            ratio=(1.0, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

datapipe = contrastive_datapipe(
    [
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
        "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    ],
    num_images=NUM_TOTAL,
    transforms=train_transform,
    ignore_mix=True,
)

# create a dataloader
train_dataloader = DataLoader(
    datapipe, batch_size=batch_size, shuffle=True, num_workers=8
)

if model_name == "resnet18":
    backbone = torch.hub.load(
        "pytorch/vision:v0.9.0",
        "resnet18",
        pretrained=True,
    )
    backbone.fc = torch.nn.Identity()
elif model_name == "vitb14-dinov2":
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14_reg",
    )
else:
    raise ValueError(f"Model {model_name} not supported")

# freeze early layers
if model_name == "resnet18":
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.layer4.parameters():
        param.requires_grad = True
    for param in backbone.fc.parameters():
        param.requires_grad = True
elif model_name == "vitb14-dinov2":
    for param in backbone.parameters():
        param.requires_grad = False
    for param in backbone.blocks[-1].parameters():
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
elif model_name == "vitb14-dinov2":
    projection_head = torch.nn.Sequential(
        torch.nn.Linear(768, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 128),
    )
else:
    raise ValueError(f"Model {model_name} not supported")

# combine the model and the projection head
model = torch.nn.Sequential(backbone, projection_head)

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# lr scheduler with linear warmup and cosine decay
lr = 0.03 * (batch_size / 256)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=n_epochs,
    steps_per_epoch=int(len(train_dataloader)) + 1,  # 1 due to a weird bug in datapipe
    pct_start=0.05,
    div_factor=1e4,
    final_div_factor=1e4,
)

# loss
criterion = InfoNCECosine(temperature=0.5)

# param count
print(f"Model: {model_name}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

# train the model
model.train().to(device)
for epoch in range(n_epochs):
    for i, (img1, img2, _) in enumerate(train_dataloader):
        img1, img2 = img1.to(device), img2.to(device)
        optimizer.zero_grad()
        img = torch.cat((img1, img2), dim=0)
        output = model(img)
        loss = criterion(output)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
            )

# save the model
torch.save(model[0].state_dict(), f"finetune_{model_name}_backbone.pth")
torch.save(model[1].state_dict(), f"finetune_{model_name}_head.pth")
