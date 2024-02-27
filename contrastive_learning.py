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
batch_size = 512
n_epochs = 200

model_name = "resnet18"

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.08, 1.0),
            ratio=(1.0, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

datapipe_1 = contrastive_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2013.zip",
    num_images=NUM_TRAIN,
    transforms=train_transform,
    ignore_mix=True,
)
datapipe_2 = contrastive_datapipe(
    "/mimer/NOBACKUP/groups/naiss2023-5-75/WHOI_Planktons/2014.zip",
    num_images=NUM_TEST,
    transforms=train_transform,
    ignore_mix=True,
)
# combine the two datapipe
datapipe = datapipe_1.concat(datapipe_2)

# create a dataloader

train_dataloader = DataLoader(
    datapipe, batch_size=batch_size, shuffle=True, num_workers=12
)

if model_name == "resnet18":
    model = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=True)
    model.fc = torch.nn.Identity()
else:
    raise ValueError(f"Model {model_name} not supported")

# projection head which will be removed after training
projection_head = torch.nn.Sequential(
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 128),
)

# combine the model and the projection head
model = torch.nn.Sequential(model, projection_head)

# optimizer
max_lr = 0.03 * batch_size / 256
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# lr scheduler with linear warmup and cosine decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=n_epochs,
    steps_per_epoch=len(train_dataloader),
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
    for img1, img2, _ in train_dataloader:
        img1, img2 = img1.to(device), img2.to(device)
        optimizer.zero_grad()
        img = torch.cat((img1, img2), dim=0)
        output = model(img)
        loss = criterion(output)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")

# remove the projection head
model = model[0]
projection_head = model[1]

# save the model
torch.save(model.state_dict(), f"finetune_{model_name}.pth")
torch.save(projection_head.state_dict(), f"finetune_{model_name}_ph.pth")
