import os
import pandas as pd
import shutil
from PIL import Image
import torchvision.transforms as T

# create folder "planktons_pytorch" in /local_storage/users/adhkal/
base_path = "/local_storage/users/adhkal/"
if not os.path.exists(os.path.join(base_path, "planktons_pytorch")):
    os.mkdir(os.path.join(base_path, "planktons_pytorch"))

# make a dataframe with columns "image" and "label"
# and save it as annotations.csv in /local_storage/users/adhkal/planktons_pytorch
df = pd.DataFrame(columns=["image", "label"])

# make subfolder "train" and "test" in /local_storage/users/adhkal/planktons_pytorch
if not os.path.exists(os.path.join(base_path, "planktons_pytorch", "train")):
    os.mkdir(os.path.join(base_path, "planktons_pytorch", "train"))
if not os.path.exists(os.path.join(base_path, "planktons_pytorch", "test")):
    os.mkdir(os.path.join(base_path, "planktons_pytorch", "test"))

# raw train images are in /local_storage/users/adhkal/planktons_dataset/data/2013/
# the subfolders in /local_storage/users/adhkal/planktons_dataset/data/2013/ are the labels
label_map = {}
label_names = sorted(
    os.listdir(os.path.join("/local_storage/users/adhkal/planktons_dataset/data/2013/"))
)
# create the label map
for i, label in enumerate(label_names):
    label_map[label] = i

# we need to move the images to /local_storage/users/adhkal/planktons_pytorch/train
# and add the image path and label to the dataframe
for label in label_names:
    # get all the images in the label folder
    images = os.listdir(
        os.path.join("/local_storage/users/adhkal/planktons_dataset/data/2013/", label)
    )
    # move the images to /local_storage/users/adhkal/planktons_pytorch/train
    for image in images:
        shutil.move(
            os.path.join(
                "/local_storage/users/adhkal/planktons_dataset/data/2013/", label, image
            ),
            os.path.join(
                "/local_storage/users/adhkal/planktons_pytorch/train",
                image,
            ),
        )
    # add the image path and label to the dataframe
    for image in images:
        df = df.append(
            {
                "image": os.path.join(
                    "/local_storage/users/adhkal/planktons_pytorch/train", image
                ),
                "label": label_map[label],
            },
            ignore_index=True,
        )

# raw test images are in /local_storage/users/adhkal/planktons_dataset/data/2014/
# the subfolders in /local_storage/users/adhkal/planktons_dataset/data/2014/ are the labels
# use same label map as train
# we need to move the images to /local_storage/users/adhkal/planktons_pytorch/test
# and add the image path and label to the dataframe
for label in label_names:
    # get all the images in the label folder
    images = os.listdir(
        os.path.join("/local_storage/users/adhkal/planktons_dataset/data/2014/", label)
    )
    # move the images to /local_storage/users/adhkal/planktons_pytorch/test
    for image in images:
        shutil.move(
            os.path.join(
                "/local_storage/users/adhkal/planktons_dataset/data/2014/", label, image
            ),
            os.path.join(
                "/local_storage/users/adhkal/planktons_pytorch/test",
                image,
            ),
        )
    # add the image path and label to the dataframe
    for image in images:
        df = df.append(
            {
                "image": os.path.join(
                    "/local_storage/users/adhkal/planktons_pytorch/test", image
                ),
                "label": label_map[label],
            },
            ignore_index=True,
        )

# save the dataframe as annotations.csv
df.to_csv(os.path.join(base_path, "planktons_pytorch", "annotations.csv"))

# preprocess the dataset
# for images in train and test, resize to 224x224 and convert to RGB
# save the processed images in /local_storage/users/adhkal/planktons_pytorch/train and /local_storage/users/adhkal/planktons_pytorch/test
for image in os.listdir(os.path.join(base_path, "planktons_pytorch", "train")):
    img = Image.open(
        os.path.join(base_path, "planktons_pytorch", "train", image)
    ).convert("RGB")
    img = T.Resize((224, 224))(img)
    img.save(os.path.join(base_path, "planktons_pytorch", "train", image))

for image in os.listdir(os.path.join(base_path, "planktons_pytorch", "test")):
    img = Image.open(
        os.path.join(base_path, "planktons_pytorch", "test", image)
    ).convert("RGB")
    img = T.Resize((224, 224))(img)
    img.save(os.path.join(base_path, "planktons_pytorch", "test", image))
