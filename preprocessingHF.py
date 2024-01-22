import os
from PIL import Image

base_path = "/local_storage/users/adhkal/planktons_dataset/data/"
# contains subfolders 2013, 2014


def expand2square(pil_img, background_color):
    """
    To resize the images to 224x224, and convert to RGB
    """
    # convert pil_img to "rgb"
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    # resize to thumbnail (224, 224)
    pil_img.thumbnail((224, 224), Image.Resampling.LANCZOS)
    # create new image of desired size and color
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# preprocess the dataset
# for images in subfolders 2013 and 2014, resize to 224x224 and convert to RGB
for year in sorted(os.listdir(base_path)):
    for label in sorted(os.listdir(os.path.join(base_path, year))):
        for image in sorted(os.listdir(os.path.join(base_path, year, label))):
            img = Image.open(os.path.join(base_path, year, label, image))
            img = expand2square(img, (200, 200, 200))
            img.save(os.path.join(base_path, year, label, image), "PNG")

# HOPE THIS WORKS :D
