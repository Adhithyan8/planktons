import albumentations as A

CONTRASTIVE_TRANSFORM = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
        A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ToRGB(),
        A.ColorJitter(),
        A.Normalize(),
    ]
)

INFERENCE_TRANSFORM = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.ToRGB(),
        A.Normalize(),
    ]
)

WHOI_TEACHER = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ToRGB(),
        A.Normalize(),
    ]
)

WHOI_STUDENT = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
        A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(),
        A.Solarize(),
        A.GaussianBlur(),
        A.ToRGB(),
        A.Normalize(),
    ]
)

CUB_CONTRASTIVE = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(),
        A.Normalize(),
    ]
)

CUB_INFERENCE = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(),
    ]
)

CUB_MU_TEACHER = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
    ]
)

CUB_MU_STUDENT = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(),
        A.Solarize(),
        A.GaussianBlur(),
        A.Normalize(),
    ]
)

CUB_MU_INFERENCE = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(),
    ]
)
