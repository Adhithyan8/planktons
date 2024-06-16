import albumentations as A

CONTRASTIVE_TRANSFORM = A.Compose(
    [
        A.ShiftScaleRotate(p=0.5),
        A.RandomResizedCrop(128, 128, scale=(0.2, 1.0)),
        A.Flip(p=0.5),
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.AdvancedBlur(),
            ],
        ),
        A.ToRGB(),
        A.Normalize(),
    ]
)

INFERENCE_TRANSFORM = A.Compose(
    [
        A.Resize(256, 256),  # inference is at higher res
        A.ToRGB(),
        A.Normalize(),
    ]
)

INFER_VIT_TRANSFORM = A.Compose(
    [
        A.Resize(252, 252),
        A.Normalize(),
    ]
)

CUB_CONTRASTIVE = A.Compose(
    [
        A.Resize(256, 256),
        # A.ShiftScaleRotate(p=0.5),
        # A.RandomResizedCrop(224, 224, scale=(0.3, 1.0)),
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

CUB_MU_TEACHER = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
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
