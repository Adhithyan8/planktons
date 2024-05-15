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
