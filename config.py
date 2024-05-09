import albumentations as A

CONTRASTIVE_TRANSFORM = A.Compose(
    [
        A.ShiftScaleRotate(p=0.5),
        A.Flip(p=0.5),
        A.CoarseDropout(fill_value=200),
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.AdvancedBlur(),
            ],
        ),
        A.ToRGB(),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0),
        A.RandomResizedCrop(128, 128, scale=(0.2, 1.0)),
    ]
)

INFERENCE_TRANSFORM = A.Compose(
    [
        A.ToRGB(),
        A.ToFloat(max_value=255),
        A.Normalize(max_pixel_value=1.0),
        A.Resize(256, 256),  # inference is at higher res than training
    ]
)
