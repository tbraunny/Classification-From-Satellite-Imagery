import albumentations as A
from albumentations.pytorch import ToTensorV2

from faster_rcnn.config import MEAN, STD

def get_train_transform():
    return A.Compose(
        [
            A.RandomScale(scale_limit=(-0.9, 0.0), p=0.8),
            A.RandomSizedBBoxSafeCrop(height=360, width=640, p=0.5), # Matched with dataset's resolution
            # A.Resize(height=640, width=360, p=1.0), # Preprocessed by source
            # A.RandomRotate90(p=0.5), # Preprocessed by source
            # A.HorizontalFlip(p=0.5), # Preprocessed by source
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            # A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            # A.GaussianBlur(blur_limit=(1, 3), p=0.3),  # Reduced blur effect
            # A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),  # Reduced noise intensity
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

def get_test_transform():
    return A.Compose(
        [
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

def get_inference_transform():
    return A.Compose(
        [
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )
