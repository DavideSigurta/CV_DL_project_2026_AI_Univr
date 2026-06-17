"""
Shared augmentation pipelines for training and validation.

Designed for torchvision-based trainers (FRCNN). YOLO uses built-in Ultralytics augmentations
configured via config YAML.

Transforms only convert PIL → Tensor + optional light augs.
Normalization is handled internally by the detection model (torchvision normalizes
with ImageNet stats inside the model forward pass). Box coordinates stay in absolute
pixel space — the FRCNN model handles coordinate scaling internally.
"""

import torchvision.transforms as T


def get_train_transforms(img_size: int = 640):
    """
    Return training augmentation pipeline.

    Args:
        img_size: target image size (assumed square, used for Resize)

    Returns:
        torchvision Compose transform for training images
    """
    return T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        T.RandomHorizontalFlip(p=0.5),
    ])


def get_val_transforms(img_size: int = 640):
    """
    Return validation/test augmentation pipeline (resize + normalize only).

    Args:
        img_size: target image size (assumed square)

    Returns:
        torchvision Compose transform for validation images
    """
    return T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size)),
    ])
