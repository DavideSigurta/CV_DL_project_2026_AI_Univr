"""
Shared augmentation pipelines for training and validation.
"""

import torch
import torchvision.transforms as T


def get_train_transforms(img_size: int = 640):
    """
    Return training augmentation pipeline.

    Args:
        img_size: target image size (assumed square)
    """
    raise NotImplementedError


def get_val_transforms(img_size: int = 640):
    """
    Return validation/test augmentation pipeline (resize + normalize only).
    """
    raise NotImplementedError
