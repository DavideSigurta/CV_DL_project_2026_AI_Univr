"""
Augmentation pipelines for torchvision-based trainers (FRCNN).

YOLO uses built-in Ultralytics augmentations — not affected by these transforms.

Transforms use the torchvision detection pattern: callable(image, target) -> (image, target).
Resize preserves aspect ratio (longest side = target_size). Bbox coordinates are scaled
to match the resized image.

Normalization is handled internally by the torchvision model (ImageNet stats in forward pass).
"""

import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter


class DetectionCompose:
    """Compose transforms that operate on (image, target) tuples."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        return f"DetectionCompose({self.transforms})"


class DetectionResize:
    """Resize image keeping aspect ratio (longest side = target_size). Scale bboxes."""

    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, image, target):
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = F.resize(image, (new_h, new_w))

        if "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
            # Clamp to image boundaries
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
            target["boxes"] = boxes

        return image, target

    def __repr__(self):
        return f"DetectionResize(target_size={self.target_size})"


class DetectionToTensor:
    """Convert PIL image to tensor. Target unchanged."""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ImageOnlyTransform:
    """Wrap a single-argument transform (e.g. ColorJitter) for (image, target) pipeline."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        return self.transform(image), target

    def __repr__(self):
        return f"ImageOnlyTransform({self.transform})"


class DetectionRandomHorizontalFlip:
    """Random horizontal flip (image + bboxes)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            if isinstance(image, torch.Tensor):
                w = image.shape[-1]  # (C, H, W)
                image = F.hflip(image)
            else:
                w = image.width
                image = F.hflip(image)
            if "boxes" in target and target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def get_frcnn_train_transforms(img_size: int = 800):
    """
    FRCNN training transforms: resize preserving aspect ratio + light augs.

    Args:
        img_size: max dimension (longest side) after resize

    Returns:
        DetectionCompose for training (image, target) -> (image, target)
    """
    return DetectionCompose([
        DetectionResize(img_size),
        DetectionToTensor(),
        ImageOnlyTransform(ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)),
        DetectionRandomHorizontalFlip(p=0.5),
    ])


def get_frcnn_val_transforms(img_size: int = 800):
    """
    FRCNN validation transforms: resize preserving aspect ratio + to tensor.

    Args:
        img_size: max dimension (longest side) after resize

    Returns:
        DetectionCompose for validation (image, target) -> (image, target)
    """
    return DetectionCompose([
        DetectionResize(img_size),
        DetectionToTensor(),
    ])
