"""
SAHI (Slicing Aided Hyper Inference) tiling wrapper for E3.

Slices high-resolution images into overlapping patches, runs detector on each,
merges predictions via global NMS.

Reference: Akyon et al. 2022 — https://github.com/obss/sahi
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional


def slice_image(image: np.ndarray, slice_height: int = 640, slice_width: int = 640,
                overlap_ratio: float = 0.2) -> list:
    """
    Slice image into overlapping patches.

    Args:
        image: (H, W, 3) RGB array
        slice_height, slice_width: patch dimensions
        overlap_ratio: fraction of overlap between adjacent patches

    Returns:
        list of dicts: { 'image': patch_array, 'x1', 'y1', 'x2', 'y2' } in original coords
    """
    raise NotImplementedError


def run_sahi_inference(model, image: np.ndarray, config: dict) -> list:
    """
    Run SAHI inference on a single image.

    Args:
        model: detection model (E1a checkpoint)
        image: (H, W, 3) RGB array
        config: SAHI parameters dict

    Returns:
        list of predictions merged in original image coordinates
    """
    raise NotImplementedError


def global_nms(predictions: list, iou_threshold: float = 0.5) -> list:
    """
    Apply global NMS to merge overlapping predictions from different patches.

    Args:
        predictions: list of [x1, y1, x2, y2, score, class_id]
        iou_threshold: NMS IoU threshold

    Returns:
        Filtered list after NMS
    """
    raise NotImplementedError
