"""
Detection metrics — mAP, AP_small/medium/large, PR curve computation.

All mAP values computed with pycocotools using the official COCO evaluation protocol.
"""

import numpy as np
from typing import Optional


def compute_map(predictions: list, ground_truths: list, iou_thresholds: list = None) -> dict:
    """
    Compute COCO-style mAP@0.5:0.95 and per-scale AP.

    Args:
        predictions: list of dicts in COCO format
        ground_truths: list of dicts in COCO format
        iou_thresholds: list of IoU thresholds (default: 0.5:0.05:0.95)

    Returns:
        dict with keys: mAP@0.5, mAP@0.5:0.95, AP_small, AP_medium, AP_large, AP_per_class
    """
    raise NotImplementedError


def compute_pr_curve(predictions: list, ground_truths: list, class_id: int) -> tuple:
    """
    Compute precision-recall curve for a single class.

    Returns:
        (precision, recall, thresholds) arrays
    """
    raise NotImplementedError


def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision from precision-recall curve (trapezoidal interpolation).
    """
    raise NotImplementedError
