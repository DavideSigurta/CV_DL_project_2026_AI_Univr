"""
Detection metrics — mAP, AP_small/medium/large, PR curve computation.

All mAP values computed with pycocotools using the official COCO evaluation protocol.
COCOeval returns 12 summary stats; we extract the relevant ones and also compute
per-class AP from the raw precision array.
"""

import numpy as np
from typing import Optional


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def compute_map(predictions: list, ground_truths: dict,
                iou_thresholds: Optional[list] = None) -> dict:
    """
    Compute COCO-style mAP@0.5:0.95 and per-scale AP.

    Args:
        predictions: list of COCO-format prediction dicts, each with keys:
            image_id, bbox ([x,y,w,h] absolute), score, category_id
        ground_truths: COCO JSON dict with 'images', 'annotations', 'categories'
        iou_thresholds: optional list of IoU thresholds (default: 0.5:0.05:0.95)

    Returns:
        dict with keys: mAP@0.5:0.95, mAP@0.5, mAP@0.75,
        AP_small, AP_medium, AP_large, AP_per_class (list of 10)
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truths
    coco_gt.createIndex()

    if len(predictions) == 0:
        return {
            "mAP@0.5:0.95": 0.0,
            "mAP@0.5": 0.0,
            "mAP@0.75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AP_per_class": [0.0] * len(VISDRONE_CLASSES),
        }

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if iou_thresholds is not None:
        coco_eval.params.iouThrs = np.array(iou_thresholds)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # coco_eval.stats: [AP@0.5:0.95, AP@0.5, AP@0.75, AP_small, AP_medium, AP_large,
    #                   AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large]
    stats = coco_eval.stats

    # Per-class AP: precision[T=IoU, R=recall, K=class, A=area, M=maxDet]
    precisions = coco_eval.eval["precision"]
    ap_per_class = []
    for cls_id in range(precisions.shape[2]):
        ap = precisions[:, :, cls_id, 0, -1]  # all IoU, all recall, class, all areas, max=100
        valid = ap[ap > -1]
        ap_per_class.append(float(valid.mean()) if len(valid) > 0 else 0.0)

    return {
        "mAP@0.5:0.95": float(stats[0]),
        "mAP@0.5": float(stats[1]),
        "mAP@0.75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AP_per_class": ap_per_class,
    }


def compute_pr_curve(predictions: list, ground_truths: dict, class_id: int) -> tuple:
    """
    Compute precision-recall curve for a single class.

    Args:
        predictions: list of COCO-format prediction dicts
        ground_truths: COCO JSON dict
        class_id: class index (0-9 for VisDrone)

    Returns:
        (precision, recall, thresholds) arrays from COCOeval
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truths
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.maxDets = [100]
    coco_eval.params.areaRng = [[0, 1e5]]  # all areas
    coco_eval.params.iouThrs = [0.5]  # single IoU threshold for PR curve
    coco_eval.params.recThrs = np.linspace(0, 1, 101)  # 101 recall points
    coco_eval.evaluate()
    coco_eval.accumulate()

    # precision[0, :, class_id, 0, -1] = precision at IoU=0.5 across recall bins
    precision = coco_eval.eval["precision"][0, :, class_id, 0, -1].copy()
    recall = coco_eval.params.recThrs.copy()

    return precision, recall


def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision from precision-recall curve (trapezoidal interpolation).

    Uses 101-point COCO interpolation: average precision at 101 equally-spaced
    recall levels [0, 0.01, ..., 1]. At each recall level r, use max precision
    for any recall >= r.

    Args:
        precision: array of precision values
        recall: array of recall values (0 to 1)

    Returns:
        AP value (float)
    """
    recall_levels = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_levels:
        mask = recall >= r
        if mask.any():
            ap += np.max(precision[mask])
        # else 0 contribution
    return ap / 101.0


def extract_best_metrics(epochs: list, metric_key: str = "val_mAP@0.5") -> dict:
    """
    Extract best-epoch validation metrics from metrics.jsonl epoch list.

    metrics.jsonl stores per-epoch dicts with ``val_``-prefixed keys
    (e.g. ``val_mAP@0.5:0.95``). Downstream code (notebook cells,
    comparison tables) expects short-form keys without prefix
    (e.g. ``mAP@0.5:0.95``).  This helper converts between the two.

    Args:
        epochs: list of dicts, each from one line of metrics.jsonl.
            Required keys: ``epoch``, ``val_mAP@0.5:0.95``,
            ``val_mAP@0.5``, ``val_mAP@0.75``, ``val_AP_small``,
            ``val_AP_medium``, ``val_AP_large``, ``val_AP_per_class``.
        metric_key: which metric to use for selecting the best epoch
            (default ``val_mAP@0.5``).

    Returns:
        dict with short-form keys:
        ``mAP@0.5:0.95``, ``mAP@0.5``, ``mAP@0.75``,
        ``AP_small``, ``AP_medium``, ``AP_large``, ``AP_per_class``.
    """
    best = max(epochs, key=lambda e: e[metric_key])

    # Map: val_XXX -> XXX (strip the ``val_`` prefix)
    short_keys = [
        "mAP@0.5:0.95", "mAP@0.5", "mAP@0.75",
        "AP_small", "AP_medium", "AP_large", "AP_per_class",
    ]
    return {k: best[f"val_{k}"] for k in short_keys}
