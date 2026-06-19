"""
SAHI (Slicing Aided Hyper Inference) tiling wrapper for E3.

Slices high-resolution images into overlapping patches, runs detector on each,
merges predictions via global NMS.

Reference: Akyon et al. 2022 — https://github.com/obss/sahi
"""

import numpy as np
import torch


def slice_image(image: np.ndarray, slice_height: int = 640, slice_width: int = 640,
                overlap_ratio: float = 0.2) -> list:
    """
    Slice image into overlapping patches.

    Args:
        image: (H, W, 3) RGB array
        slice_height, slice_width: patch dimensions in pixels
        overlap_ratio: fraction of overlap between adjacent patches (0 to 1)

    Returns:
        list of dicts: { 'image': patch_array, 'x1', 'y1', 'x2', 'y2' }
        where (x1,y1)-(x2,y2) are patch coordinates in the original image
    """
    H, W = image.shape[:2]
    stride_h = int(slice_height * (1 - overlap_ratio))
    stride_w = int(slice_width * (1 - overlap_ratio))
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    patches = []
    for y in range(0, H, stride_h):
        for x in range(0, W, stride_w):
            x1 = x
            y1 = y
            x2 = min(x + slice_width, W)
            y2 = min(y + slice_height, H)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            patch = image[y1:y2, x1:x2]
            # Pad patch if smaller than expected slice size
            if patch.shape[0] < slice_height or patch.shape[1] < slice_width:
                pad_h = max(0, slice_height - patch.shape[0])
                pad_w = max(0, slice_width - patch.shape[1])
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)),
                               mode="constant", constant_values=0)
            patches.append({
                "image": patch,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
            })
    return patches


def _extract_predictions(results, patch, conf_thr: float) -> list:
    """
    Extract predictions from a model output and map to original image coords.

    Works with torchvision FRCNN output format: list of dicts with
    'boxes', 'scores', 'labels' keys.

    Args:
        results: model output (list of dicts with boxes/scores/labels)
        patch: patch dict with 'x1', 'y1' offset in original image
        conf_thr: confidence threshold

    Returns:
        list of [x1, y1, x2, y2, score, class_id] in original image coords
    """
    preds = []
    for result in results:
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if score < conf_thr:
                continue
            x1, y1, x2, y2 = box
            x1 = x1 + patch["x1"]
            y1 = y1 + patch["y1"]
            x2 = x2 + patch["x1"]
            y2 = y2 + patch["y1"]
            preds.append([float(x1), float(y1), float(x2), float(y2),
                          float(score), int(label)])
    return preds


def _extract_yolo_predictions(results, patch, conf_thr: float) -> list:
    """
    Extract predictions from Ultralytics YOLO output and map to original image coords.

    YOLO returns list[Results]; each Results has .boxes with .xyxy, .conf, .cls.

    Args:
        results: Ultralytics Results list (len=1 for single-image input)
        patch: patch dict with 'x1', 'y1' offset in original image
        conf_thr: confidence threshold

    Returns:
        list of [x1, y1, x2, y2, score, class_id] in original image coords
    """
    preds = []
    for res in results:
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        labels = res.boxes.cls.cpu().numpy().astype(int)
        for box, score, label in zip(boxes, scores, labels):
            if score < conf_thr:
                continue
            x1, y1, x2, y2 = box
            x1 = x1 + patch["x1"]
            y1 = y1 + patch["y1"]
            x2 = x2 + patch["x1"]
            y2 = y2 + patch["y1"]
            preds.append([float(x1), float(y1), float(x2), float(y2),
                          float(score), int(label)])
    return preds


def run_sahi_inference(model, image: np.ndarray, config: dict) -> list:
    """
    Run SAHI inference on a single image using a torchvision FRCNN model.

    Slices image into overlapping patches, runs model on each,
    maps predictions back to original coordinates, merges via per-class NMS.

    Args:
        model: torchvision FRCNN model (returns list[dict] with boxes/scores/labels)
        image: (H, W, 3) RGB array
        config: SAHI parameters dict

    Returns:
        list of merged predictions: [x1, y1, x2, y2, score, class_id]
    """
    slice_h = config.get("slice_height", 640)
    slice_w = config.get("slice_width", 640)
    overlap = config.get("overlap_height_ratio", config.get("overlap_ratio", 0.2))
    conf_thr = config.get("confidence_threshold", 0.25)
    nms_thr = config.get("nms_iou_threshold", 0.5)

    patches = slice_image(image, slice_h, slice_w, overlap)
    all_preds = []

    for patch in patches:
        results = model(patch["image"])
        all_preds.extend(_extract_predictions(results, patch, conf_thr))

    merged = global_nms(all_preds, nms_thr)
    return merged


def run_sahi_yolo(model, image: np.ndarray, config: dict) -> list:
    """
    Run SAHI inference on a single image using an Ultralytics YOLO model.

    Same pipeline as run_sahi_inference but handles YOLO Results output format.
    Model should be a YOLO instance (not the underlying nn.Module).

    Args:
        model: Ultralytics YOLO model
        image: (H, W, 3) RGB array (uint8, 0-255)
        config: SAHI parameters dict

    Returns:
        list of merged predictions: [x1, y1, x2, y2, score, class_id]
    """
    slice_h = config.get("slice_height", 640)
    slice_w = config.get("slice_width", 640)
    overlap = config.get("overlap_height_ratio", config.get("overlap_ratio", 0.2))
    conf_thr = config.get("confidence_threshold", 0.25)
    nms_thr = config.get("nms_iou_threshold", 0.5)

    patches = slice_image(image, slice_h, slice_w, overlap)
    all_preds = []

    for patch in patches:
        results = model(patch["image"])  # YOLO accepts numpy arrays
        all_preds.extend(_extract_yolo_predictions(results, patch, conf_thr))

    merged = global_nms(all_preds, nms_thr)
    return merged


def global_nms(predictions: list, iou_threshold: float = 0.5) -> list:
    """
    Apply per-class NMS to merge overlapping predictions from different patches.

    Uses torchvision.ops.nms per class to avoid over-suppressing different
    object types that happen to overlap spatially (e.g. pedestrian inside car).

    Args:
        predictions: list of [x1, y1, x2, y2, score, class_id]
        iou_threshold: NMS IoU threshold

    Returns:
        Filtered list after per-class NMS
    """
    from torchvision.ops import nms

    if len(predictions) == 0:
        return []

    arr = np.array(predictions)
    merged = []
    # Per-class NMS — prevents merging different object types
    for cls_id in np.unique(arr[:, 5]):
        mask = arr[:, 5] == cls_id
        cls_preds = arr[mask]
        boxes = torch.as_tensor(cls_preds[:, :4], dtype=torch.float32)
        scores = torch.as_tensor(cls_preds[:, 4], dtype=torch.float32)
        keep = nms(boxes, scores, iou_threshold).numpy()
        for i in keep:
            merged.append(predictions[int(np.where(mask)[0][int(i)])])

    # Sort by score descending
    merged.sort(key=lambda x: x[4], reverse=True)
    return merged
