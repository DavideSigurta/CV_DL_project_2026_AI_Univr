"""
VisDrone dataset classes — YOLO+ format loader + COCO-format adapter for pycocotools evaluation.

Annotation format (7 columns, extended YOLO):
    <class_id> <cx> <cy> <w> <h> <occlusion> <truncation>

All normalized to [0, 1] relative to image size.
Columns 6-7 are preserved from original VisDrone annotations for E4 analysis.

Boxes returned by __getitem__ are in absolute pixel coordinates (xyxy format),
as required by torchvision detection models.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Optional


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


class VisDroneDataset(Dataset):
    """
    Load VisDrone images and YOLO+ annotations.

    Each annotation file (.txt) contains one row per object:
        <class_id> <cx> <cy> <w> <h> <occlusion> <truncation>

    Columns 1-5: standard YOLO format (normalized to [0,1]).
    Columns 6-7: occlusion and truncation flags (for E4 analysis).

    Returns (image, target) where:
        image: PIL Image or Tensor (after transforms)
        target: dict with keys 'boxes' (Nx4 xyxy abs), 'labels' (N),
                'occlusion' (N), 'truncation' (N), 'image_id'
    """

    def __init__(self, img_dir: Path, ann_dir: Path, transforms=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.images = sorted(self.img_dir.glob("*.jpg"))
        if len(self.images) == 0:
            raise FileNotFoundError(f"No .jpg images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Parse YOLO+ annotation
        ann_path = self.ann_dir / img_path.with_suffix(".txt").name
        boxes, labels, occlusions, truncations = [], [], [], []

        if ann_path.exists():
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx = float(parts[1]) * orig_w
                    cy = float(parts[2]) * orig_h
                    bw = float(parts[3]) * orig_w
                    bh = float(parts[4]) * orig_h
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)
                    occlusions.append(int(parts[5]) if len(parts) >= 7 else -1)
                    truncations.append(int(parts[6]) if len(parts) >= 7 else -1)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "occlusion": torch.as_tensor(occlusions, dtype=torch.int64),
            "truncation": torch.as_tensor(truncations, dtype=torch.int64),
            "image_id": torch.as_tensor([idx], dtype=torch.int64),
            "orig_size": torch.as_tensor([orig_h, orig_w], dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def visdrone_to_coco_json(data_dir: Path = None, split: str = "val",
                          img_dir: Path = None, ann_dir: Path = None) -> dict:
    """
    Convert VisDrone YOLO annotations to COCO JSON format for pycocotools evaluation.

    Reads YOLO+ .txt annotations and builds an in-memory COCO JSON dict with:
        - images: id, file_name, width, height
        - annotations: id, image_id, bbox (x,y,w,h abs), area, category_id, occlusion, truncation
        - categories: id -> name (10 VisDrone classes)

    Boxes in COCO format are [x, y, width, height] (absolute pixels).
    Annotation id is globally unique (accumulated across all images).

    Args:
        data_dir: Path to VisDrone2019-DET directory (used with split)
        split: 'train' or 'val' (used with data_dir)
        img_dir: Explicit image directory override (takes precedence over data_dir+split)
        ann_dir: Explicit annotation directory override (takes precedence over data_dir+split)

    Returns:
        COCO JSON-compatible dictionary
    """
    if img_dir is not None and ann_dir is not None:
        pass  # use explicit paths
    else:
        img_dir = data_dir / "images" / split
        ann_dir = data_dir / "annotations" / split

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(VISDRONE_CLASSES)
    ]

    images = []
    annotations = []
    ann_id = 1

    img_files = sorted(img_dir.glob("*.jpg"))
    for img_id, img_path in enumerate(img_files, start=1):
        # Get image size
        with Image.open(img_path) as img:
            w, h = img.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        # Parse annotations
        ann_path = ann_dir / img_path.with_suffix(".txt").name
        if not ann_path.exists():
            continue

        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                occlusion = int(parts[5]) if len(parts) >= 7 else -1
                truncation = int(parts[6]) if len(parts) >= 7 else -1

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x1, y1, bw, bh],  # COCO format: [x, y, w, h]
                    "area": bw * bh,
                    "iscrowd": 0,
                    "occlusion": occlusion,
                    "truncation": truncation,
                })
                ann_id += 1

    # Sanity check: clamp tiny floating-point negatives to 0, reject truly negative
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            raise ValueError(f"Zero/negative area bbox: {ann}")
        if x < -0.1 or y < -0.1:
            raise ValueError(f"Negative coordinates in GT (beyond fp tolerance): {ann}")
        # Clamp tiny fp negatives to 0
        ann["bbox"][0] = max(0.0, x)
        ann["bbox"][1] = max(0.0, y)

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
