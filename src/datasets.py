"""
VisDrone dataset classes — YOLO+ format loader + COCO-format adapter for pycocotools evaluation.

Annotation format (7 columns, extended YOLO):
    <class_id> <cx> <cy> <w> <h> <occlusion> <truncation>

All normalized to [0, 1] relative to image size.
    Columns 6-7 are preserved from original VisDrone annotations for E4 analysis.

import torch
from torch.utils.data import Dataset
from pathlib import Path


class VisDroneDataset(Dataset):
    """
    Load VisDrone images and YOLO+ annotations.

    Each annotation file (.txt) contains one row per object:
        <class_id> <cx> <cy> <w> <h> <occlusion> <truncation>

    Columns 1-5: standard YOLO format (normalized).
    Columns 6-7: occlusion and truncation flags (for E4 analysis).
    """

    def __init__(self, img_dir: Path, ann_dir: Path, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        # TODO: build image list
        # TODO: build annotation lookup
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def visdrone_to_coco_json(data_dir: Path, split: str = "val") -> dict:
    """
    Convert VisDrone YOLO annotations to COCO JSON format for pycocotools evaluation.

    Args:
        data_dir: Path to VisDrone2019-DET directory
        split: 'train' or 'val'

    Returns:
        COCO JSON-compatible dictionary with 'images', 'annotations', 'categories'
    """
    raise NotImplementedError
