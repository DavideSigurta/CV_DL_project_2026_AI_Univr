"""
Faster R-CNN trainer — torchvision implementation for E2a and E2b experiments.

E2a: ResNet50 backbone, NO FPN (single-scale)
    - Manual backbone construction: remove avgpool+fc from resnet50
    - Single-scale RoI align (no FPN multi-scale)
E2b: ResNet50 backbone + FPN (Feature Pyramid Network)
    - Standard fasterrcnn_resnet50_fpn from torchvision

Inherits from BaseTrainer. Uses torchvision.models.detection.
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path

from .base_trainer import BaseTrainer


class FRCNNTrainer(BaseTrainer):
    """
    Trainer for Faster R-CNN experiments.

    Config determines:
    - E2a: fpn=False, no FPN backbone
    - E2b: fpn=True, standard torchvision FPN backbone
    """

    def __init__(self, config: dict, paths: dict):
        super().__init__(config, paths)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

    def _build_backbone_no_fpn(self):
        """Build ResNet50 backbone without FPN for E2a."""
        backbone = torchvision.models.resnet50(pretrained=True)
        # Remove avgpool and fc layers — keep conv layers up to layer4
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        return backbone

    def setup_model(self):
        """Initialize model, optimizer, scheduler, and data loaders."""
        import torchvision
        from src.datasets import VisDroneDataset
        from src.augmentations import get_train_transforms, get_val_transforms
        from src.env import get_paths as _gp

        paths = self.paths
        data_dir = paths["data"]
        use_fpn = self.config.get("fpn", True)
        imgsz = self.config.get("imgsz_max", 800)

        # Model
        num_classes = 11  # 10 VisDrone + background
        if use_fpn:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                num_classes=num_classes,
                trainable_backbone_layers=3,
            )
        else:
            backbone = self._build_backbone_no_fpn()
            from torchvision.models.detection import FasterRCNN
            from torchvision.models.detection.rpn import AnchorGenerator

            anchor_gen = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),),
            )
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=["0"], output_size=7, sampling_ratio=2,
            )
            self.model = FasterRCNN(
                backbone,
                num_classes=num_classes,
                rpn_anchor_generator=anchor_gen,
                box_roi_pool=roi_pooler,
            )

        self.model.to(self.device)

        # Optimizer & scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = SGD(
            params,
            lr=self.config.get("lr", 0.005),
            momentum=self.config.get("momentum", 0.9),
            weight_decay=self.config.get("weight_decay", 0.0001),
        )
        milestones = self.config.get("lr_milestones", [18, 24])
        gamma = self.config.get("lr_gamma", 0.1)
        self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        # DataLoaders
        train_img_dir = data_dir / "images" / "train"
        train_ann_dir = data_dir / "annotations" / "train"
        val_img_dir = data_dir / "images" / "val"
        val_ann_dir = data_dir / "annotations" / "val"

        train_dataset = VisDroneDataset(
            train_img_dir, train_ann_dir,
            transforms=get_train_transforms(img_size=imgsz),
        )
        val_dataset = VisDroneDataset(
            val_img_dir, val_ann_dir,
            transforms=get_val_transforms(img_size=imgsz),
        )

        batch_size = self.config.get("batch_size", 2)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=2, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=2, pin_memory=True,
        )

    def train_epoch(self) -> dict:
        """Run one training epoch. Return average loss dict."""
        self.model.train()
        total_losses = {}

        for images, targets in self.train_loader:
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = self.model(images, targets)
            losses = sum(loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()

        n = len(self.train_loader)
        avg_losses = {k: v / n for k, v in total_losses.items()}
        avg_losses["total_loss"] = sum(avg_losses.values())
        return avg_losses

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Run validation. Generate predictions, convert to COCO format, compute mAP.

        Returns:
            dict with mAP@0.5:0.95, mAP@0.5, AP_small, AP_medium, AP_large, AP_per_class
        """
        self.model.eval()
        from src.datasets import visdrone_to_coco_json
        from src.metrics import compute_map
        from src.env import get_paths as _gp

        data_dir = self.paths["data"]

        # Build COCO ground truth
        coco_gt = visdrone_to_coco_json(data_dir, split="val")
        # Build image_id -> (orig_w, orig_h) mapping for rescaling
        img_id_to_size = {}
        for img_info in coco_gt["images"]:
            img_id_to_size[img_info["id"]] = (img_info["width"], img_info["height"])

        predictions = []
        for images, targets in self.val_loader:
            image_ids = [t["image_id"].item() for t in targets]
            images = [img.to(self.device) for img in images]

            outputs = self.model(images)

            for img_id, output in zip(image_ids, outputs):
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    predictions.append({
                        "image_id": img_id,
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                        "category_id": int(label),
                    })

        metrics = compute_map(predictions, coco_gt)
        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save current state to disk (uses BaseTrainer generic save)."""
        super().save_checkpoint(is_best=is_best)

    def load_checkpoint(self, checkpoint_path):
        """Load state from disk (uses BaseTrainer generic load)."""
        super().load_checkpoint(checkpoint_path)


def train_frcnn(config: dict, paths: dict):
    """
    Convenience function: instantiate FRCNNTrainer and run train().

    Called from modal_app.py.
    """
    trainer = FRCNNTrainer(config, paths)
    trainer.train()
