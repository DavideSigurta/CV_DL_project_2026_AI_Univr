"""
Faster R-CNN trainer — torchvision implementation for E2a and E2b experiments.

E2a: ResNet50 backbone, NO FPN (single-scale)
    - Manual backbone: ResNet50 without avgpool+fc, single-scale RoI align
    - Small anchors (8-128px) adapted for VisDrone small objects
E2b: ResNet50 backbone + FPN (Feature Pyramid Network)
    - Standard fasterrcnn_resnet50_fpn from torchvision
    - Default multi-scale anchors (FPN handles scale natively)

Inherits from BaseTrainer. Uses torchvision.models.detection.

Reference: master plan Sec 6.4-6.5. Anchor sizes for E2a adapted to VisDrone
per master plan recommendation: (8, 16, 32, 64, 128) instead of COCO default
(32, 64, 128, 256, 512). Without FPN, only one feature map (C5 stride 32) —
small anchors critical for small object recall.
"""

import json
import shutil
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
        # weights="IMAGENET1K_V1" instead of deprecated pretrained=True
        backbone = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        # Remove avgpool and fc layers — keep conv layers up to layer4
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        return backbone

    def setup_model(self):
        """Initialize model, optimizer, scheduler, and data loaders."""
        from src.datasets import VisDroneDataset
        from src.augmentations import get_frcnn_train_transforms, get_frcnn_val_transforms

        paths = self.paths
        data_dir = paths["data"]
        use_fpn = self.config.get("fpn", True)
        imgsz = self.config.get("imgsz_max", 800)

        # Model
        num_classes = 11  # 10 VisDrone + background
        if use_fpn:
            # torchvision 0.27+: pretrained=True con num_classes≠91 dà errore.
            # weights_backbone carica solo backbone ResNet50 (ImageNet), num_classes libero.
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights_backbone="ResNet50_Weights.IMAGENET1K_V1",
                num_classes=num_classes,
                trainable_backbone_layers=0,  # frozen backbone (same as E2a) — only FPN differs
            )
        else:
            backbone = self._build_backbone_no_fpn()
            from torchvision.models.detection import FasterRCNN
            from torchvision.models.detection.rpn import AnchorGenerator

            # Small anchors per master plan: VisDrone objects are tiny,
            # default COCO anchors (32-512) too coarse for single-scale C5 (stride 32).
            anchor_gen = AnchorGenerator(
                sizes=((8, 16, 32, 64, 128),),
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

        # E2b backbone already frozen via trainable_backbone_layers=0 in constructor.
        # E2a (custom backbone, no FPN constructor) needs manual freeze.
        # Clean ablation: backbone frozen in BOTH. Only difference = FPN.
        if not use_fpn:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("[FRCNN] Backbone frozen — training only RPN + detection heads.")

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

        # DataLoaders — respect use_debug_subset for fast pipeline validation
        use_debug = self.config.get("use_debug_subset", False)
        if use_debug:
            subset_dir = self.paths["subsets"] / "debug_500"
            train_img_dir = subset_dir / "images"
            train_ann_dir = subset_dir / "annotations"
            val_img_dir = subset_dir / "images"
            val_ann_dir = subset_dir / "annotations"
        else:
            train_img_dir = data_dir / "images" / "train"
            train_ann_dir = data_dir / "annotations" / "train"
            val_img_dir = data_dir / "images" / "val"
            val_ann_dir = data_dir / "annotations" / "val"

        train_dataset = VisDroneDataset(
            train_img_dir, train_ann_dir,
            transforms=get_frcnn_train_transforms(img_size=imgsz),
        )
        val_dataset = VisDroneDataset(
            val_img_dir, val_ann_dir,
            transforms=get_frcnn_val_transforms(img_size=imgsz),
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
        """Run one training epoch with AMP. Return average loss dict."""
        self.model.train()
        total_losses = {}
        # AMP: automatic mixed precision (fp16) for ~1.5-2× speedup on CUDA
        amp_enabled = self.device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=amp_enabled)
        n_batches = len(self.train_loader)
        log_interval = max(1, n_batches // 4)  # log at 25%, 50%, 75%

        for batch_idx, (images, targets) in enumerate(self.train_loader, 1):
            images = [img.to(self.device) for img in images]
            targets = [
                {k: v.to(self.device) for k, v in t.items()}
                for t in targets
            ]

            with torch.amp.autocast(device_type=self.device.type, enabled=amp_enabled):
                loss_dict = self.model(images, targets)
                losses = sum(loss_dict.values())

            self.optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(self.optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()

            if batch_idx % log_interval == 0:
                curr_loss = sum(loss_dict[k].item() for k in loss_dict)
                print(f"  [{batch_idx}/{n_batches}] loss={curr_loss:.4f}")

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
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

        data_dir = self.paths["data"]
        imgsz_max = self.config.get("imgsz_max", 800)

        # Build COCO ground truth — respect use_debug_subset (debug_500)
        use_debug = self.config.get("use_debug_subset", False)
        if use_debug:
            subset_dir = self.paths["subsets"] / "debug_500"
            coco_gt = visdrone_to_coco_json(
                img_dir=subset_dir / "images",
                ann_dir=subset_dir / "annotations",
            )
            print(f"[VALIDATE] Using debug_500 GT ({len(coco_gt['images'])} images)")
        else:
            coco_gt = visdrone_to_coco_json(data_dir, split="val")

        predictions = []
        for images, targets in self.val_loader:
            image_ids = [t["image_id"].item() for t in targets]
            # orig_size from dataset: [orig_h, orig_w] in original px
            orig_sizes = [t["orig_size"].tolist() for t in targets]
            images = [img.to(self.device) for img in images]

            outputs = self.model(images)

            for img_id, output, (orig_h, orig_w) in zip(image_ids, outputs, orig_sizes):
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # Model predictions are in RESIZED image space (longest side = imgsz_max).
                # Scale back to original image coordinates so they align with COCO GT.
                scale = imgsz_max / max(orig_w, orig_h)
                inv_scale = 1.0 / scale if scale > 0 else 1.0

                for box, score, label in zip(boxes, scores, labels):
                    # image_id: VisDroneDataset uses 0-based idx, visdrone_to_coco_json uses
                    # 1-based (enumerate start=1). Add +1 to align.
                    # category_id: model outputs 1-10 (bg=0, VisDrone=1..10).
                    # visdrone_to_coco_json returns 1-indexed GT (1..10) — no shift needed.
                    x1, y1, x2, y2 = box
                    x1 *= inv_scale
                    y1 *= inv_scale
                    x2 *= inv_scale
                    y2 *= inv_scale
                    w = x2 - x1
                    h = y2 - y1
                    predictions.append({
                        "image_id": img_id + 1,
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                        "category_id": int(label),
                    })

        # Debug: verify image_id alignment
        pred_img_ids = set(p["image_id"] for p in predictions)
        gt_img_ids = set(im["id"] for im in coco_gt["images"])
        overlap = pred_img_ids & gt_img_ids
        print(f"[DEBUG] {len(predictions)} predictions, {len(pred_img_ids)} unique image_ids")
        print(f"[DEBUG] GT has {len(gt_img_ids)} images")
        print(f"[DEBUG] Overlap pred ∩ GT: {len(overlap)}/{len(pred_img_ids)}")
        if len(overlap) < len(pred_img_ids):
            missing = pred_img_ids - gt_img_ids
            print(f"[DEBUG] Sample missing from GT: {sorted(missing)[:5]}")

        metrics = compute_map(predictions, coco_gt)
        return metrics

    def _has_valid_run(self) -> bool:
        """Check if completed run exists via test_metrics.json with non-zero mAP@0.5."""
        metrics_path = self.output_dir / "test_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                return m.get("mAP@0.5", 0.0) > 0.001
            except (json.JSONDecodeError, KeyError):
                pass
        return False

    def train(self):
        """
        Full training loop. Overrides BaseTrainer.train() for metric persistence.

        Skips if force_retrain=False and valid test_metrics.json exists.
        Saves test_metrics.json, copies checkpoints, moves figures to figures/.
        """
        force = self.config.get("force_retrain", False)
        if not force and self._has_valid_run():
            print(f"[SKIP] {self.experiment} already completed. "
                  f"Set force_retrain=True to retrain.")
            with open(self.output_dir / "test_metrics.json") as f:
                val_metrics = json.load(f)
            return val_metrics

        print(f"[TRAIN] Setting up model (downloading pretrained weights if needed)...")
        self.setup_model()
        self.model.to(self.device)
        print(f"[TRAIN] Model ready on {self.device}. Starting training loop.")

        patience = self.config.get("patience", 7)
        from src.checkpointing import append_metric

        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            self.current_epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*60}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            combined = {"epoch": epoch}
            for k, v in train_metrics.items():
                combined[f"train_{k}"] = v
            for k, v in val_metrics.items():
                combined[f"val_{k}"] = v

            # Log to console
            desc = f"Epoch {epoch}"
            for k, v in combined.items():
                if isinstance(v, float):
                    desc += f" | {k}: {v:.4f}"
                elif isinstance(v, list):
                    desc += f" | {k}: {[f'{x:.3f}' for x in v[:3]]}..."
                else:
                    desc += f" | {k}: {v}"
            print(desc)

            append_metric(self.output_dir, combined)

            val_map = val_metrics.get("mAP@0.5", 0.0)
            is_best = val_map > self.best_map
            self.save_checkpoint(is_best=is_best)

            if self._check_early_stopping(val_map, patience):
                self.early_stop = True
                break

            if hasattr(self, "scheduler") and self.scheduler:
                self.scheduler.step()

        print(f"\nTraining complete. Best mAP@0.5: {self.best_map:.4f}")

        # Move figures to figures/
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("*.png", "*.jpg"):
            for f in self.output_dir.glob(ext):
                if f.parent == figures_dir:
                    continue
                shutil.move(str(f), str(figures_dir / f.name))

        with open(self.output_dir / "test_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        print(f"\n{'='*60}")
        print(f"{self.experiment.upper()} complete. Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'='*60}")

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


def build_frcnn_model(config: dict) -> torch.nn.Module:
    """
    Build FRCNN model matching config without optimizer/dataloaders.

    Args:
        config: experiment config dict with keys 'fpn', 'imgsz_max', etc.

    Returns:
        torch.nn.Module (in eval mode, on CPU)
    """
    use_fpn = config.get("fpn", True)
    num_classes = config.get("num_classes", 11)

    if use_fpn:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights_backbone="ResNet50_Weights.IMAGENET1K_V1",
            num_classes=num_classes,
            trainable_backbone_layers=0,
        )
    else:
        backbone = torchvision.models.resnet50(pretrained=False)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        anchor_gen = AnchorGenerator(
            sizes=((8, 16, 32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2,
        )
        model = FasterRCNN(
            backbone, num_classes=num_classes,
            rpn_anchor_generator=anchor_gen,
            box_roi_pool=roi_pooler,
        )
        for param in model.backbone.parameters():
            param.requires_grad = False
    model.eval()
    return model
