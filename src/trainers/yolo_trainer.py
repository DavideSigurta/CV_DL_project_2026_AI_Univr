"""
YOLOv5s trainer — wraps Ultralytics API for E1a and E1b experiments.

Inherits from BaseTrainer. Integrates with get_paths() and seed_everything().

Design (thin wrapper):
- Ultralytics handles internal training loop, data loading, augmentations
- We configure via our YAML config, generate temp data.yaml for dataset paths
- After training, we parse Ultralytics results.csv for per-epoch metrics
- We run model.val() for final COCO-format evaluation
- Checkpoints are copied from Ultralytics output to our structure

E1a: imgsz=640
E1b: imgsz=1280
"""

import shutil
import csv
import json
import yaml
from pathlib import Path

import torch

from .base_trainer import BaseTrainer
from src.datasets import VISDRONE_CLASSES
from src.checkpointing import append_metric


class YOLOTrainer(BaseTrainer):
    """
    Thin wrapper around Ultralytics YOLO for E1a and E1b experiments.

    Overrides train() to use Ultralytics' full training loop.
    """

    def __init__(self, config: dict, paths: dict):
        super().__init__(config, paths)
        self.model = None
        self._ultralytics_output_dir = None

    def _create_data_yaml(self) -> Path:
        """Create data.yaml for Ultralytics pointing to VisDrone paths.

        Always regenerates (no caching) to prevent stale paths from debug runs.
        Uses full dataset unless self.config has 'use_debug_subset'=True.
        Ensures a sibling ``labels/`` dir exists for Ultralytics (it looks for
        labels there, not in ``annotations/``).
        """
        import shutil

        data_dir = self.paths["data"]
        yaml_path = self.output_dir / "data.yaml"

        # Remove stale cached data.yaml (e.g. from a debug run)
        if yaml_path.exists():
            yaml_path.unlink()

        use_debug = self.config.get("use_debug_subset", False)
        if use_debug:
            subset_dir = self.paths["subsets"] / "debug_500"
            train_dir = subset_dir / "images"
            if not train_dir.exists():
                raise FileNotFoundError(
                    f"Debug subset not found at {train_dir}. "
                    "Run data/scripts/create_debug_subset.py first."
                )
            val_dir = train_dir

            # Ultralytics expects labels in a sibling ``labels/`` directory.
            # If only ``annotations/`` exists, populate ``labels/`` from it.
            labels_dir = subset_dir / "labels"
            ann_dir = subset_dir / "annotations"
            if not labels_dir.exists() and ann_dir.exists():
                labels_dir.mkdir(exist_ok=True)
                for txt in ann_dir.glob("*.txt"):
                    shutil.copy2(txt, labels_dir / txt.name)

            data_yaml = {
                "path": str(subset_dir),
                "train": "images",
                "val": "images",
                "nc": len(VISDRONE_CLASSES),
                "names": VISDRONE_CLASSES,
            }
        else:
            train_dir = data_dir / "images" / "train"
            val_dir = data_dir / "images" / "val"

            # Same logic: populate ``labels/`` from ``annotations/`` if missing.
            for split in ("train", "val"):
                labels_dir = data_dir / "labels" / split
                ann_dir = data_dir / "annotations" / split
                if not labels_dir.exists() and ann_dir.exists():
                    labels_dir.mkdir(parents=True, exist_ok=True)
                    for txt in ann_dir.glob("*.txt"):
                        shutil.copy2(txt, labels_dir / txt.name)

            data_yaml = {
                "path": str(data_dir),
                "train": "images/train",
                "val": "images/val",
                "nc": len(VISDRONE_CLASSES),
                "names": VISDRONE_CLASSES,
            }
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        return yaml_path

    def setup_model(self):
        """Initialize Ultralytics YOLO model."""
        from ultralytics import YOLO

        # ---- yolo weight path ----
        # YOLO("yolov5su.pt") resolves relative to CWD (notebooks/).
        # Use project-root/models/ path so weight file is organised.
        model_path = Path(__file__).resolve().parent.parent.parent / "models" / "yolov5su.pt"
        self.model = YOLO(str(model_path))

        # ---- MPS autocast workaround ----
        # PyTorch <2.3 raises RuntimeError on torch.amp.autocast('mps', ...).
        # The validator imports autocast at module level, so patching
        # ultralytics.utils.torch_utils.autocast is NOT enough — the validator
        # already holds its own reference.  We patch torch.amp.autocast.__init__
        # directly so every call site is covered.
        if self.device.type == "mps":
            _orig_init = torch.amp.autocast.__init__

            def _mps_autocast_init(self_, device_type, dtype=None, enabled=True, cache_enabled=True):
                if device_type == "mps":
                    device_type = "cpu"
                return _orig_init(self_, device_type, dtype, enabled, cache_enabled)

            torch.amp.autocast.__init__ = _mps_autocast_init

    def train_epoch(self) -> dict:
        """Not used — Ultralytics handles internal training loop."""
        return {"loss": 0.0}

    def validate(self) -> dict:
        """
        Run validation using Ultralytics' internal evaluator.

        Returns dict matching COCO convention for downstream notebook cells:
            mAP@0.5:0.95, mAP@0.5, mAP@0.75,
            AP_small, AP_medium, AP_large, AP_per_class (list of 10)
        """
        import json as _json

        data_yaml = self._create_data_yaml()

        # Run Ultralytics validation — gives us standard metrics reliably.
        results = self.model.val(
            data=str(data_yaml),
            imgsz=self.config["imgsz"],
            batch=self.config["batch_size"],
            device=self.device,
            split="val",
            verbose=False,
            save_json=True,
            save_hybrid=False,
            project=str(self.output_dir.parent),
            name=self.experiment,
            exist_ok=True,
        )

        # ── Standard + per-class metrics from Ultralytics ──
        ap_per_class = []
        if hasattr(results, "box") and hasattr(results.box, "maps"):
            ap_per_class = [float(v) for v in results.box.maps]
        while len(ap_per_class) < len(VISDRONE_CLASSES):
            ap_per_class.append(0.0)

        metrics = {
            "mAP@0.5:0.95": float(results.box.map),
            "mAP@0.5": float(results.box.map50),
            "mAP@0.75": float(results.box.map75),
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AP_per_class": ap_per_class[: len(VISDRONE_CLASSES)],
        }

        # ── Per-scale AP via COCOeval using model predictions ──
        # We build predictions manually (bypass Ultralytics JSON format)
        # to avoid image_id format mismatches.
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            use_debug = self.config.get("use_debug_subset", False)
            if use_debug:
                subset_dir = self.paths["subsets"] / "debug_500"
                img_dir = subset_dir / "images"
                ann_dir = subset_dir / "annotations"
            else:
                data_dir = self.paths["data"]
                img_dir = data_dir / "images" / "val"
                ann_dir = data_dir / "annotations" / "val"

            from src.datasets import visdrone_to_coco_json
            coco_gt_dict = visdrone_to_coco_json(img_dir=img_dir, ann_dir=ann_dir)

            # Shift category IDs 0‑based → 1‑based for COCO
            for cat in coco_gt_dict["categories"]:
                cat["id"] = cat["id"] + 1
            for ann in coco_gt_dict["annotations"]:
                ann["category_id"] = ann["category_id"] + 1

            coco_gt = COCO()
            coco_gt.dataset = coco_gt_dict
            coco_gt.createIndex()

            # Run inference on val images, build COCO prediction list
            img_files = sorted(img_dir.glob("*.jpg"))
            preds = []
            for img_path in img_files:
                img_id = None
                for im in coco_gt_dict["images"]:
                    if im["file_name"] == img_path.name:
                        img_id = im["id"]
                        break
                if img_id is None:
                    continue

                results_img = self.model(img_path, verbose=False)
                for result in results_img:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(int)
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box
                        preds.append({
                            "image_id": img_id,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                            "category_id": int(label) + 1,  # 0‑based → 1‑based
                        })

            if preds:
                coco_dt = coco_gt.loadRes(preds)
                coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                s = coco_eval.stats  # [AP50:95, AP50, AP75, APs, APm, APl, ...]
                metrics["mAP@0.5:0.95"] = float(s[0])
                metrics["mAP@0.5"] = float(s[1])
                metrics["mAP@0.75"] = float(s[2])
                metrics["AP_small"] = float(s[3])
                metrics["AP_medium"] = float(s[4])
                metrics["AP_large"] = float(s[5])

                # Per-class AP from COCOeval precision array
                prec = coco_eval.eval["precision"]  # (T, R, K, A, M)
                num_cats = len(coco_gt.getCatIds())
                per_class = []
                for c in range(num_cats):
                    ap = prec[:, :, c, 0, -1]
                    ap = ap[ap > -1]
                    per_class.append(float(ap.mean()) if ap.size > 0 else 0.0)
                metrics["AP_per_class"] = per_class[: len(VISDRONE_CLASSES)]
        except Exception as e:
            print(f"[WARN] COCOeval per-scale AP failed: {e}")

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """
        Copy Ultralytics checkpoints to our output structure.

        Ultralytics saves weights in runs/detect/train*/weights/.
        We copy best.pt and last.pt to output_dir/checkpoints/.
        """
        if self._ultralytics_output_dir is None:
            return

        src_best = self._ultralytics_output_dir / "weights" / "best.pt"
        src_last = self._ultralytics_output_dir / "weights" / "last.pt"
        dst_dir = self.output_dir / "checkpoints"

        if src_best.exists():
            shutil.copy2(src_best, dst_dir / "best.pt")
        if src_last.exists():
            shutil.copy2(src_last, dst_dir / "last.pt")

    def load_checkpoint(self, checkpoint_path):
        """Load YOLO model from checkpoint."""
        from ultralytics import YOLO
        self.model = YOLO(str(checkpoint_path))

    def _parse_ultralytics_results_csv(self, results_csv: Path) -> list:
        """Parse Ultralytics results.csv for per-epoch metrics."""
        if not results_csv.exists():
            return []
        epochs = []
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append({
                    "epoch": int(row.get("epoch", 0)),
                    "train_box_loss": float(row.get("train/box_loss", 0)),
                    "train_cls_loss": float(row.get("train/cls_loss", 0)),
                    "train_dfl_loss": float(row.get("train/dfl_loss", 0)),
                    "val_mAP@0.5:0.95": float(row.get("metrics/mAP50-95(B)", 0)),
                    "val_mAP@0.5": float(row.get("metrics/mAP50(B)", 0)),
                })
        return epochs

    def _has_valid_run(self) -> bool:
        """Check if a completed run exists with valid metrics.

        Checks two sources:
        1. pycocotools ``test_metrics.json`` (may be zero for YOLO due to
           format mismatches — not a failure indicator).
        2. Ultralytics ``results.csv`` (always reflects actual validation).
        Returns True if either has non-zero mAP.
        """
        # Source 1 — pycocotools metrics
        metrics_path = self.output_dir / "test_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    m = json.load(f)
                if any(v != 0 for v in m.values() if isinstance(v, (int, float))):
                    return True
            except (json.JSONDecodeError, KeyError):
                pass

        # Source 2 — Ultralytics results.csv (fallback)
        results_csv = self.output_dir / "results.csv"
        if results_csv.exists():
            try:
                with open(results_csv) as f:
                    for line in f:
                        if "mAP50" in line or "metrics/mAP50" in line:
                            continue  # header
                        parts = line.strip().split(",")
                        # Ultralytics CSV columns include metrics/mAP50(B) etc.
                        for part in parts:
                            try:
                                val = float(part)
                                if val > 0.001:  # non-trivial mAP
                                    return True
                            except ValueError:
                                continue
            except (OSError, IndexError):
                pass

        return False

    def train(self):
        """
        Full training using Ultralytics YOLO.

        Overrides BaseTrainer.train() since Ultralytics has its own loop.
        Skips training if ``force_retrain`` is False and a valid run already
        exists (detected by non-zero metrics in test_metrics.json).
        """
        force = self.config.get("force_retrain", False)
        if not force and self._has_valid_run():
            print(f"[SKIP] {self.experiment} already completed. "
                  f"Set force_retrain=True to retrain.")
            with open(self.output_dir / "test_metrics.json") as f:
                val_metrics = json.load(f)
            return val_metrics

        self.setup_model()
        data_yaml = self._create_data_yaml()

        # Extract augment config from our YAML
        aug = self.config.get("augment", {})
        hsv_h = aug.get("hsv_h", 0.015)
        hsv_s = aug.get("hsv_s", 0.7)
        hsv_v = aug.get("hsv_v", 0.4)
        scale = aug.get("scale", 0.5)
        fliplr = aug.get("fliplr", 0.5)
        mosaic = aug.get("mosaic", 1.0)

        # Run Ultralytics training
        self.model.train(
            data=str(data_yaml),
            imgsz=self.config["imgsz"],
            epochs=self.config["epochs"],
            batch=self.config["batch_size"],
            optimizer=self.config.get("optimizer", "AdamW").lower(),
            lr0=self.config.get("lr0", 0.001),
            lrf=self.config.get("lrf", 0.01),
            weight_decay=self.config.get("weight_decay", 0.0005),
            warmup_epochs=self.config.get("warmup_epochs", 3),
            patience=self.config.get("patience", 10),
            seed=self.config.get("seed", 42),
            device=self.device,
            project=str(self.output_dir.parent),
            name=self.experiment,
            exist_ok=True,
            amp=False,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            scale=scale,
            fliplr=fliplr,
            mosaic=mosaic,
        )

        # Locate Ultralytics output directory
        weights_dir = self.output_dir.parent / self.experiment / "weights"
        if weights_dir.exists():
            self._ultralytics_output_dir = weights_dir.parent
        else:
            candidates = sorted(self.output_dir.parent.glob(f"{self.experiment}*/weights"))
            if candidates:
                self._ultralytics_output_dir = candidates[0].parent

        # Parse per-epoch metrics from results.csv
        results_csv = None
        if self._ultralytics_output_dir:
            candidate = self._ultralytics_output_dir / "results.csv"
            if candidate.exists():
                results_csv = candidate
        if results_csv is None:
            candidate = self.output_dir.parent / self.experiment / "results.csv"
            if candidate.exists():
                results_csv = candidate

        epoch_metrics = self._parse_ultralytics_results_csv(results_csv) if results_csv else []
        for em in epoch_metrics:
            append_metric(self.output_dir, em)

        # Save checkpoint (copy from Ultralytics)
        self.save_checkpoint(is_best=True)

        # Move Ultralytics plot files (*.png, *.jpg) from run root → figures/
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("*.png", "*.jpg"):
            for f in self.output_dir.glob(ext):
                if f.parent == figures_dir:
                    continue
                shutil.move(str(f), str(figures_dir / f.name))

        # Run final validation
        val_metrics = self.validate()
        with open(self.output_dir / "test_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        print(f"\n{'='*60}")
        print(f"{self.experiment.upper()} complete. Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'='*60}")


def train_yolo(config: dict, paths: dict):
    """
    Convenience function: instantiate YOLOTrainer and run train().

    Called from modal_app.py.
    """
    trainer = YOLOTrainer(config, paths)
    trainer.train()
