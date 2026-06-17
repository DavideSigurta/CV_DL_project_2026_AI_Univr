"""
Abstract base trainer — train loop, logging, checkpointing, early stopping.

All experiment trainers (YOLO, Faster R-CNN) inherit from this class.
"""

import torch
import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseTrainer(ABC):
    """
    Abstract base class for all detection trainers.

    Provides:
    - Training loop with epoch iteration
    - Validation loop
    - Checkpoint save/load/resume (via src.checkpointing)
    - Metric logging (metrics.jsonl)
    - Early stopping
    - Seed management
    """

    def __init__(self, config: dict, paths: dict):
        self.config = config
        self.paths = paths
        self.experiment = config["experiment"]
        self.output_dir = paths["output"] / self.experiment
        self.start_epoch = 1
        self.current_epoch = 0
        self.best_map = 0.0
        self.patience_counter = 0
        self.early_stop = False

        # Resolve device
        env_device = config.get("device", "auto")
        if env_device == "auto" or env_device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(env_device)

        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)

        # Save config copy
        config_path = self.output_dir / "config.yaml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        # Seed
        seed = config.get("seed", 42)
        self._seed_everything(seed)

    @abstractmethod
    def setup_model(self):
        """Initialize model, optimizer, scheduler."""
        pass

    @abstractmethod
    def train_epoch(self) -> dict:
        """Run one training epoch. Return loss dict."""
        pass

    @abstractmethod
    def validate(self) -> dict:
        """Run validation. Return metrics dict."""
        pass

    def save_checkpoint(self, is_best: bool = False):
        """Save current state to disk."""
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") and self.scheduler else None,
            "best_map": self.best_map,
            "config": self.config,
            "seed": self.config.get("seed", 42),
        }
        from src.checkpointing import save_checkpoint as _save
        _save(state, is_best, self.output_dir)

    def load_checkpoint(self, checkpoint_path: Path):
        """Load state from disk."""
        from src.checkpointing import load_checkpoint as _load
        state = _load(checkpoint_path, self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if hasattr(self, "optimizer") and self.optimizer and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if hasattr(self, "scheduler") and self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.start_epoch = state["epoch"] + 1
        self.best_map = state.get("best_map", 0.0)
        return state

    @staticmethod
    def _seed_everything(seed: int = 42):
        """Fix all sources of randomness."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _log_metrics(self, metrics: dict):
        """Append metrics dict to metrics.jsonl."""
        from src.checkpointing import append_metric
        append_metric(self.output_dir, metrics)
        # Also print to console
        desc = f"Epoch {self.current_epoch}"
        for k, v in metrics.items():
            if isinstance(v, float):
                desc += f" | {k}: {v:.4f}"
            elif isinstance(v, list):
                desc += f" | {k}: {[f'{x:.3f}' for x in v[:3]]}..."
            else:
                desc += f" | {k}: {v}"
        print(desc)

    def _check_early_stopping(self, current_map: float, patience: int) -> bool:
        """Check early stopping. Returns True if training should stop."""
        if current_map > self.best_map:
            self.best_map = current_map
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        if self.patience_counter >= patience:
            print(f"[Early stopping] No improvement for {patience} epochs. Best mAP@0.5: {self.best_map:.4f}")
            return True
        return False

    def train(self):
        """
        Full training loop:
        1. Setup model
        2. Resume from checkpoint if exists
        3. For each epoch: train_epoch() -> validate() -> save_checkpoint()
        4. Early stopping based on val mAP@0.5
        """
        self.setup_model()
        self.model.to(self.device)

        patience = self.config.get("patience", 10)

        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            self.current_epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            combined = {}
            combined["epoch"] = epoch
            for k, v in train_metrics.items():
                combined[f"train_{k}"] = v
            for k, v in val_metrics.items():
                combined[f"val_{k}"] = v

            self._log_metrics(combined)

            # Checkpoint
            val_map = val_metrics.get("mAP@0.5", val_metrics.get("map", 0.0))
            is_best = val_map > self.best_map
            self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self._check_early_stopping(val_map, patience):
                self.early_stop = True
                break

        print(f"\nTraining complete. Best mAP@0.5: {self.best_map:.4f}")
