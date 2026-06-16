"""
YOLOv5s trainer — wraps Ultralytics API for E1a and E1b experiments.

Inherits from BaseTrainer. Integrates with get_paths() and seed_everything().
"""

from .base_trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):
    """
    Trainer for YOLOv5s experiments.

    E1a: imgsz=640
    E1b: imgsz=1280

    Uses Ultralytics YOLOv5 under the hood.
    """

    def setup_model(self):
        raise NotImplementedError

    def train_epoch(self) -> dict:
        raise NotImplementedError

    def validate(self) -> dict:
        raise NotImplementedError

    def save_checkpoint(self, is_best: bool = False):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError


def train_yolo(config: dict, paths: dict):
    """
    Convenience function: instantiate YOLOTrainer and run train().

    Called from modal_app.py.
    """
    trainer = YOLOTrainer(config, paths)
    trainer.train()
