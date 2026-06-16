"""
Faster R-CNN trainer — torchvision implementation for E2a and E2b experiments.

E2a: ResNet50 backbone, NO FPN (single-scale)
E2b: ResNet50 backbone + FPN (Feature Pyramid Network)

Inherits from BaseTrainer. Uses torchvision.models.detection.
"""

from .base_trainer import BaseTrainer


class FRCNNTrainer(BaseTrainer):
    """
    Trainer for Faster R-CNN experiments.

    Config determines:
    - E2a: fpn=False, custom backbone without FPN
    - E2b: fpn=True, fasterrcnn_resnet50_fpn from torchvision
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


def train_frcnn(config: dict, paths: dict):
    """
    Convenience function: instantiate FRCNNTrainer and run train().

    Called from modal_app.py.
    """
    trainer = FRCNNTrainer(config, paths)
    trainer.train()
