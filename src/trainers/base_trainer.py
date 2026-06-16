"""
Abstract base trainer — train loop, logging, checkpointing, early stopping.

All experiment trainers (YOLO, Faster R-CNN) inherit from this class.
"""

import torch
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
        self.output_dir = paths["output"] / config["experiment"]
        self.device = torch.device("cpu")  # resolved in setup()
        # TODO: setup directories, seed, device

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

    @abstractmethod
    def save_checkpoint(self, is_best: bool = False):
        """Save current state to disk."""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path):
        """Load state from disk."""
        pass

    def seed_everything(self, seed: int = 42):
        """Fix all sources of randomness."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        """
        Full training loop:
        1. Resume from checkpoint if exists
        2. For each epoch: train_epoch() -> validate() -> save_checkpoint()
        3. Early stopping based on val mAP@0.5
        """
        raise NotImplementedError
