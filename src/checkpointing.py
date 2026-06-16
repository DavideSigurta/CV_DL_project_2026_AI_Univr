"""
Checkpoint save/load logic, resume-from-checkpoint for all trainers.

Rules:
- Save best.pt when val_mAP@0.5 improves.
- Save last.pt unconditionally at end of each epoch.
- Save metrics.jsonl with one JSON line per epoch.
- All checkpoint files include: epoch, model_state_dict, optimizer_state_dict,
  scheduler_state_dict, best_map, config, seed.
"""

import torch
import json
from pathlib import Path
from typing import Optional, Any


def save_checkpoint(state: dict, is_best: bool, output_dir: Path):
    """
    Save training checkpoint.

    Args:
        state: dict with keys (epoch, model_state_dict, optimizer_state_dict,
               scheduler_state_dict, best_map, config, seed)
        is_best: if True, also save as best.pt
        output_dir: directory to save checkpoints
    """
    raise NotImplementedError


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """
    Load checkpoint from disk.

    Returns:
        Checkpoint dict
    """
    raise NotImplementedError


def get_resume_state(output_dir: Path, device: torch.device) -> Optional[dict]:
    """
    Check if a checkpoint exists and return state for resume.

    Returns None if no checkpoint found.
    """
    raise NotImplementedError


def append_metric(output_dir: Path, metrics: dict):
    """
    Append one line of metrics to metrics.jsonl.
    """
    raise NotImplementedError
