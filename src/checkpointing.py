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
from typing import Optional


def save_checkpoint(state: dict, is_best: bool, output_dir: Path):
    """
    Save training checkpoint.

    Args:
        state: dict with keys (epoch, model_state_dict, optimizer_state_dict,
               scheduler_state_dict, best_map, config, seed)
        is_best: if True, also save as best.pt
        output_dir: directory to save checkpoints
    """
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pt"
    torch.save(state, last_path)
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """
    Load checkpoint from disk.

    Returns:
        Checkpoint dict
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required_keys = ["epoch", "model_state_dict", "optimizer_state_dict", "best_map"]
    for key in required_keys:
        if key not in state:
            raise KeyError(f"Checkpoint missing required key: {key}")
    return state


def get_resume_state(output_dir: Path, device: torch.device) -> Optional[dict]:
    """
    Check if a checkpoint exists and return state for resume.

    Returns None if no checkpoint found.
    """
    last_path = output_dir / "checkpoints" / "last.pt"
    if not last_path.exists():
        return None
    return load_checkpoint(last_path, device)


def append_metric(output_dir: Path, metrics: dict):
    """
    Append one line of metrics to metrics.jsonl.
    """
    metrics_path = output_dir / "metrics.jsonl"
    with open(metrics_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
