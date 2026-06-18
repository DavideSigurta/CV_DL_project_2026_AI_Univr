"""
Environment resolution — single source of truth for device and path configuration.

Every other file imports from here and never calls torch.device(...) or constructs paths directly.
"""

import os
import torch
from pathlib import Path


def get_device() -> torch.device:
    """Return the best available device. Priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_paths() -> dict[str, Path]:
    """
    Resolve all project paths from the VISDRONE_ENV environment variable.

    Supported values:
        local  (default) — MacBook M1, paths relative to repo root
        modal             — Modal.com A100, volumes mounted at /data and /outputs

    Usage:
        export VISDRONE_ENV=modal   # before launching Modal function
        paths = get_paths()
    """
    env = os.environ.get("VISDRONE_ENV", "local")

    if env == "local":
        root = Path(__file__).resolve().parent.parent
        return {
            "data":    root / "data" / "VisDrone2019-DET",
            "subsets": root / "data" / "subsets",
            "output":  root / "results" / "runs",
            "configs": root / "configs",
        }
    elif env == "modal":
        return {
            "data":    Path("/data/VisDrone2019-DET"),
            "subsets": Path("/data/subsets"),
            "output":  Path("/outputs/runs"),
            "configs": Path("/root/project/configs"),
        }
    else:
        raise ValueError(f"Unknown VISDRONE_ENV: {env!r}. Choose 'local' or 'modal'.")


def get_env_info() -> dict:
    """Return a dict suitable for logging alongside each run config."""
    device = get_device()
    return {
        "env":    os.environ.get("VISDRONE_ENV", "local"),
        "device": str(device),
        "cuda_available":  torch.cuda.is_available(),
        "mps_available":   torch.backends.mps.is_available(),
        "torch_version":   torch.__version__,
    }


def set_matplotlib_backend() -> None:
    """Set Agg backend only in headless environments (Modal).

    In notebooks (local env) leaves default backend alone so
    Jupyter inline display works. Must be called *before*
    ``import matplotlib.pyplot``.
    """
    env = os.environ.get("VISDRONE_ENV", "local")
    if env == "modal":
        import matplotlib
        matplotlib.use("Agg")


def seed_everything(seed: int = 42) -> None:
    """Fix all sources of randomness for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False