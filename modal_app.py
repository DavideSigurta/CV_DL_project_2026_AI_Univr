"""
Modal.com cloud training entrypoint.

Usage:
    modal run modal_app.py::train_experiment --experiment-name e1a [--debug] [--epochs N]
    modal run modal_app.py::revalidate --experiment-name e1a
    modal run modal_app.py::download_results --experiment-name e1a
    modal run modal_app.py::list_outputs
"""

import modal

app = modal.App("visdrone-detection")

data_volume    = modal.Volume.from_name("visdrone-data",    create_if_missing=True)
outputs_volume = modal.Volume.from_name("visdrone-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", "/root/project", copy=True)
    .env({"PYTHONPATH": "/root/project"})
)

_GPU_FUNCTION_KWARGS = dict(
    image=image,
    gpu="A100",
    timeout=3600 * 6,
    volumes={"/data": data_volume, "/outputs": outputs_volume},
    secrets=[modal.Secret.from_dict({"VISDRONE_ENV": "modal"})],
)

CONFIG_FILES = {
    "e1a": "e1a_yolo_640.yaml",
    "e1b": "e1b_yolo_1280.yaml",
    "e2a": "e2a_frcnn_nofpn.yaml",
    "e2b": "e2b_frcnn_fpn.yaml",
    "e3":  "e3_sahi.yaml",
}


def _load_config(experiment_name: str, paths: dict) -> dict:
    import yaml
    key = experiment_name.replace("_debug", "")
    fname = CONFIG_FILES.get(key)
    if fname is None:
        raise ValueError(f"Unknown experiment: {experiment_name}. Choose from {list(CONFIG_FILES)}")
    with open(paths["configs"] / fname) as f:
        return yaml.safe_load(f)


@app.function(**_GPU_FUNCTION_KWARGS)
def train_experiment(experiment_name: str, debug: bool = False, epochs: int = None):
    """Train one experiment on Modal A100."""
    from src.env import get_paths
    from src.trainers.yolo_trainer import train_yolo
    from src.trainers.frcnn_trainer import train_frcnn

    paths = get_paths()
    config = _load_config(experiment_name, paths)

    if debug:
        config["epochs"] = epochs or 2
        config["use_debug_subset"] = True
        config["experiment"] = f"{experiment_name}_debug"
    elif epochs:
        config["epochs"] = epochs

    if experiment_name in ("e1a", "e1b"):
        train_yolo(config, paths)
    elif experiment_name in ("e2a", "e2b"):
        train_frcnn(config, paths)
    else:
        raise ValueError(f"Experiment {experiment_name!r} has no trainer (e3 runs locally).")


@app.function(**_GPU_FUNCTION_KWARGS)
def revalidate(experiment_name: str):
    """Re-run validation on an existing checkpoint using current code."""
    import json
    from src.env import get_paths
    from src.trainers.yolo_trainer import YOLOTrainer

    paths = get_paths()
    config = _load_config(experiment_name, paths)

    trainer = YOLOTrainer(config, paths)
    ckpt_path = trainer.output_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    print(f"Loading {ckpt_path}...")
    trainer.load_checkpoint(ckpt_path)

    print("Running validation...")
    val_metrics = trainer.validate()

    out = trainer.output_dir / "test_metrics.json"
    with open(out, "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\n=== {experiment_name} revalidated ===")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.4f}' for x in v[:5])}...]")
    print(f"Saved → {out}")


@app.local_entrypoint()
def download_results(experiment_name: str):
    """Download experiment results from Modal volume to local disk."""
    import subprocess
    import shutil
    from pathlib import Path

    local = Path(f"./results/runs/{experiment_name}")
    if local.exists():
        shutil.rmtree(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "visdrone-outputs", f"/runs/{experiment_name}", str(local)],
        check=True,
    )
    print(f"[OK] → {local}")


@app.local_entrypoint()
def list_outputs():
    """List all experiment outputs on the volume."""
    for entry in outputs_volume.listdir("/runs/"):
        print(entry)