"""
Modal.com cloud training entrypoint.

Usage:
    modal run modal_app.py::train_experiment --experiment-name e1a

Environment:
    VISDRONE_ENV=modal (injected via Modal secret)
    Persistent volumes: visdrone-data (/data), visdrone-outputs (/outputs)
"""

import modal

app = modal.App("visdrone-detection")

data_volume    = modal.Volume.from_name("visdrone-data",    create_if_missing=True)
outputs_volume = modal.Volume.from_name("visdrone-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 6,
    volumes={
        "/data":    data_volume,
        "/outputs": outputs_volume,
    },
    secrets=[modal.Secret.from_dict({"VISDRONE_ENV": "modal"})],
)
def train_experiment(experiment_name: str, config_override: dict = None):
    """
    Run a single training experiment on Modal A100.

    Args:
        experiment_name: one of 'e1a', 'e1b', 'e2a', 'e2b'
        config_override: optional dict to override YAML defaults at runtime
    """
    import yaml
    from src.env import get_paths
    from src.trainers.yolo_trainer import train_yolo
    from src.trainers.frcnn_trainer import train_frcnn

    paths = get_paths()
    config_path = paths["configs"] / f"{experiment_name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_override:
        config.update(config_override)

    if experiment_name in ("e1a", "e1b"):
        train_yolo(config, paths)
    elif experiment_name in ("e2a", "e2b"):
        train_frcnn(config, paths)
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
