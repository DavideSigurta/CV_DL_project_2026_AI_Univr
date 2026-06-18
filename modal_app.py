"""
Modal.com cloud training entrypoint.

Usage:
    modal run modal_app.py::train_experiment --experiment-name e1a [--debug] [--epochs N]
    modal run modal_app.py::revalidate --experiment-name e1a
    modal run modal_app.py::download_results --experiment-name e1a
    modal run modal_app.py::list_outputs
    modal run modal_app.py::setup_data
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

    # Persist results
    out = trainer.output_dir / "test_metrics.json"
    with open(out, "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\n=== {experiment_name} revalidated ===")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            vals = ", ".join(f"{x:.4f}" for x in v[:5])
            print(f"  {k}: [{vals}...]")
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


@app.function(**_GPU_FUNCTION_KWARGS)
def setup_data():
    """
    Initialize annotations/ on Modal volume from annotations_raw/.

    Reads raw VisDrone annotations (comma-separated, 8 columns) from
    annotations_raw/{train,val}, converts to YOLO+ format (space-separated,
    7 columns: cls cx cy w h occ trunc), and writes to annotations/{train,val}.

    Coordinates are clamped to [0,1]. Score==0 regions are skipped.
    Zero-area bboxes are skipped.

    Run ONCE after uploading images + annotations_raw to the volume:
        modal volume put visdrone-data data/VisDrone2019-DET/images /VisDrone2019-DET/images
        modal volume put visdrone-data data/VisDrone2019-DET/annotations_raw /VisDrone2019-DET/annotations_raw
        modal run modal_app.py::setup_data
    """
    from pathlib import Path
    from PIL import Image

    data_root = Path("/data/VisDrone2019-DET")
    SPLITS = ["train", "val"]

    for split in SPLITS:
        raw_dir = data_root / "annotations_raw" / split
        ann_dir = data_root / "annotations" / split
        img_dir = data_root / "images" / split

        if not raw_dir.exists():
            print(f"[ERROR] annotations_raw/{split} not found at {raw_dir}")
            print("Upload data first:")
            print(f"  modal volume put visdrone-data data/VisDrone2019-DET/images /VisDrone2019-DET/images")
            print(f"  modal volume put visdrone-data data/VisDrone2019-DET/annotations_raw /VisDrone2019-DET/annotations_raw")
            return

        if not img_dir.exists():
            print(f"[ERROR] images/{split} not found at {img_dir}")
            return

        ann_dir.mkdir(parents=True, exist_ok=True)
        raw_files = sorted(raw_dir.glob("*.txt"))
        print(f"[{split}] {len(raw_files)} raw annotation files found")

        total_boxes = 0
        oob_clamped = 0
        skipped_score0 = 0
        skipped_zero_area = 0

        for txt in raw_files:
            img_path = img_dir / (txt.stem + ".jpg")
            if not img_path.exists():
                img_path = img_dir / (txt.stem + ".JPG")
            if not img_path.exists():
                continue

            img = Image.open(img_path)
            w_img, h_img = img.size

            out_lines = []
            with open(txt) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 8:
                        continue

                    x1 = int(parts[0])
                    y1 = int(parts[1])
                    bw = int(parts[2])
                    bh = int(parts[3])
                    score   = int(parts[4])
                    cls_id  = int(parts[5])
                    trunc   = int(parts[6])
                    occ     = int(parts[7])

                    # Skip ignored regions
                    if score == 0:
                        skipped_score0 += 1
                        continue

                    # Skip zero-area boxes
                    if bw <= 0 or bh <= 0:
                        skipped_zero_area += 1
                        continue

                    # Normalize to [0,1]
                    cx = (x1 + bw / 2.0) / w_img
                    cy = (y1 + bh / 2.0) / h_img
                    nw = bw / w_img
                    nh = bh / h_img

                    # Track OOB before clamping
                    if cx > 1.0 or cy > 1.0 or nw > 1.0 or nh > 1.0:
                        oob_clamped += 1

                    # Clamp to [0,1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))

                    # Skip collapsed bbox after clamp
                    if nw < 1e-6 or nh < 1e-6:
                        continue

                    # 0-indexed class (VisDrone is 1-indexed)
                    yolo_cls = cls_id - 1

                    # YOLO+ format: cls cx cy w h occ trunc
                    out_lines.append(
                        f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {occ} {trunc}\n"
                    )
                    total_boxes += 1

            (ann_dir / txt.name).write_text("".join(out_lines))

        print(f"[{split}] Done: {total_boxes} boxes, {oob_clamped} clamped, "
              f"{skipped_score0} score==0 skipped, {skipped_zero_area} zero-area skipped")

        # Verify: no OOB values in generated annotations (sample 100 files)
        oob_violations = 0
        check_files = sorted(ann_dir.glob("*.txt"))[:100]
        for cf in check_files:
            for line in cf.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                vals = [float(p) for p in parts[1:5]]
                if any(v < 0 or v > 1.0001 for v in vals):
                    oob_violations += 1
        print(f"[{split}] OOB check on {len(check_files)} files: {oob_violations} violations (must be 0)")
        if oob_violations > 0:
            print(f"[WARN] OOB violations found! Bug in conversion.")

        # Commit changes to volume
        data_volume.commit()

    print("[OK] setup_data() completed.")