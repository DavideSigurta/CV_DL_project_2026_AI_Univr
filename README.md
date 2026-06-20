# VisDrone2019-DET — Small Object Detection in Aerial Imagery

Controlled comparative study for **Computer Vision & Deep Learning** (UniVR, A.Y. 2025–26, Prof. V. Murino, F. Dibitonto).

## Research question

> *Is performance bottleneck of small object detection on aerial imagery an architectural problem (detector design) or a pipeline problem (input resolution fed to the model)?*

## Dataset

[**VisDrone2019-DET**](https://github.com/VisDrone/VisDrone-Dataset) — drone-captured images, 10 object classes, 2.6M+ annotations. Download from [Kaggle](https://kaggle.com/datasets/kushagrapandya/visdrone-dataset) (~2.25 GB). Training split: 6,471 images, validation: 548, test-dev: 1,610.

Annotations stored in **YOLO+ format** (7 columns): standard YOLO (cls cx cy w h) + occlusion + truncation flags. Original VisDrone format preserved in `annotations_raw/`.

> 50%+ instances are **small** (<32² px per COCO convention) — the core challenge.

## Experiments

| Step | Experiment | Variable | vs Reference |
|------|-----------|----------|-------------|
| P | EDA | Dataset analysis | — |
| E1a | YOLOv5s @ 640px | One-stage baseline | — |
| E1b | YOLOv5s @ 1280px | Input resolution (↑2×) | E1a |
| E2a | Faster R-CNN no FPN | Two-stage, single-scale | E1a |
| E2b | Faster R-CNN + FPN | Feature Pyramid Network | E2a |
| E3 | YOLOv5s + SAHI | Inference-time tiling | E1a |
| E4 | Error analysis | Cross-cutting synthesis | All |

## Cloud training (Modal)

`modal_app.py` runs experiments on **Modal.com** (A100-40GB, ~$3–4 per full run). Two Modal Volumes required:

| Volume | Mount | Content |
|--------|-------|---------|
| `visdrone-data` | `/data` | Dataset (images + annotations) |
| `visdrone-outputs` | `/outputs` | Experiment results (checkpoints, metrics, figures) |

### Setup

```bash
# 1. Upload dataset to volume
modal volume put visdrone-data data/VisDrone2019-DET/images /VisDrone2019-DET/images
modal volume put visdrone-data data/VisDrone2019-DET/annotations_raw /VisDrone2019-DET/annotations_raw

# 2. Convert annotations to YOLO+ format (run once)
modal run modal_app.py::setup_data

# 3. Train experiment
modal run modal_app.py::train_experiment --experiment-name e1b --debug

# 4. Download results locally
modal run modal_app.py::download_results --experiment-name e1b
```

### All entrypoints

| Command | Description |
|---------|-------------|
| `train_experiment` | Train E1a/E1b (YOLO) or E2a/E2b (FRCNN). `--debug` = 2 epochs on debug_500 |
| `revalidate` | Re-run validation on existing checkpoint with current code |
| `setup_data` | Convert raw VisDrone annotations → YOLO+ format on volume |
| `download_results` | Pull experiment dir from `visdrone-outputs` to `results/runs/` |
| `list_outputs` | List all experiment outputs on the volume |

`VISDRONE_ENV=modal` env var switches paths + device config. Locale runs on MPS (Apple Silicon) or CUDA. Single codebase, zero manual edits between environments.

## Setup

```bash
conda env create -f environment.yml
conda activate visdrone
```

## Repository structure

```
project/
├── data/           ← VisDrone2019-DET dataset + debug subsets
├── src/            ← Python modules (env, datasets, trainers, metrics, ...)
├── notebooks/      ← Jupyter notebooks (00_setup → 04_analysis)
├── configs/        ← YAML configs per experiment
├── models/         ← Pretrained weights (yolov5su.pt) — gitignored
├── results/        ← Checkpoints, metrics, figures per run
├── report/         ← LaTeX report
├── modal_app.py    ← Modal.com cloud entrypoint
├── environment.yml ← Conda environment spec
└── requirements.txt← Pip deps (subset for Modal image)
```

## Final ranking (mAP@0.5)

| Rank | Experiment | mAP@0.5 |
|------|-----------|:-------:|
| 1 | **E1b** — YOLOv5s @ 1280px | **0.524** |
| 2 | E1a — YOLOv5s @ 640px | 0.371 |
| 3 | E3 — YOLOv5s + SAHI | 0.289 |
| 4 | E2b — Faster R-CNN + FPN | 0.249 |
| 5 | E2a — Faster R-CNN no FPN | 0.083 |

**Key finding:** doubling input resolution (E1a → E1b) doubles AP_small (+100%). Resolution is the bottleneck, not architecture.
