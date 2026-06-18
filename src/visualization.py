"""
Visualization utilities — bbox drawing, GradCAM/EigenCAM, PR curve plots, heatmaps.

All figures are report-ready: axis labels, legends, consistent palette, DPI >= 150.
"""

import numpy as np
import matplotlib
from src.env import set_matplotlib_backend

set_matplotlib_backend()  # Agg only on Modal; leaves notebook backend alone
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional


# Consistent color palette for all experiments
COLORS = plt.cm.tab10.colors  # 10 distinct colors for 10 classes

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def set_plot_style():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.format": "png",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def draw_bboxes(image: np.ndarray, boxes: list, labels: list,
                scores: Optional[list] = None,
                class_names: Optional[list] = None) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: RGB image array (H, W, 3), values in [0, 255] or [0, 1]
        boxes: list of [x1, y1, x2, y2] in pixel coordinates
        labels: list of integer class labels
        scores: optional list of confidence scores
        class_names: optional list of class name strings (default: VISDRONE_CLASSES)

    Returns:
        Image array with overlaid boxes (RGB, uint8)
    """
    if class_names is None:
        class_names = VISDRONE_CLASSES

    # Ensure image is uint8 in [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(image)
    ax.axis("off")

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = COLORS[int(label) % len(COLORS)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        label_text = class_names[int(label)] if int(label) < len(class_names) else str(label)
        if scores is not None:
            label_text = f"{label_text} {scores[i]:.2f}"
        ax.text(
            x1, max(y1 - 4, 0), label_text,
            fontsize=7, color="white",
            bbox=dict(facecolor=color, alpha=0.7, pad=1, boxstyle="round,pad=0.2"),
        )

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[..., :3]  # drop alpha channel
    plt.close(fig)
    return img


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, class_name: str,
                  save_path: Optional[Path] = None,
                  ap_value: Optional[float] = None):
    """
    Plot precision-recall curve for a single class.

    Args:
        precision: array of precision values (101 pts from COCOeval)
        recall: array of recall values (101 pts, 0 to 1)
        class_name: class label for title
        save_path: optional path to save figure
        ap_value: optional AP value to display in legend
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    label = f"AP={ap_value:.3f}" if ap_value is not None else ""
    ax.plot(recall, precision, linewidth=2, color="steelblue", label=label)
    ax.fill_between(recall, precision, alpha=0.15, color="steelblue")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {class_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_ap_heatmap(ap_dict: dict, save_path: Optional[Path] = None,
                            show: bool = False):
    """
    Plot heatmap: N configurations x 10 classes.

    Args:
        ap_dict: dict mapping config_name -> list of AP per class (len 10)
        save_path: optional path to save figure
        show: if True, call plt.show() before closing (inline notebook display)
    """
    configs = list(ap_dict.keys())
    n_configs = len(configs)
    data = np.array([ap_dict[c] for c in configs])

    fig, ax = plt.subplots(figsize=(10, max(3, n_configs * 0.6)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(10))
    ax.set_xticklabels(VISDRONE_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(configs, fontsize=9)

    for i in range(n_configs):
        for j in range(10):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("AP", fontsize=9)
    ax.set_title("Per-Class AP", fontsize=12)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_summary_table(metrics: dict, save_path: Optional[Path] = None,
                       show: bool = False):
    """
    Plot the full metrics summary table.

    Args:
        metrics: dict mapping config_name -> dict of metric_name -> value
        save_path: optional path to save figure
        show: if True, call plt.show() before closing (inline notebook display)
    """
    configs = list(metrics.keys())
    metric_names = list(metrics[configs[0]].keys())
    n_configs = len(configs)
    n_metrics = len(metric_names)

    # Filter out non-scalar metrics (lists, dicts)
    scalar_metrics = [m for m in metric_names
                      if isinstance(metrics[configs[0]][m], (int, float))]
    if not scalar_metrics:
        scalar_metrics = metric_names

    cell_text = []
    for cfg in configs:
        row = []
        for m in scalar_metrics:
            val = metrics[cfg][m]
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            elif isinstance(val, int):
                row.append(str(val))
            else:
                row.append(str(val))
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(8, n_configs * 3), max(3, n_metrics * 0.5)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=configs,
        colLabels=scalar_metrics,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Highlight best value per column (higher is better)
    for j, m in enumerate(scalar_metrics):
        col_vals = [metrics[c][m] for c in configs]
        if all(isinstance(v, (int, float)) for v in col_vals):
            best_idx = int(np.argmax(col_vals))
            table[(best_idx + 1, j)].set_facecolor("lightgreen")

    ax.set_title("Experiment Summary — Metrics Comparison", fontsize=12, pad=20)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_altitude_performance(altitude_groups: dict, save_path: Optional[Path] = None):
    """
    Box plot: altitude group vs mAP.

    Args:
        altitude_groups: dict mapping altitude_range_label -> list of per-image mAPs
        save_path: optional path to save figure
    """
    labels = list(altitude_groups.keys())
    data = [altitude_groups[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Altitude range")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Performance vs Altitude (proxy: median bbox area)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[list] = None,
                          save_path: Optional[Path] = None):
    """
    Plot confusion matrix heatmap.

    Args:
        cm: (C, C) numpy array, rows=ground truth, cols=predicted
        class_names: list of class names (default: VISDRONE_CLASSES)
        save_path: optional path to save figure
    """
    if class_names is None:
        class_names = VISDRONE_CLASSES
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=8)

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Count", fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=12)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
