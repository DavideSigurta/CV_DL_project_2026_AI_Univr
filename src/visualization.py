"""
Visualization utilities — bbox drawing, GradCAM/EigenCAM, PR curve plots, heatmaps.

All figures are report-ready: axis labels, legends, consistent palette, DPI >= 150.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# Consistent color palette for all experiments
COLORS = plt.cm.tab10.colors  # 10 distinct colors for 10 classes


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


def draw_bboxes(image: np.ndarray, boxes: list, labels: list, scores: list = None) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: RGB image array (H, W, 3)
        boxes: list of [x1, y1, x2, y2] in pixel coordinates
        labels: list of integer class labels
        scores: optional list of confidence scores

    Returns:
        Image array with overlaid boxes
    """
    raise NotImplementedError


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, class_name: str,
                  save_path: Optional[Path] = None):
    """
    Plot precision-recall curve for a single class.
    """
    raise NotImplementedError


def plot_per_class_ap_heatmap(ap_dict: dict, save_path: Optional[Path] = None):
    """
    Plot heatmap: 5 configurations × 10 classes.

    Args:
        ap_dict: dict mapping config_name -> list of AP per class (len 10)
    """
    raise NotImplementedError


def plot_summary_table(metrics: dict, save_path: Optional[Path] = None):
    """
    Plot the full metrics summary table (Section 4.1).
    """
    raise NotImplementedError


def plot_altitude_performance(altitude_groups: dict, save_path: Optional[Path] = None):
    """
    Violin/box plot: altitude group vs mAP.
    """
    raise NotImplementedError
