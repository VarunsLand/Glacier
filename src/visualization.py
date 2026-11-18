"""Visual Results and Diagnostic Figure Generator for Glacier Monitoring.

Produces:
1. Original Sentinel-2 True Color RGB composite (B4/B3/B2).
2. Ground-truth glacier mask.
3. Predicted glacier mask.
4. Boundary overlay on satellite image.
5. Error map: True Positives, False Positives, False Negatives.
6. K-shot performance comparison graphs (1, 5, 10, 20 shots).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.metrics import extract_boundary


def get_rgb_composite(bands13: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Create a realistic True-Color RGB composite from 13 Sentinel-2 bands."""
    if isinstance(bands13, torch.Tensor):
        bands13 = bands13.detach().cpu().numpy()

    # DL4GAM band indexing: B1=0, B2(Blue)=1, B3(Green)=2, B4(Red)=3
    # Sentinel-2 true color composite: Red=B4, Green=B3, Blue=B2
    r = bands13[3]
    g = bands13[2]
    b = bands13[1]

    rgb = np.stack([r, g, b], axis=-1)

    # Robust 2% - 98% percentile stretch
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


def save_prediction_comparison(
    image_bands13: Union[torch.Tensor, np.ndarray],
    gt_mask: Union[torch.Tensor, np.ndarray],
    pred_prob_or_mask: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    title_prefix: str = "Glacier Prediction",
    threshold: float = 0.5,
) -> None:
    """Generate and save the 5-panel genuine visual comparison."""
    if isinstance(image_bands13, torch.Tensor):
        image_bands13 = image_bands13.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_prob_or_mask, torch.Tensor):
        pred_prob_or_mask = pred_prob_or_mask.detach().cpu().numpy()

    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]
    if pred_prob_or_mask.ndim == 3:
        pred_prob_or_mask = pred_prob_or_mask[0]

    rgb = get_rgb_composite(image_bands13)
    binary_gt = (gt_mask >= 0.5).astype(np.uint8)
    binary_pred = (pred_prob_or_mask >= threshold).astype(np.uint8)

    # Extract contours
    t_gt = torch.from_numpy(binary_gt[None, None, :, :]).float()
    t_pr = torch.from_numpy(binary_pred[None, None, :, :]).float()
    b_gt = extract_boundary(t_gt)[0, 0].numpy() > 0.5
    b_pr = extract_boundary(t_pr)[0, 0].numpy() > 0.5

    # Create error map: TP (white/gray), FP (red), FN (blue), TN (black)
    error_map = np.zeros((*binary_gt.shape, 3), dtype=np.float32)
    tp = (binary_pred == 1) & (binary_gt == 1)
    fp = (binary_pred == 1) & (binary_gt == 0)
    fn = (binary_pred == 0) & (binary_gt == 1)

    error_map[tp] = [0.2, 0.9, 0.2]  # True Positive: Green
    error_map[fp] = [1.0, 0.1, 0.1]  # False Positive: Red
    error_map[fn] = [0.2, 0.4, 1.0]  # False Negative: Blue

    # Create overlay
    overlay = rgb.copy()
    overlay[b_gt] = [0.0, 1.0, 0.0]  # Ground truth boundary: Bright Green
    overlay[b_pr] = [1.0, 0.0, 0.0]  # Predicted boundary: Bright Red

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), dpi=150)

    axes[0].imshow(rgb)
    axes[0].set_title("1. Sentinel-2 RGB (B4/B3/B2)", fontsize=11, fontweight="bold")

    axes[1].imshow(binary_gt, cmap="Blues_r", vmin=0, vmax=1)
    axes[1].set_title("2. Ground Truth Mask", fontsize=11, fontweight="bold")

    axes[2].imshow(binary_pred, cmap="Blues_r", vmin=0, vmax=1)
    axes[2].set_title("3. Predicted Glacier Mask", fontsize=11, fontweight="bold")

    axes[3].imshow(overlay)
    axes[3].set_title("4. Boundary Overlay\n(Green: GT, Red: Pred)", fontsize=11, fontweight="bold")

    axes[4].imshow(error_map)
    axes[4].set_title("5. Error Map\n(Green: TP, Red: FP, Blue: FN)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title_prefix, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_fewshot_k_curve(
    k_values: List[int],
    miou_means: List[float],
    miou_stds: List[float],
    dice_means: List[float],
    dice_stds: List[float],
    save_path: Union[str, Path],
    baseline_miou: Optional[float] = None,
    baseline_dice: Optional[float] = None,
) -> None:
    """Plot K-shot performance comparison curve across K in {1, 5, 10, 20}."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    ax.errorbar(
        k_values,
        miou_means,
        yerr=miou_stds,
        fmt="o-",
        color="#1f77b4",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        label="Few-Shot Mean IoU (mIoU)",
    )
    ax.errorbar(
        k_values,
        dice_means,
        yerr=dice_stds,
        fmt="s--",
        color="#2ca02c",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        label="Few-Shot Dice Coefficient",
    )

    if baseline_miou is not None:
        ax.axhline(
            baseline_miou,
            color="#d62728",
            linestyle=":",
            linewidth=2,
            label=f"Supervised U-Net Baseline mIoU ({baseline_miou:.3f})",
        )
    if baseline_dice is not None:
        ax.axhline(
            baseline_dice,
            color="#ff7f0e",
            linestyle=":",
            linewidth=2,
            label=f"Supervised U-Net Baseline Dice ({baseline_dice:.3f})",
        )

    ax.set_xlabel("Number of Support Shots (K)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Segmentation Metric Score", fontsize=12, fontweight="bold")
    ax.set_title("Objective 1: Few-Shot Glacier Segmentation vs Support Shots (K)", fontsize=13, fontweight="bold")
    ax.set_xticks(k_values)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
