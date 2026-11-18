"""Comprehensive Evaluation Metrics for Glacier Segmentation.

Calculates:
- Foreground IoU (Glacier IoU)
- Background IoU
- Mean IoU (mIoU)
- Dice Similarity Coefficient (DSC)
- Precision, Recall, F1 Score
- Boundary F1 Score (contour alignment metric)
- Confusion Matrix elements (TP, FP, FN, TN)
- Handles empty masks and zero-division cases with epsilon guardrails.
"""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F


def extract_boundary(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Extract binary boundary contours from binary mask using morphological dilation/erosion."""
    # mask shape: (B, 1, H, W) float {0.0, 1.0}
    kernel_size = 2 * radius + 1
    pad = radius
    max_pooled = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    min_pooled = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (max_pooled - min_pooled) > 0.5
    return boundary.float()


def calculate_segmentation_metrics(
    logits_or_probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    is_logits: bool = True,
    compute_boundary: bool = True,
) -> Dict[str, float]:
    """Calculate comprehensive segmentation metrics for a batch or single image."""
    if isinstance(logits_or_probs, np.ndarray):
        logits_or_probs = torch.from_numpy(logits_or_probs)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    if logits_or_probs.ndim == 2:
        logits_or_probs = logits_or_probs[None, None, :, :]
    elif logits_or_probs.ndim == 3:
        logits_or_probs = logits_or_probs[:, None, :, :]

    if targets.ndim == 2:
        targets = targets[None, None, :, :]
    elif targets.ndim == 3:
        targets = targets[:, None, :, :]

    if is_logits:
        probs = torch.sigmoid(logits_or_probs)
    else:
        probs = logits_or_probs

    preds = (probs >= threshold).bool()
    truth = (targets >= 0.5).bool()

    tp = (preds & truth).sum().item()
    fp = (preds & ~truth).sum().item()
    fn = (~preds & truth).sum().item()
    tn = (~preds & ~truth).sum().item()

    eps = 1e-8

    # Foreground (glacier) metrics
    fg_iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    f1 = dice

    # Background metrics
    bg_iou = tn / (tn + fp + fn + eps)

    # Mean IoU
    miou = (fg_iou + bg_iou) / 2.0

    # Overall Pixel Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    # Boundary F1 Score
    boundary_f1 = 0.0
    if compute_boundary:
        with torch.no_grad():
            b_pred = extract_boundary(preds.float())
            b_true = extract_boundary(truth.float())
            
            b_tp = ((b_pred > 0.5) & (b_true > 0.5)).sum().item()
            b_fp = ((b_pred > 0.5) & ~(b_true > 0.5)).sum().item()
            b_fn = (~(b_pred > 0.5) & (b_true > 0.5)).sum().item()
            
            b_prec = b_tp / (b_tp + b_fp + eps)
            b_rec = b_tp / (b_tp + b_fn + eps)
            boundary_f1 = (2.0 * b_prec * b_rec) / (b_prec + b_rec + eps)

    return {
        "fg_iou": float(fg_iou),
        "bg_iou": float(bg_iou),
        "miou": float(miou),
        "dice": float(dice),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "boundary_f1": float(boundary_f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice Loss for backpropagation."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    cardinality = probs.sum(dims) + targets.sum(dims)
    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    return (1.0 - dice_score).mean()


def compute_combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 3.18,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Combined weighted BCE and Dice Loss for robust handling of class imbalance."""
    pw = torch.tensor([pos_weight], device=logits.device, dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    dl = dice_loss(logits, targets)
    return bce_weight * bce + dice_weight * dl
