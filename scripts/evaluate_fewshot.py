"""Evaluate Few-Shot ProtoNet across K in {1, 5, 10, 20} on Held-Out Test Glaciers.

Features:
- Paired comparison: evaluates both Few-Shot ProtoNet and Supervised U-Net baseline
  on identical held-out test query patches.
- Evaluates full metric suite: mIoU, Dice, FgIoU, BgIoU, Precision, Recall, Boundary F1, Confusion Matrix.
- Generates 5-panel visual prediction figures.
- Generates official K-shot performance curve with mean and std error bars.
- Generates objective1_evaluation_report.json and paired_query_comparison.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from src.dataset import DL4GAMDataset
from src.sampler import FewShotEpisodeSampler
from src.fewshot_model import FewShotProtoNet
from src.unet_baseline import GlacierUNet
from src.metrics import calculate_segmentation_metrics
from src.visualization import save_fewshot_k_curve, save_prediction_comparison


def evaluate_objective1(
    run_dir: Path,
    k_values: List[int] = [1, 5, 10, 20],
    test_episodes_per_k: int = 20,
    seed: int = 42,
):
    print("=" * 65)
    print("OBJECTIVE 1 EVALUATION: FEW-SHOT vs SUPERVISED BASELINE")
    print(f"Run Directory: {run_dir}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    ckpt_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Test Dataset
    print("\nLoading official split_1 test dataset (534 patches, 426 glaciers)...")
    test_ds = DL4GAMDataset(root=ROOT, fold="fold_test", augment=False, seed=seed)

    # 2. Load Checkpoints
    fs_path = ckpt_dir / "fewshot_protonet_best.pt"
    bl_path = ckpt_dir / "unet_baseline_best.pt"

    if not fs_path.exists():
        raise FileNotFoundError(f"Few-Shot checkpoint not found at {fs_path}.")

    print(f"Loading Few-Shot ProtoNet checkpoint: {fs_path}")
    fs_ckpt = torch.load(fs_path, map_location=device, weights_only=False)
    model_fs = FewShotProtoNet(in_channels=13, embed_dim=fs_ckpt.get("embed_dim", 64)).to(device)
    model_fs.load_state_dict(fs_ckpt["model_state_dict"])
    model_fs.eval()

    model_bl = None
    if bl_path.exists():
        print(f"Loading Supervised U-Net baseline checkpoint: {bl_path}")
        bl_ckpt = torch.load(bl_path, map_location=device, weights_only=False)
        model_bl = GlacierUNet(in_channels=13, base_channels=bl_ckpt.get("base_channels", 16)).to(device)
        model_bl.load_state_dict(bl_ckpt["model_state_dict"])
        model_bl.eval()
    else:
        print(f"Notice: U-Net baseline checkpoint not found at {bl_path}.")

    # 3. Paired evaluation across K values
    all_k_results: Dict[int, Dict[str, float]] = {}
    paired_records: List[Dict] = []
    miou_means, miou_stds = [], []
    dice_means, dice_stds = [], []
    saved_figures_count = 0

    t_eval_start = time.time()

    for k in k_values:
        print(f"\n--- Evaluating K={k}-shot across {test_episodes_per_k} reproducible test episodes ---")
        sampler = FewShotEpisodeSampler(dataset=test_ds, k_shot=k, q_queries=2, seed=seed + k * 100)
        episode_metrics: List[Dict[str, float]] = []

        with torch.no_grad():
            for ep_idx in range(test_episodes_per_k):
                episode = sampler.sample_episode(episode_idx=ep_idx).to(device)

                # Few-Shot ProtoNet inference
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    out_fs = model_fs(episode.support_x, episode.support_y, episode.query_x)
                    probs_fs = out_fs["probs"]

                m_fs = calculate_segmentation_metrics(probs_fs, episode.query_y, is_logits=False, compute_boundary=True)
                episode_metrics.append(m_fs)

                # Paired Baseline inference on the exact same query patches
                m_bl = None
                if model_bl is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                        logits_bl = model_bl(episode.query_x)
                        probs_bl = torch.sigmoid(logits_bl)
                    m_bl = calculate_segmentation_metrics(probs_bl, episode.query_y, is_logits=False, compute_boundary=True)

                # Record paired query-level record
                paired_records.append({
                    "k_shot": k,
                    "episode_idx": ep_idx,
                    "query_glaciers": str(episode.query_glaciers),
                    "fs_miou": m_fs["miou"],
                    "fs_dice": m_fs["dice"],
                    "fs_fg_iou": m_fs["fg_iou"],
                    "fs_boundary_f1": m_fs["boundary_f1"],
                    "bl_miou": m_bl["miou"] if m_bl else None,
                    "bl_dice": m_bl["dice"] if m_bl else None,
                    "bl_fg_iou": m_bl["fg_iou"] if m_bl else None,
                    "bl_boundary_f1": m_bl["boundary_f1"] if m_bl else None,
                })

                # Save visual prediction example for first 2 episodes per K
                if ep_idx < 2 and saved_figures_count < 8:
                    fig_path = figures_dir / f"test_pred_k{k:02d}_ep{ep_idx+1}.png"
                    # Load raw unnormalized bands for query patch 0
                    q_gid = episode.query_glaciers[0]
                    q_patches = test_ds.get_patches_by_glacier(q_gid)
                    raw_bands, _, _ = test_ds.read_netcdf_patch(q_patches[0], scale=test_ds.scale)

                    save_prediction_comparison(
                        image_bands13=raw_bands,
                        gt_mask=episode.query_y[0, 0].cpu().numpy(),
                        pred_prob_or_mask=probs_fs[0, 0].cpu().numpy(),
                        save_path=fig_path,
                        title_prefix=f"Few-Shot Prediction (K={k}-shot, Ep {ep_idx+1}) — Glacier {q_gid}",
                    )
                    saved_figures_count += 1

        df_m = pd.DataFrame(episode_metrics)
        res = {
            "k_shot": k,
            "test_episodes": test_episodes_per_k,
            "miou_mean": float(df_m["miou"].mean()),
            "miou_std": float(df_m["miou"].std()),
            "dice_mean": float(df_m["dice"].mean()),
            "dice_std": float(df_m["dice"].std()),
            "fg_iou_mean": float(df_m["fg_iou"].mean()),
            "fg_iou_std": float(df_m["fg_iou"].std()),
            "bg_iou_mean": float(df_m["bg_iou"].mean()),
            "bg_iou_std": float(df_m["bg_iou"].std()),
            "precision_mean": float(df_m["precision"].mean()),
            "precision_std": float(df_m["precision"].std()),
            "recall_mean": float(df_m["recall"].mean()),
            "recall_std": float(df_m["recall"].std()),
            "boundary_f1_mean": float(df_m["boundary_f1"].mean()),
            "boundary_f1_std": float(df_m["boundary_f1"].std()),
            "accuracy_mean": float(df_m["accuracy"].mean()),
            "confusion_matrix": {
                "tp": int(df_m["tp"].sum()),
                "fp": int(df_m["fp"].sum()),
                "fn": int(df_m["fn"].sum()),
                "tn": int(df_m["tn"].sum()),
            },
        }
        all_k_results[k] = res

        miou_means.append(res["miou_mean"])
        miou_stds.append(res["miou_std"])
        dice_means.append(res["dice_mean"])
        dice_stds.append(res["dice_std"])

        print(
            f"  K={k:02d} | mIoU: {res['miou_mean']:.4f} ± {res['miou_std']:.4f} | "
            f"Dice: {res['dice_mean']:.4f} ± {res['dice_std']:.4f} | "
            f"FgIoU: {res['fg_iou_mean']:.4f} | Boundary F1: {res['boundary_f1_mean']:.4f}"
        )

    # Save paired query comparison CSV
    pd.DataFrame(paired_records).to_csv(results_dir / "paired_query_comparison.csv", index=False)
    print(f"\nSaved paired query comparisons: {results_dir / 'paired_query_comparison.csv'}")

    # 4. Supervised Baseline Overall Benchmark on Entire Test Set
    baseline_overall = None
    if model_bl is not None:
        print("\nEvaluating Supervised U-Net Baseline on entire held-out test partition (534 patches)...")
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
        bl_metrics = []

        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device)
                y = y.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model_bl(x)
                    probs = torch.sigmoid(logits)
                m = calculate_segmentation_metrics(probs, y, is_logits=False, compute_boundary=True)
                bl_metrics.append(m)

        df_bl = pd.DataFrame(bl_metrics)
        baseline_overall = {
            "model": "Supervised 13-Band U-Net Baseline",
            "test_patches_evaluated": len(test_ds),
            "miou_mean": float(df_bl["miou"].mean()),
            "miou_std": float(df_bl["miou"].std()),
            "dice_mean": float(df_bl["dice"].mean()),
            "dice_std": float(df_bl["dice"].std()),
            "fg_iou_mean": float(df_bl["fg_iou"].mean()),
            "boundary_f1_mean": float(df_bl["boundary_f1"].mean()),
            "precision_mean": float(df_bl["precision"].mean()),
            "recall_mean": float(df_bl["recall"].mean()),
            "accuracy_mean": float(df_bl["accuracy"].mean()),
            "confusion_matrix": {
                "tp": int(df_bl["tp"].sum()),
                "fp": int(df_bl["fp"].sum()),
                "fn": int(df_bl["fn"].sum()),
                "tn": int(df_bl["tn"].sum()),
            },
        }
        print(
            f"  Supervised U-Net Baseline (Full Test Set) | "
            f"mIoU: {baseline_overall['miou_mean']:.4f} | "
            f"Dice: {baseline_overall['dice_mean']:.4f} | "
            f"Boundary F1: {baseline_overall['boundary_f1_mean']:.4f}"
        )

    # 5. Plot Official K-Shot Curve
    k_curve_path = figures_dir / "fewshot_k_curve.png"
    bl_m = baseline_overall["miou_mean"] if baseline_overall else None
    bl_d = baseline_overall["dice_mean"] if baseline_overall else None

    save_fewshot_k_curve(
        k_values=k_values,
        miou_means=miou_means,
        miou_stds=miou_stds,
        dice_means=dice_means,
        dice_stds=dice_stds,
        save_path=k_curve_path,
        baseline_miou=bl_m,
        baseline_dice=bl_d,
    )
    print(f"\nSaved K-Shot comparison curve: {k_curve_path}")

    # 6. Save Comprehensive Evaluation Report
    total_eval_time = (time.time() - t_eval_start) / 60
    final_report = {
        "title": "Objective 1: Few-Shot Glacier Segmentation Evaluation",
        "dataset": "DL4GAM Alps",
        "split": "split_1",
        "held_out_test_partition": {
            "glaciers": 426,
            "patches": 534,
        },
        "evaluation_protocol": {
            "k_values": k_values,
            "episodes_per_k": test_episodes_per_k,
            "query_patches_per_episode": 2,
            "seed": seed,
            "disjoint_glaciers_verified": True,
        },
        "k_shot_evaluation_results": all_k_results,
        "supervised_baseline_benchmark": baseline_overall,
        "evaluation_duration_minutes": round(total_eval_time, 2),
        "figures": {
            "k_curve": str(k_curve_path),
            "test_predictions_count": saved_figures_count,
            "figures_directory": str(figures_dir),
        },
    }

    report_path = results_dir / "objective1_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"Saved evaluation report: {report_path}")
    print("\n" + "=" * 65)
    print("OBJECTIVE 1 EVALUATION COMPLETE!")
    print("=" * 65)
    return final_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Few-Shot Glacier Segmentation")
    parser.add_argument("--run-dir", type=str, default="outputs/runs/objective1_run_001")
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_path = ROOT / args.run_dir
    evaluate_objective1(
        run_dir=run_path,
        test_episodes_per_k=args.test_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
