"""Train Prototypical Few-Shot Segmentation Network on DL4GAM Alps.

Supports custom --run-dir to preserve previous outputs.
Validates exclusively on split_1 validation glaciers with early stopping.
Guarantees zero glacier-level leakage.
Records training duration, GPU memory, and validation metrics history.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from src.dataset import DL4GAMDataset
from src.sampler import FewShotEpisodeSampler
from src.fewshot_model import FewShotProtoNet
from src.metrics import calculate_segmentation_metrics, compute_combined_loss


def train_fewshot(
    run_dir: Path,
    epochs: int = 15,
    episodes_per_epoch: int = 50,
    val_episodes: int = 15,
    k_shot: int = 5,
    q_queries: int = 2,
    lr: float = 1e-3,
    embed_dim: int = 64,
    temperature: float = 20.0,
    patience: int = 4,
    seed: int = 42,
):
    print("=" * 65)
    print(f"TRAINING FEW-SHOT PROTONET (K={k_shot}, Q={q_queries})")
    print(f"Run Directory: {run_dir}")
    print("=" * 65)

    ckpt_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    total_vram = 0.0
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**3)
        print(f"GPU Model:  {props.name} ({total_vram:.2f} GB VRAM)")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Save run configuration
    config = {
        "model": "FewShotProtoNet",
        "objective": "Objective 1: Few-Shot Glacier Segmentation",
        "split": "split_1",
        "train_glaciers": 986,
        "train_patches": 1475,
        "val_glaciers": 181,
        "val_patches": 242,
        "test_glaciers": 426,
        "test_patches": 534,
        "k_shot": k_shot,
        "q_queries": q_queries,
        "epochs": epochs,
        "episodes_per_epoch": episodes_per_epoch,
        "val_episodes": val_episodes,
        "lr": lr,
        "embed_dim": embed_dim,
        "temperature": temperature,
        "patience": patience,
        "seed": seed,
        "mixed_precision": "FP16 (torch.amp.autocast)",
        "channel_order": DL4GAMDataset.BAND_NAMES,
    }
    with open(run_dir / "fewshot_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Initialize datasets
    print("\nLoading datasets with official split_1 partitions...")
    train_ds = DL4GAMDataset(root=ROOT, fold="fold_train", augment=True, seed=seed)
    val_ds = DL4GAMDataset(root=ROOT, fold="fold_valid", augment=False, seed=seed + 1)

    train_sampler = FewShotEpisodeSampler(train_ds, k_shot=k_shot, q_queries=q_queries, seed=seed)
    val_sampler = FewShotEpisodeSampler(val_ds, k_shot=k_shot, q_queries=q_queries, seed=seed + 999)

    model = FewShotProtoNet(in_channels=13, embed_dim=embed_dim, temperature=temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_dice = -1.0
    best_ckpt_path = ckpt_dir / "fewshot_protonet_best.pt"
    history = []
    bad_epochs = 0
    t_start = time.time()

    print("\nStarting episodic training...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses, train_dices, train_mious = [], [], []

        for ep_idx in range(episodes_per_epoch):
            episode = train_sampler.sample_episode(episode_idx=ep_idx + (epoch - 1) * episodes_per_epoch).to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                out = model(episode.support_x, episode.support_y, episode.query_x)
                loss = compute_combined_loss(out["logits"], episode.query_y, pos_weight=train_ds.pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            with torch.no_grad():
                m = calculate_segmentation_metrics(out["logits"], episode.query_y, is_logits=True, compute_boundary=False)
                train_dices.append(m["dice"])
                train_mious.append(m["miou"])

        # Validation phase on unseen validation glaciers
        model.eval()
        val_losses, val_dices, val_mious, val_bf1s = [], [], [], []

        with torch.no_grad():
            for ep_idx in range(val_episodes):
                ep = val_sampler.sample_episode(episode_idx=ep_idx).to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    out = model(ep.support_x, ep.support_y, ep.query_x)
                    v_loss = compute_combined_loss(out["logits"], ep.query_y, pos_weight=val_ds.pos_weight)

                val_losses.append(v_loss.item())
                m = calculate_segmentation_metrics(out["logits"], ep.query_y, is_logits=True, compute_boundary=True)
                val_dices.append(m["dice"])
                val_mious.append(m["miou"])
                val_bf1s.append(m["boundary_f1"])

        avg_tr_loss = float(pd.Series(train_losses).mean())
        avg_tr_dice = float(pd.Series(train_dices).mean())
        avg_tr_miou = float(pd.Series(train_mious).mean())

        avg_val_loss = float(pd.Series(val_losses).mean())
        avg_val_dice = float(pd.Series(val_dices).mean())
        avg_val_miou = float(pd.Series(val_mious).mean())
        avg_val_bf1 = float(pd.Series(val_bf1s).mean())

        scheduler.step(avg_val_dice)

        epoch_vram = 0.0
        if device.type == "cuda":
            epoch_vram = torch.cuda.max_memory_allocated(0) / (1024**2)

        row = {
            "epoch": epoch,
            "train_loss": avg_tr_loss,
            "train_dice": avg_tr_dice,
            "train_miou": avg_tr_miou,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_miou": avg_val_miou,
            "val_boundary_f1": avg_val_bf1,
            "lr": optimizer.param_groups[0]["lr"],
            "peak_vram_mb": epoch_vram,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(results_dir / "fewshot_protonet_history.csv", index=False)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {avg_tr_loss:.4f} Dice: {avg_tr_dice:.4f} mIoU: {avg_tr_miou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Dice: {avg_val_dice:.4f} mIoU: {avg_val_miou:.4f} BF1: {avg_val_bf1:.4f} | "
            f"VRAM: {epoch_vram:.1f}MB | {row['elapsed_sec']}s"
        )

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_dice": best_val_dice,
                    "best_val_miou": avg_val_miou,
                    "k_shot": k_shot,
                    "q_queries": q_queries,
                    "embed_dim": embed_dim,
                    "config": config,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    duration_min = (time.time() - t_start) / 60
    final_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0
    print(f"\nFew-Shot Training Finished in {duration_min:.1f} minutes.")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Peak GPU VRAM Allocated: {final_vram:.2f} MB ({final_vram / 1024:.2f} GB / {total_vram:.2f} GB)")

    summary = {
        "training_duration_minutes": round(duration_min, 2),
        "total_epochs_run": len(history),
        "best_epoch": int(pd.DataFrame(history)["val_dice"].idxmax()) + 1,
        "best_val_dice": best_val_dice,
        "peak_vram_mb": round(final_vram, 2),
        "checkpoint_path": str(best_ckpt_path),
    }
    with open(results_dir / "fewshot_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train Few-Shot ProtoNet")
    parser.add_argument("--run-dir", type=str, default="outputs/runs/objective1_run_001")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--episodes-per-epoch", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=15)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-queries", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_path = ROOT / args.run_dir
    train_fewshot(
        run_dir=run_path,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        val_episodes=args.val_episodes,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
