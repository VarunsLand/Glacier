"""Train Supervised 13-Band U-Net Baseline on DL4GAM Alps.

Supports custom --run-dir to preserve previous outputs.
Trained strictly on the official split_1 training partition (1,475 patches).
Validates on split_1 validation partition (242 patches) with early stopping.
Strictly distinguished from the Few-Shot ProtoNet model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Prevent Windows standby during training
if sys.platform == "win32":
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import DL4GAMDataset
from src.unet_baseline import GlacierUNet
from src.metrics import calculate_segmentation_metrics, compute_combined_loss


def train_baseline(
    run_dir: Path,
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 1e-3,
    base_channels: int = 16,
    patience: int = 4,
    seed: int = 42,
):
    print("=" * 65)
    print("TRAINING SUPERVISED 13-BAND U-NET BASELINE")
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

    config = {
        "model": "GlacierUNet (Supervised Baseline)",
        "objective": "Objective 1: Supervised Baseline Benchmark",
        "split": "split_1",
        "train_patches": 1475,
        "val_patches": 242,
        "test_patches": 534,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "base_channels": base_channels,
        "patience": patience,
        "seed": seed,
        "mixed_precision": "FP16 (torch.amp.autocast)",
        "channel_order": DL4GAMDataset.BAND_NAMES,
    }
    with open(run_dir / "baseline_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nLoading training (1,475 patches) and validation (242 patches) sets...")
    train_ds = DL4GAMDataset(root=ROOT, fold="fold_train", augment=True, seed=seed)
    val_ds = DL4GAMDataset(root=ROOT, fold="fold_valid", augment=False, seed=seed + 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=False
    )

    model = GlacierUNet(in_channels=13, base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_dice = -1.0
    best_ckpt_path = ckpt_dir / "unet_baseline_best.pt"
    history = []
    bad_epochs = 0
    t_start = time.time()

    print("\nStarting supervised training across 1,475 patches...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = compute_combined_loss(logits, y, pos_weight=train_ds.pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            if n_batches % 200 == 0 or n_batches == len(train_loader):
                print(f"  [Epoch {epoch:02d} | Step {n_batches:03d}/{len(train_loader)}] Current Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

        avg_tr_loss = train_loss / max(n_batches, 1)

        # Validation phase on unseen validation partition
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_metrics = []

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = compute_combined_loss(logits, y, pos_weight=val_ds.pos_weight)

                val_loss += loss.item()
                val_batches += 1
                m = calculate_segmentation_metrics(logits, y, is_logits=True, compute_boundary=False)
                all_metrics.append(m)

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_dice = float(pd.Series([m["dice"] for m in all_metrics]).mean())
        avg_val_miou = float(pd.Series([m["miou"] for m in all_metrics]).mean())

        scheduler.step(avg_val_dice)

        epoch_vram = 0.0
        if device.type == "cuda":
            epoch_vram = torch.cuda.max_memory_allocated(0) / (1024**2)

        row = {
            "epoch": epoch,
            "train_loss": avg_tr_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_miou": avg_val_miou,
            "lr": optimizer.param_groups[0]["lr"],
            "peak_vram_mb": epoch_vram,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(results_dir / "unet_baseline_history.csv", index=False)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {avg_tr_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Dice: {avg_val_dice:.4f} mIoU: {avg_val_miou:.4f} | "
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
                    "base_channels": base_channels,
                    "input_channels": 13,
                    "config": config,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best baseline checkpoint: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    duration_min = (time.time() - t_start) / 60
    final_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0
    print(f"\nU-Net Baseline Training Finished in {duration_min:.1f} minutes.")
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
    with open(results_dir / "baseline_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train Supervised U-Net Baseline")
    parser.add_argument("--run-dir", type=str, default="outputs/runs/objective1_run_001")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_path = ROOT / args.run_dir
    train_baseline(
        run_dir=run_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
