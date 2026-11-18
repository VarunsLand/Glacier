"""GlacierVision: Few-Shot ProtoNet Recovery and Continuation Script.

Recovers from Epoch 7 checkpoint in objective1_run_001:
- Restores original epochs 1-7 history records from fewshot_protonet_history.csv.
- Loads model weights from fewshot_protonet_best.pt (Val Dice: 0.46448).
- Initializes a fresh AdamW optimizer (Warm Restart).
- Saves all new outputs into outputs/runs/objective1_run_002_recovery/ (Zero modification to run_001).
- Saves atomic last checkpoint after EVERY epoch (model state, optimizer state, epoch, best score).
- Saves atomic best checkpoint on improvement.
- Runs epochs 8 through 15 (50 train episodes, 15 val episodes per epoch, K=5, Q=2).
- Enables Windows Sleep Prevention (SetThreadExecutionState) to avoid OS standby stalls.
- Detailed granular progress logging with flush=True to detect stalls.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import time
from datetime import datetime
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


def prevent_windows_sleep(enable: bool = True):
    """Prevent Windows from sleeping or entering modern standby during training."""
    if os.name == "nt":
        try:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            if enable:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                print("[PowerManagement] Windows system sleep prevention: ENABLED")
            else:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                print("[PowerManagement] Windows system sleep prevention: RESET")
        except Exception as e:
            print(f"[PowerManagement] Warning: Could not adjust thread execution state: {e}")


def save_atomic_checkpoint(payload: dict, target_path: Path):
    """Save checkpoint atomically via a temporary file to avoid corruption."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, target_path)


def run_recovery_training(
    original_run_dir: Path,
    recovery_run_dir: Path,
    target_final_epoch: int = 15,
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
    print("=" * 70, flush=True)
    print("GLACIERVISION: OBJECTIVE 1 FEW-SHOT RECOVERY & CONTINUATION", flush=True)
    print(f"Original Run Directory: {original_run_dir}", flush=True)
    print(f"Recovery Run Directory: {recovery_run_dir}", flush=True)
    print("=" * 70, flush=True)

    prevent_windows_sleep(True)

    ckpt_dir = recovery_run_dir / "checkpoints"
    results_dir = recovery_run_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}", flush=True)
    total_vram = 0.0
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**3)
        print(f"GPU Model:  {props.name} ({total_vram:.2f} GB VRAM)", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Inspect and restore existing history from Run 001
    orig_history_path = original_run_dir / "results" / "fewshot_protonet_history.csv"
    if not orig_history_path.exists():
        raise FileNotFoundError(f"Original history CSV missing: {orig_history_path}")

    df_orig = pd.read_csv(orig_history_path)
    print(f"\n[Step 1/5] Loaded original history: {len(df_orig)} completed epochs found.", flush=True)
    print(df_orig[["epoch", "train_loss", "val_loss", "val_dice", "val_miou", "elapsed_sec"]].to_string(index=False), flush=True)

    last_completed_epoch = int(df_orig["epoch"].max())
    best_orig_dice = float(df_orig["val_dice"].max())
    best_orig_row = df_orig.loc[df_orig["val_dice"].idxmax()]
    best_orig_epoch = int(best_orig_row["epoch"])

    print(f"\nLast completed epoch in history: {last_completed_epoch}", flush=True)
    print(f"Best recorded validation Dice:  {best_orig_dice:.6f} (at Epoch {best_orig_epoch})", flush=True)

    if last_completed_epoch >= target_final_epoch:
        print(f"All {target_final_epoch} epochs already completed. Nothing to continue.")
        return

    # Copy original history to recovery directory immediately
    rec_history_path = results_dir / "fewshot_protonet_history.csv"
    df_orig.to_csv(rec_history_path, index=False)
    print(f"Restored history saved to: {rec_history_path}", flush=True)

    # 2. Inspect and load best checkpoint from Run 001
    orig_best_ckpt = original_run_dir / "checkpoints" / "fewshot_protonet_best.pt"
    if not orig_best_ckpt.exists():
        raise FileNotFoundError(f"Original best checkpoint missing: {orig_best_ckpt}")

    print(f"\n[Step 2/5] Inspecting original checkpoint: {orig_best_ckpt}", flush=True)
    ckpt_data = torch.load(orig_best_ckpt, map_location="cpu", weights_only=False)
    saved_epoch = ckpt_data.get("epoch", "unknown")
    saved_dice = ckpt_data.get("best_val_dice", "unknown")
    print(f"Checkpoint Epoch: {saved_epoch} | Best Val Dice: {saved_dice}", flush=True)

    # Initialize model and load weights
    model = FewShotProtoNet(in_channels=13, embed_dim=embed_dim, temperature=temperature).to(device)
    load_res = model.load_state_dict(ckpt_data["model_state_dict"])
    print(f"Model load status: {load_res} (100% matched)", flush=True)

    # Also copy the initial best checkpoint into the recovery checkpoints folder
    best_ckpt_path = ckpt_dir / "fewshot_protonet_best.pt"
    last_ckpt_path = ckpt_dir / "fewshot_protonet_last.pt"
    shutil.copy2(orig_best_ckpt, best_ckpt_path)
    print(f"Seeded initial best checkpoint at: {best_ckpt_path}", flush=True)

    # 3. Warm Restart: Initialize fresh optimizer and scheduler
    print(f"\n[Step 3/5] Initializing fresh AdamW optimizer (Warm Restart)...", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Calculate patience baseline
    # Epochs since last best epoch
    epochs_since_best = last_completed_epoch - best_orig_epoch
    print(f"Warm restart status: Epochs since last best ({best_orig_epoch}): {epochs_since_best}", flush=True)

    # Save recovery configuration
    recovery_config = {
        "model": "FewShotProtoNet",
        "objective": "Objective 1: Few-Shot Glacier Segmentation (Warm Recovery)",
        "recovery_source_run": str(original_run_dir),
        "resumed_from_epoch": last_completed_epoch,
        "start_epoch": last_completed_epoch + 1,
        "target_final_epoch": target_final_epoch,
        "initial_best_val_dice": best_orig_dice,
        "initial_best_epoch": best_orig_epoch,
        "optimizer_status": "Warm Restart (fresh AdamW state)",
        "k_shot": k_shot,
        "q_queries": q_queries,
        "episodes_per_epoch": episodes_per_epoch,
        "val_episodes": val_episodes,
        "lr": lr,
        "embed_dim": embed_dim,
        "temperature": temperature,
        "patience": patience,
        "seed": seed,
        "mixed_precision": "FP16 (torch.amp.autocast)",
        "timestamp_start": datetime.now().isoformat(),
    }
    with open(recovery_run_dir / "recovery_config.json", "w", encoding="utf-8") as f:
        json.dump(recovery_config, f, indent=2)

    # 4. Initialize Data Loaders and Samplers
    print(f"\n[Step 4/5] Initializing official split_1 datasets...", flush=True)
    train_ds = DL4GAMDataset(root=ROOT, fold="fold_train", augment=True, seed=seed)
    val_ds = DL4GAMDataset(root=ROOT, fold="fold_valid", augment=False, seed=seed + 1)

    train_sampler = FewShotEpisodeSampler(train_ds, k_shot=k_shot, q_queries=q_queries, seed=seed)
    val_sampler = FewShotEpisodeSampler(val_ds, k_shot=k_shot, q_queries=q_queries, seed=seed + 999)

    history_records = df_orig.to_dict(orient="records")
    best_val_dice = best_orig_dice
    bad_epochs = epochs_since_best
    start_epoch = last_completed_epoch + 1
    t_continuation_start = time.time()

    print(f"\n[Step 5/5] Resuming training from Epoch {start_epoch} to Epoch {target_final_epoch}...", flush=True)

    try:
        for epoch in range(start_epoch, target_final_epoch + 1):
            t_epoch_start = time.time()
            model.train()
            train_losses, train_dices, train_mious = [], [], []

            print(f"\n>>> Starting Epoch {epoch:02d}/{target_final_epoch} at {datetime.now().strftime('%H:%M:%S')} <<<", flush=True)

            for ep_idx in range(episodes_per_epoch):
                t_ep0 = time.time()
                # Deterministic episode index ensuring continuous pseudo-random stream
                ep_global_idx = ep_idx + (epoch - 1) * episodes_per_epoch
                episode = train_sampler.sample_episode(episode_idx=ep_global_idx).to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    out = model(episode.support_x, episode.support_y, episode.query_x)
                    loss = compute_combined_loss(out["logits"], episode.query_y, pos_weight=train_ds.pos_weight)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                ep_loss = loss.item()
                train_losses.append(ep_loss)

                with torch.no_grad():
                    m = calculate_segmentation_metrics(out["logits"], episode.query_y, is_logits=True, compute_boundary=False)
                    train_dices.append(m["dice"])
                    train_mious.append(m["miou"])

                # Progress heartbeat every 10 episodes (and first episode)
                if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
                    running_loss = float(pd.Series(train_losses).mean())
                    running_dice = float(pd.Series(train_dices).mean())
                    vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0
                    print(
                        f"  [Epoch {epoch:02d} | Train Ep {ep_idx+1:02d}/{episodes_per_epoch}] "
                        f"Loss: {running_loss:.4f} | Dice: {running_dice:.4f} | "
                        f"Ep time: {time.time()-t_ep0:.2f}s | VRAM: {vram_mb:.1f}MB | "
                        f"{datetime.now().strftime('%H:%M:%S')}",
                        flush=True
                    )

            # Validation Phase on unseen validation partition (181 glaciers)
            print(f"  Validating Epoch {epoch:02d} on {val_episodes} validation episodes...", flush=True)
            model.eval()
            val_losses, val_dices, val_mious, val_bf1s = [], [], [], []

            with torch.no_grad():
                for v_idx in range(val_episodes):
                    ep = val_sampler.sample_episode(episode_idx=v_idx).to(device)
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
            epoch_duration = round(time.time() - t_epoch_start, 1)

            epoch_vram = 0.0
            if device.type == "cuda":
                epoch_vram = torch.cuda.max_memory_allocated(0) / (1024**2)

            # Record history
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
                "elapsed_sec": epoch_duration,
            }
            history_records.append(row)
            pd.DataFrame(history_records).to_csv(rec_history_path, index=False)

            print(
                f"\n>>> Epoch {epoch:02d}/{target_final_epoch} Summary | "
                f"Train Loss: {avg_tr_loss:.4f} Dice: {avg_tr_dice:.4f} mIoU: {avg_tr_miou:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} Dice: {avg_val_dice:.4f} mIoU: {avg_val_miou:.4f} BF1: {avg_val_bf1:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']} | VRAM: {epoch_vram:.1f}MB | Duration: {epoch_duration}s <<<",
                flush=True
            )

            # Save ATOMIC last checkpoint after EVERY epoch
            last_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_dice": avg_val_dice,
                "val_miou": avg_val_miou,
                "best_val_dice": best_val_dice,
                "k_shot": k_shot,
                "q_queries": q_queries,
                "embed_dim": embed_dim,
                "config": recovery_config,
                "timestamp": datetime.now().isoformat(),
            }
            save_atomic_checkpoint(last_payload, last_ckpt_path)
            print(f"  [Checkpoint] Saved atomic last checkpoint: {last_ckpt_path}", flush=True)

            # Check for improvement and save ATOMIC best checkpoint
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                bad_epochs = 0
                best_payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "best_val_miou": avg_val_miou,
                    "k_shot": k_shot,
                    "q_queries": q_queries,
                    "embed_dim": embed_dim,
                    "config": recovery_config,
                    "timestamp": datetime.now().isoformat(),
                }
                save_atomic_checkpoint(best_payload, best_ckpt_path)
                print(f"  [Checkpoint] *** NEW BEST VALIDATION DICE: {best_val_dice:.4f} *** Saved: {best_ckpt_path}", flush=True)
            else:
                bad_epochs += 1
                print(f"  [EarlyStopping] No improvement ({bad_epochs}/{patience} patience). Best remains: {best_val_dice:.4f}", flush=True)
                if bad_epochs >= patience:
                    print(f"\n[EarlyStopping] Early stopping triggered after {patience} epochs without improvement.", flush=True)
                    break

    finally:
        prevent_windows_sleep(False)

    total_time_min = (time.time() - t_continuation_start) / 60
    final_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if device.type == "cuda" else 0.0

    print("\n" + "=" * 70, flush=True)
    print(f"RECOVERY TRAINING SESSION COMPLETED IN {total_time_min:.1f} MINUTES", flush=True)
    print(f"Completed Total Epochs: {len(history_records)} / {target_final_epoch}", flush=True)
    print(f"Overall Best Validation Dice: {best_val_dice:.4f}", flush=True)
    print(f"Peak VRAM: {final_vram:.2f} MB / {total_vram:.2f} GB", flush=True)
    print(f"Best Checkpoint: {best_ckpt_path}", flush=True)
    print(f"Last Checkpoint: {last_ckpt_path}", flush=True)
    print("=" * 70, flush=True)

    summary = {
        "status": "COMPLETED",
        "recovery_run_dir": str(recovery_run_dir),
        "total_epochs": len(history_records),
        "resumed_epoch": start_epoch,
        "final_epoch": history_records[-1]["epoch"],
        "best_val_dice": best_val_dice,
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "history_file": str(rec_history_path),
        "continuation_duration_minutes": round(total_time_min, 2),
        "peak_vram_mb": round(final_vram, 2),
    }
    with open(results_dir / "recovery_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Few-Shot ProtoNet Recovery Trainer")
    parser.add_argument("--orig-dir", type=str, default="outputs/runs/objective1_run_001")
    parser.add_argument("--rec-dir", type=str, default="outputs/runs/objective1_run_002_recovery")
    parser.add_argument("--final-epoch", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=15)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-queries", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_recovery_training(
        original_run_dir=ROOT / args.orig_dir,
        recovery_run_dir=ROOT / args.rec_dir,
        target_final_epoch=args.final_epoch,
        episodes_per_epoch=args.episodes,
        val_episodes=args.val_episodes,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
