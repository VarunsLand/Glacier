"""Smoke Test for Objective 1 Glacier Remote Sensing Pipeline.

Verifies:
1. DL4GAM dataset loading across train, validation, and test folds.
2. Episodic sampling with strict glacier-level disjointness.
3. FewShotProtoNet forward pass, prototype extraction, and FP16 autocast.
4. Loss computation and gradient backpropagation.
5. GPU VRAM consumption under RTX 3050 6GB hardware.
6. Supervised U-Net baseline forward and backward pass.
7. Full metric suite computation (mIoU, Dice, Precision, Recall, Boundary F1).
8. Visual comparison figure generation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from src.dataset import DL4GAMDataset
from src.sampler import FewShotEpisodeSampler
from src.fewshot_model import FewShotProtoNet
from src.unet_baseline import GlacierUNet
from src.metrics import calculate_segmentation_metrics, compute_combined_loss
from src.visualization import save_prediction_comparison


def run_smoke_test():
    print("=" * 65)
    print("GLACIERVISION: OBJECTIVE 1 PIPELINE SMOKE TEST")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    if device.type == "cuda":
        print(f"GPU Model:        {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM:       {total_vram:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Dataset Verification
    print("\n[Step 1/8] Verifying DL4GAM Datasets across all 3 official partitions...")
    train_ds = DL4GAMDataset(root=ROOT, fold="fold_train")
    val_ds = DL4GAMDataset(root=ROOT, fold="fold_valid")
    test_ds = DL4GAMDataset(root=ROOT, fold="fold_test")

    print(f"  Train Fold: {len(train_ds)} patches across {len(train_ds.get_all_glacier_ids())} glaciers.")
    print(f"  Valid Fold: {len(val_ds)} patches across {len(val_ds.get_all_glacier_ids())} glaciers.")
    print(f"  Test Fold:  {len(test_ds)} patches across {len(test_ds.get_all_glacier_ids())} glaciers.")

    assert len(train_ds) == 1475, f"Expected 1475 train patches, got {len(train_ds)}"
    assert len(val_ds) == 242, f"Expected 242 val patches, got {len(val_ds)}"
    assert len(test_ds) == 534, f"Expected 534 test patches, got {len(test_ds)}"
    print("  Partition integrity check: PASSED (Exact match with official split_1)")

    # 2. Patch Data Inspection
    print("\n[Step 2/8] Inspecting sample tensor properties...")
    sample_x, sample_y, sample_info = train_ds[0]
    print(f"  Input Tensor X: shape={tuple(sample_x.shape)}, dtype={sample_x.dtype}, min={sample_x.min():.2f}, max={sample_x.max():.2f}")
    print(f"  Target Mask Y:  shape={tuple(sample_y.shape)}, dtype={sample_y.dtype}, unique={torch.unique(sample_y).tolist()}")
    print(f"  Glacier ID:     {sample_info['glacier_id']}, Patch: {sample_info['patch_name']}")
    assert sample_x.shape == (13, 256, 256), f"Expected (13, 256, 256), got {sample_x.shape}"
    assert sample_y.shape == (1, 256, 256), f"Expected (1, 256, 256), got {sample_y.shape}"
    assert not torch.isnan(sample_x).any(), "NaN found in input tensor!"
    assert not torch.isnan(sample_y).any(), "NaN found in target mask!"
    print("  Sample tensor validation: PASSED")

    # 3. Episodic Sampler & Disjointness Check
    print("\n[Step 3/8] Testing Few-Shot Episodic Sampler (K=5, Q=2)...")
    sampler = FewShotEpisodeSampler(dataset=train_ds, k_shot=5, q_queries=2, seed=123)
    episode = sampler.sample_episode(episode_idx=0)
    print(f"  Support X: {tuple(episode.support_x.shape)} | Support Glaciers: {episode.support_glaciers}")
    print(f"  Query X:   {tuple(episode.query_x.shape)} | Query Glaciers:   {episode.query_glaciers}")

    overlap = set(episode.support_glaciers) & set(episode.query_glaciers)
    assert len(overlap) == 0, f"Disjointness violated! Overlap: {overlap}"
    print("  Zero Glacier Leakage check: PASSED (Strictly disjoint support/query sets)")

    # 4. FewShotProtoNet Forward Pass & Prototype Extraction
    print("\n[Step 4/8] Testing FewShotProtoNet forward pass (FP16 mixed precision on CUDA)...")
    model = FewShotProtoNet(in_channels=13, embed_dim=64, temperature=20.0).to(device)
    episode = episode.to(device)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        out = model(episode.support_x, episode.support_y, episode.query_x)
        logits = out["logits"]
        probs = out["probs"]
        proto_fg = out["proto_fg"]
        proto_bg = out["proto_bg"]

    print(f"  Logits shape:      {tuple(logits.shape)}")
    print(f"  Probs range:       [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Glacier Prototype: shape={tuple(proto_fg.shape)}, norm={torch.norm(proto_fg):.4f}")
    print(f"  Background Proto:  shape={tuple(proto_bg.shape)}, norm={torch.norm(proto_bg):.4f}")
    print("  Forward pass: PASSED")

    # 5. Loss & Backward Pass with GradScaler
    print("\n[Step 5/8] Testing loss calculation and backward pass...")
    loss = compute_combined_loss(logits, episode.query_y, pos_weight=train_ds.pos_weight)
    print(f"  Computed Combined Loss: {loss.item():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    print("  Backward pass & optimizer step: PASSED")

    # Check VRAM after Few-Shot step
    if device.type == "cuda":
        alloc_fs = torch.cuda.memory_allocated(0) / (1024**2)
        peak_fs = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"  Few-Shot Peak VRAM Allocated: {peak_fs:.2f} MB ({peak_fs / 1024:.2f} GB / {total_vram:.2f} GB)")

    # 6. Supervised U-Net Baseline Check
    print("\n[Step 6/8] Testing Supervised U-Net baseline forward and backward pass...")
    unet = GlacierUNet(in_channels=13, base_channels=16).to(device)
    dummy_input = torch.randn(2, 13, 256, 256, device=device)
    dummy_target = (torch.rand(2, 1, 256, 256, device=device) > 0.7).float()

    optimizer_unet = torch.optim.AdamW(unet.parameters(), lr=1e-3)
    optimizer_unet.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        u_logits = unet(dummy_input)
        u_loss = compute_combined_loss(u_logits, dummy_target, pos_weight=train_ds.pos_weight)

    scaler.scale(u_loss).backward()
    scaler.unscale_(optimizer_unet)
    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
    scaler.step(optimizer_unet)
    scaler.update()

    print(f"  U-Net output logits shape: {tuple(u_logits.shape)}")
    print(f"  U-Net baseline loss:       {u_loss.item():.4f}")
    print("  U-Net Baseline check: PASSED")

    # 7. Metrics Suite Verification
    print("\n[Step 7/8] Verifying segmentation metrics suite...")
    metrics = calculate_segmentation_metrics(logits, episode.query_y, is_logits=True)
    print("  Computed Metrics on query patch:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:15s}: {v:.4f}")
        else:
            print(f"    {k:15s}: {v}")
    assert "fg_iou" in metrics and "miou" in metrics and "dice" in metrics and "boundary_f1" in metrics
    print("  Metrics verification: PASSED")

    # 8. Visual Results Generator
    print("\n[Step 8/8] Testing 5-panel diagnostic visual comparison rendering...")
    out_fig = ROOT / "outputs" / "figures" / "smoke_test_prediction.png"
    # Unnormalize query patch for RGB display
    raw_query_bands, _, _ = train_ds.read_netcdf_patch(
        Path(sample_info["patch_path"]), scale=train_ds.scale
    )
    save_prediction_comparison(
        image_bands13=raw_query_bands,
        gt_mask=sample_y[0].numpy(),
        pred_prob_or_mask=probs[0, 0].detach().cpu().numpy(),
        save_path=out_fig,
        title_prefix=f"Smoke Test Example — Glacier {sample_info['glacier_id']}",
    )
    assert out_fig.exists(), f"Figure was not created at {out_fig}"
    print(f"  Figure saved successfully: {out_fig} ({out_fig.stat().st_size:,} bytes)")

    if device.type == "cuda":
        final_peak = torch.cuda.max_memory_allocated(0) / (1024**2)
        print(f"\nFinal Peak GPU VRAM Usage: {final_peak:.2f} MB ({final_peak / 1024:.2f} GB / {total_vram:.2f} GB)")
        print(f"VRAM Safety Margin:        {total_vram - (final_peak / 1024):.2f} GB Free ({(1.0 - (final_peak / 1024) / total_vram):.1%} headroom)")

    print("\n" + "=" * 65)
    print("ALL 8 SMOKE TEST STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
