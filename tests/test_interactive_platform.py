"""Comprehensive Automated Verification for GlacierVision Interactive Platform Engine.

Tests:
1. Model loading: U-Net (13 bands), ProtoNet (13 bands), CLIP ViT-B-32, CoOp prompt learner.
2. NetCDF ingestion & validation: 13 Sentinel-2 channels, RGB & SWIR generation, NDSI index.
3. Live segmentation execution: U-Net and Few-Shot ProtoNet (K-shot support set from fold_train).
4. Real-time metrics calculation: Dice, IoU, Precision, Recall, Accuracy, and 4-color Error Maps.
5. Live Vision-Language classification: Zero-Shot CLIP and CoOp prompt tuning on test RGB images.
6. Dynamic GLIMS retreat and area budget accounting: alias resolution and parameter simulation.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.inference import (
    ROOT,
    load_unet_model,
    load_protonet_model,
    load_clip_coop_bundle,
    read_and_validate_netcdf,
    run_unet_inference,
    run_protonet_inference,
    run_clip_coop_classification,
    compute_segmentation_metrics,
    generate_contour_overlay,
    generate_error_classification_map,
    get_glacier_dynamic_assessment,
    get_demo_netcdf_candidates,
)


def test_models_load_successfully():
    """Verify all 3 core deep learning architectures initialize and load checkpoints."""
    unet = load_unet_model()
    assert unet is not None
    assert isinstance(unet, torch.nn.Module)

    protonet = load_protonet_model()
    assert protonet is not None
    assert isinstance(protonet, torch.nn.Module)

    bundle = load_clip_coop_bundle()
    assert "clip_model" in bundle
    assert "handcrafted_text_feats" in bundle
    assert "coop_text_feats" in bundle
    assert bundle["handcrafted_text_feats"].shape == (2, 512)
    assert bundle["coop_text_feats"].shape == (2, 512)


def test_netcdf_validation_and_reading():
    """Verify NetCDF validation, channel extraction, RGB/SWIR synthesis, and NDSI calculation."""
    candidates = get_demo_netcdf_candidates()
    assert len(candidates) > 0, "No demo NetCDF patches available"

    sample_path = candidates[0]["path"]
    data = read_and_validate_netcdf(sample_path)

    assert "raw_bands" in data
    assert data["raw_bands"].shape == (13, 256, 256)
    assert data["raw_bands"].dtype == np.float32

    assert "rgb_preview" in data
    assert data["rgb_preview"].shape == (256, 256, 3)
    assert data["rgb_preview"].dtype == np.uint8

    assert "swir_preview" in data
    assert data["swir_preview"].shape == (256, 256, 3)
    assert data["swir_preview"].dtype == np.uint8

    assert "ndsi" in data
    assert data["ndsi"].shape == (256, 256)
    assert data["ndsi"].min() >= -1.0
    assert data["ndsi"].max() <= 1.0

    meta = data["metadata"]
    assert meta["channels"] == 13
    assert meta["dimensions"] == "256x256"
    assert meta["has_ground_truth"] is True
    assert meta["glacier_pixel_ratio"] is not None


def test_live_unet_inference():
    """Verify real-time U-Net inference outputs valid probabilities and binary mask."""
    candidates = get_demo_netcdf_candidates()
    sample = read_and_validate_netcdf(candidates[0]["path"])

    probs, mask, latency = run_unet_inference(sample["raw_bands"])

    assert probs.shape == (256, 256)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)).issubset({0, 1})
    assert latency > 0.0


def test_live_protonet_inference():
    """Verify real-time ProtoNet few-shot inference outputs valid probabilities and binary mask."""
    candidates = get_demo_netcdf_candidates()
    sample = read_and_validate_netcdf(candidates[0]["path"])

    probs, mask, latency = run_protonet_inference(sample["raw_bands"], k_shots=5)

    assert probs.shape == (256, 256)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)).issubset({0, 1})
    assert latency > 0.0


def test_segmentation_metrics_and_error_map():
    """Verify calculation of segmentation metrics and generation of visual overlays."""
    candidates = get_demo_netcdf_candidates()
    sample = read_and_validate_netcdf(candidates[0]["path"])
    gt_mask = sample["gt_mask"]
    assert gt_mask is not None

    _, pred_mask, _ = run_unet_inference(sample["raw_bands"])

    metrics = compute_segmentation_metrics(pred_mask, gt_mask)
    for m in ["dice", "iou", "precision", "recall", "accuracy"]:
        assert m in metrics
        assert 0.0 <= metrics[m] <= 1.0

    # Test error map
    error_map = generate_error_classification_map(pred_mask, gt_mask)
    assert error_map.shape == (256, 256, 3)
    assert error_map.dtype == np.uint8

    # Test contour overlay
    overlay = generate_contour_overlay(sample["rgb_preview"], pred_mask)
    assert overlay.shape == (256, 256, 3)
    assert overlay.dtype == np.uint8


def test_live_clip_coop_classification():
    """Verify live Zero-Shot CLIP and Learnable CoOp classification on demo RGB images."""
    demo_glacier = ROOT / "outputs" / "demo_samples" / "demo_glacier_1_clean_ice.png"
    demo_bedrock = ROOT / "outputs" / "demo_samples" / "demo_bedrock_1_alpine_ridge.png"

    if demo_glacier.exists():
        img_g = Image.open(demo_glacier)
        res_g = run_clip_coop_classification(img_g)
        assert res_g["clip"]["prediction"] in ["Glacier Ice", "Mountain Bedrock"]
        assert res_g["coop"]["prediction"] in ["Glacier Ice", "Mountain Bedrock"]
        assert res_g["coop"]["prediction"] == "Glacier Ice"  # Strong glacier signal
        assert abs(sum(res_g["clip"]["probabilities"].values()) - 1.0) < 0.01
        assert abs(sum(res_g["coop"]["probabilities"].values()) - 1.0) < 0.01

    if demo_bedrock.exists():
        img_b = Image.open(demo_bedrock)
        res_b = run_clip_coop_classification(img_b)
        assert res_b["clip"]["prediction"] in ["Glacier Ice", "Mountain Bedrock"]
        assert res_b["coop"]["prediction"] in ["Glacier Ice", "Mountain Bedrock"]
        assert res_b["coop"]["prediction"] == "Mountain Bedrock"  # Non-glacier bedrock signal


def test_dynamic_glacier_retreat_assessment():
    """Verify dynamic retreat calculation, alias mapping, and scenario simulation."""
    # 1. Alias lookup: Rhone
    res_rhone = get_glacier_dynamic_assessment("Rhone")
    assert res_rhone["glacier_id"] == "G008379E46648N"
    assert res_rhone["area_change_km2"] < 0  # Retreating
    assert res_rhone["dynamics_class"] in ["Significant Retreat", "Moderate Retreat"]

    # 2. Alias lookup: Great Aletsch
    res_aletsch = get_glacier_dynamic_assessment("Great Aletsch")
    assert res_aletsch["glacier_id"] == "G008032E46504N"
    assert res_aletsch["area_change_km2"] < 0

    # 3. Custom dynamic override scenario
    res_sim = get_glacier_dynamic_assessment("Rhone", override_area_earlier=100.0, override_area_later=80.0, override_years=10.0)
    assert res_sim["earlier_area_km2"] == 100.0
    assert res_sim["later_area_km2"] == 80.0
    assert res_sim["area_change_km2"] == -20.0
    assert res_sim["percent_change"] == -20.0
    assert res_sim["annual_rate_km2_yr"] == -2.0
    assert res_sim["dynamics_class"] == "Significant Retreat"


def test_uploaded_netcdf_bytesio_and_unet_inference():
    """Verify that in-memory BytesIO NetCDF uploads (e.g. glacier_test.nc) are successfully validated and evaluated."""
    import io
    test_nc = Path("C:/Users/Varun's LOQ/Desktop/glacier_test.nc")
    if not test_nc.exists():
        pytest.skip("glacier_test.nc not present on Desktop")

    # Simulate in-memory Streamlit UploadedFile
    with open(test_nc, "rb") as f:
        raw_bytes = f.read()

    bio = io.BytesIO(raw_bytes)
    data = read_and_validate_netcdf(bio)

    assert data["raw_bands"].shape == (13, 256, 256)
    assert data["raw_bands"].dtype == np.float32
    assert data["gt_mask"] is not None
    assert data["gt_mask"].shape == (256, 256)
    assert data["rgb_preview"].shape == (256, 256, 3)
    assert data["swir_preview"].shape == (256, 256, 3)
    assert data["ndsi"].shape == (256, 256)

    # Test live U-Net inference on the uploaded bands
    probs, mask, latency = run_unet_inference(data["raw_bands"])
    assert probs.shape == (256, 256)
    assert mask.shape == (256, 256)
    assert latency > 0.0

    # Compute metrics
    metrics = compute_segmentation_metrics(mask, data["gt_mask"])
    assert metrics["dice"] > 0.50
    assert metrics["accuracy"] > 0.70


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
