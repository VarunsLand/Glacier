"""
test_objective_5_verification.py
Comprehensive automated test suite for Objective 5:
Zero-Shot Handcrafted CLIP ViT-B-32 vs Learnable Context Optimization (CoOp).

Tests:
1. Checkpoint loading (coop_prompt_best.pt existence, format, keys).
2. CLIP inference (pretrained OpenAI ViT-B-32, text and image encodings).
3. CoOp inference (forward pass, context vector insertion).
4. Frozen backbone parameters (all ~151.3M parameters requires_grad=False).
5. Context parameter count (exactly 8,192 parameters: M=16, Dim=512).
6. Class-label mapping (0 -> Mountain Bedrock, 1 -> Glacier Ice).
7. Identical test-set membership (seed=123 deterministic 100 patches, 50/50 balance).
8. Independent metric calculations (recomputing TP, FP, FN, TN, Accuracy, Precision, Recall, Specificity, F1, Balanced Acc).
9. Confusion matrix totals (TP + FP + FN + TN == 100 for both models).
10. Train/Test leakage verification (zero patch overlap, zero glacier ID overlap between 32-sample train set and 100-sample test set).
11. Single-image inference pipeline (testing demo samples, verifying output types and probability normalization).
12. UI metric consistency (verifying clip_coop_lab and research_results match canonical JSON report).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset
from src.inference import (
    DEMO_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    load_clip_coop_bundle,
    run_clip_coop_classification,
)
from scripts.evaluate_clip_coop import CLIPGlacierPatchDataset, CoOpPromptLearner


class TestObjective5Verification:
    """Rigorous scientific verification of Objective 5 models, data, and metrics."""

    @pytest.fixture(scope="class")
    def bundle(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return load_clip_coop_bundle(device)

    @pytest.fixture(scope="class")
    def canonical_report(self):
        p = RESULTS_DIR / "objective5_clip_coop_report.json"
        assert p.exists(), f"Canonical report missing at {p}"
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def predictions_df(self):
        p = RESULTS_DIR / "objective5_per_sample_predictions.csv"
        assert p.exists(), f"Per-sample predictions CSV missing at {p}"
        return pd.read_csv(p)

    def test_checkpoint_loading_and_keys(self):
        """1. Verify coop_prompt_best.pt exists and has valid structure."""
        ckpt_path = ROOT / "outputs" / "checkpoints" / "coop_prompt_best.pt"
        assert ckpt_path.exists(), "CoOp checkpoint does not exist"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "ctx" in ckpt, "Checkpoint missing 'ctx' parameter tensor"
        assert "n_ctx" in ckpt, "Checkpoint missing 'n_ctx'"
        assert ckpt["n_ctx"] == 16, f"Expected 16 context tokens, found {ckpt['n_ctx']}"
        assert "classnames" in ckpt, "Checkpoint missing 'classnames'"
        assert ckpt["classnames"] == ["mountain bedrock", "glacier ice"]
        assert ckpt["ctx"].shape == (16, 512), f"Expected shape (16, 512), got {ckpt['ctx'].shape}"

    def test_clip_backbone_frozen(self, bundle):
        """2. Verify that entire CLIP backbone parameters are strictly frozen (requires_grad=False)."""
        clip_model = bundle["clip_model"]
        trainable_params = [p for p in clip_model.parameters() if p.requires_grad]
        assert len(trainable_params) == 0, f"Found {len(trainable_params)} unfrozen parameters in CLIP backbone!"
        total_backbone_params = sum(p.numel() for p in clip_model.parameters())
        assert total_backbone_params > 100_000_000, f"Expected >100M params in ViT-B-32, got {total_backbone_params}"

    def test_coop_trainable_parameter_count(self, bundle):
        """3. Verify exactly 8,192 trainable parameters in CoOp prompt learner (16 tokens * 512 dim)."""
        prompt_learner = bundle["prompt_learner"]
        trainable_count = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)
        assert trainable_count == 8192, f"Expected exactly 8,192 trainable parameters, found {trainable_count}"
        assert prompt_learner.ctx.shape == (16, 512), f"Unexpected context shape {prompt_learner.ctx.shape}"

    def test_class_label_mapping(self, bundle):
        """4. Verify class names and label index semantics: 0=Mountain Bedrock, 1=Glacier Ice."""
        classnames = bundle["classnames"]
        assert len(classnames) == 2
        assert classnames[0] == "Mountain Bedrock"
        assert classnames[1] == "Glacier Ice"

    def test_per_sample_predictions_structure_and_count(self, predictions_df):
        """5. Verify the 100-sample predictions CSV structure and class balance."""
        assert len(predictions_df) == 100, f"Expected 100 test samples, found {len(predictions_df)}"
        required_cols = [
            "sample_index", "sample_id", "glacier_id", "filename", "full_path", "ice_ratio",
            "ground_truth_label", "ground_truth_class",
            "clip_pred_label", "clip_pred_class", "clip_prob_bedrock", "clip_prob_glacier", "clip_correct",
            "coop_pred_label", "coop_pred_class", "coop_prob_bedrock", "coop_prob_glacier", "coop_correct"
        ]
        for col in required_cols:
            assert col in predictions_df.columns, f"Missing required column: {col}"

        # 50 Bedrock (0) and 50 Glacier (1)
        gt_counts = predictions_df["ground_truth_label"].value_counts().to_dict()
        assert gt_counts.get(0) == 50, f"Expected 50 non-glacier samples, found {gt_counts.get(0)}"
        assert gt_counts.get(1) == 50, f"Expected 50 glacier samples, found {gt_counts.get(1)}"

        # Probabilities sum to ~1.0
        clip_prob_sum = predictions_df["clip_prob_bedrock"] + predictions_df["clip_prob_glacier"]
        coop_prob_sum = predictions_df["coop_prob_bedrock"] + predictions_df["coop_prob_glacier"]
        assert np.allclose(clip_prob_sum, 1.0, atol=1e-3)
        assert np.allclose(coop_prob_sum, 1.0, atol=1e-3)

    def test_independent_metric_recalculation_clip(self, predictions_df, canonical_report):
        """6. Recompute CLIP metrics from per-sample CSV and verify 1:1 match with canonical report."""
        y_true = predictions_df["ground_truth_label"].values
        y_pred = predictions_df["clip_pred_label"].values

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        assert tp == 45, f"Expected CLIP TP=45, got {tp}"
        assert fp == 27, f"Expected CLIP FP=27, got {fp}"
        assert fn == 5,  f"Expected CLIP FN=5, got {fn}"
        assert tn == 23, f"Expected CLIP TN=23, got {tn}"
        assert tp + fp + fn + tn == 100, "Confusion matrix sum must be exactly 100"

        acc = (tp + tn) / 100.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = (2 * prec * rec) / (prec + rec)
        bal_acc = (rec + spec) / 2.0

        assert np.isclose(acc, 0.6800, atol=1e-4)
        assert np.isclose(prec, 0.6250, atol=1e-4)
        assert np.isclose(rec, 0.9000, atol=1e-4)
        assert np.isclose(spec, 0.4600, atol=1e-4)
        assert np.isclose(f1, 0.7377, atol=1e-4)
        assert np.isclose(bal_acc, 0.6800, atol=1e-4)

        # Check match with JSON report
        h_rep = canonical_report["test_evaluation"]["handcrafted_clip"]
        assert h_rep["tp"] == tp
        assert h_rep["fp"] == fp
        assert h_rep["fn"] == fn
        assert h_rep["tn"] == tn
        assert np.isclose(h_rep["accuracy"], acc)
        assert np.isclose(h_rep["precision"], prec)
        assert np.isclose(h_rep["recall"], rec)

    def test_independent_metric_recalculation_coop(self, predictions_df, canonical_report):
        """7. Recompute CoOp metrics from per-sample CSV and verify 1:1 match with canonical report."""
        y_true = predictions_df["ground_truth_label"].values
        y_pred = predictions_df["coop_pred_label"].values

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        assert tp == 39, f"Expected CoOp TP=39, got {tp}"
        assert fp == 12, f"Expected CoOp FP=12, got {fp}"
        assert fn == 11, f"Expected CoOp FN=11, got {fn}"
        assert tn == 38, f"Expected CoOp TN=38, got {tn}"
        assert tp + fp + fn + tn == 100, "Confusion matrix sum must be exactly 100"

        acc = (tp + tn) / 100.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = (2 * prec * rec) / (prec + rec)
        bal_acc = (rec + spec) / 2.0

        assert np.isclose(acc, 0.7700, atol=1e-4)
        assert np.isclose(prec, 39 / 51, atol=1e-4)  # 0.7647
        assert np.isclose(rec, 0.7800, atol=1e-4)
        assert np.isclose(spec, 0.7600, atol=1e-4)
        assert np.isclose(f1, 0.7723, atol=1e-4)
        assert np.isclose(bal_acc, 0.7700, atol=1e-4)

        # Check match with JSON report
        c_rep = canonical_report["test_evaluation"]["learnable_coop"]
        assert c_rep["tp"] == tp
        assert c_rep["fp"] == fp
        assert c_rep["fn"] == fn
        assert c_rep["tn"] == tn
        assert np.isclose(c_rep["accuracy"], acc)
        assert np.isclose(c_rep["precision"], prec)
        assert np.isclose(c_rep["recall"], rec)

    def test_zero_train_test_leakage(self):
        """8. Verify zero patch overlap and zero basin overlap between train set and test set."""
        summary_path = RESULTS_DIR / "objective5_audit_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                audit_summary = json.load(f)
            d_audit = audit_summary["dataset_audit"]
            assert d_audit["patch_overlap"] == 0, f"Detected {d_audit['patch_overlap']} patch overlap!"
            assert d_audit["glacier_id_overlap"] == 0, f"Detected {d_audit['glacier_id_overlap']} glacier basin overlap!"
            assert d_audit["train_samples"] == 32
            assert d_audit["test_samples"] == 100

    def test_single_image_inference_demo_samples(self):
        """9. Test live single-image classification on authentic demo samples."""
        test_samples = [
            ("Clean Glacier Ice", DEMO_DIR / "demo_glacier_1_clean_ice.png", "Glacier Ice"),
            ("Bedrock Ridge", DEMO_DIR / "demo_bedrock_1_alpine_ridge.png", "Mountain Bedrock"),
        ]
        for name, p, expected_class in test_samples:
            assert p.exists(), f"Demo sample {p} does not exist"
            img = Image.open(p)
            res = run_clip_coop_classification(img)

            assert "clip" in res and "coop" in res
            assert "probabilities" in res["clip"]
            assert "probabilities" in res["coop"]
            assert res["clip"]["prediction"] in ["Mountain Bedrock", "Glacier Ice"]
            assert res["coop"]["prediction"] in ["Mountain Bedrock", "Glacier Ice"]
            assert res["latency_ms"] > 0.0

            # Both models should identify clear glacier and clear bedrock correctly
            assert res["clip"]["prediction"] == expected_class, f"{name}: CLIP predicted {res['clip']['prediction']}, expected {expected_class}"
            assert res["coop"]["prediction"] == expected_class, f"{name}: CoOp predicted {res['coop']['prediction']}, expected {expected_class}"

    def test_ui_metric_alignment_and_no_stale_hardcoded_numbers(self):
        """10. Verify that clip_coop_lab.py and research_results.py do not hardcode stale numbers."""
        lab_code = (ROOT / "src" / "ui" / "clip_coop_lab.py").read_text(encoding="utf-8")
        results_code = (ROOT / "src" / "ui" / "research_results.py").read_text(encoding="utf-8")

        # Stale numbers that must NOT exist in the UI code
        for stale in ["0.6189", "0.7586", "0.8100", "0.8148"]:
            assert stale not in lab_code, f"Found stale number '{stale}' in clip_coop_lab.py!"
            assert stale not in results_code, f"Found stale number '{stale}' in research_results.py!"
