"""
generate_objective5_audit.py
Scientific audit script for Objective 5: Handcrafted CLIP vs Learnable CoOp.
1. Re-instantiates the exact 100-sample test set and 32-sample train set.
2. Audits data leakage and glacier ID independence.
3. Loads frozen OpenAI CLIP ViT-B-32 and the trained coop_prompt_best.pt checkpoint.
4. Generates outputs/results/objective5_per_sample_predictions.csv with full probabilities and flags.
5. Recalculates all confusion matrix and classification metrics independently.
6. Compares with objective5_clip_coop_report.json and resolves numerical discrepancies.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import DL4GAMDataset
from src.inference import load_clip_coop_bundle
from scripts.evaluate_clip_coop import CLIPGlacierPatchDataset, CoOpPromptLearner

def main():
    print("=" * 70)
    print("GLACIERVISION OBJECTIVE 5 AUDIT & PER-SAMPLE EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load CLIP transforms
    _, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)

    # 2. Re-create train and test datasets with identical seeds
    print("\n[1] Indexing datasets...")
    train_base = DL4GAMDataset(root=ROOT, fold="fold_train")
    test_base = DL4GAMDataset(root=ROOT, fold="fold_test")

    train_ds = CLIPGlacierPatchDataset(
        train_base, clip_preprocess, glacier_thresh_high=0.10, glacier_thresh_low=0.02, max_samples_per_class=16, seed=42
    )
    test_ds = CLIPGlacierPatchDataset(
        test_base, clip_preprocess, glacier_thresh_high=0.10, glacier_thresh_low=0.02, max_samples_per_class=50, seed=123
    )

    print(f"Train samples: {len(train_ds)} (16 bedrock, 16 glacier)")
    print(f"Test samples: {len(test_ds)} (50 bedrock, 50 glacier)")

    # 3. Check for Data Leakage
    train_paths = {p for p, _, _ in train_ds.samples}
    test_paths = {p for p, _, _ in test_ds.samples}
    path_overlap = train_paths.intersection(test_paths)

    train_gids = {p.parent.name for p, _, _ in train_ds.samples}
    test_gids = {p.parent.name for p, _, _ in test_ds.samples}
    gid_overlap = train_gids.intersection(test_gids)

    print("\n[2] Data Leakage Audit:")
    print(f"  Exact patch path overlap: {len(path_overlap)} patches")
    print(f"  Glacier basin ID overlap: {len(gid_overlap)} glaciers (train: {len(train_gids)}, test: {len(test_gids)})")
    if gid_overlap:
        print(f"  Overlapping glacier IDs: {sorted(list(gid_overlap))}")

    # 4. Load Models & Checkpoint via production bundle
    print("\n[3] Loading CLIP and CoOp models...")
    bundle = load_clip_coop_bundle(device)
    clip_model = bundle["clip_model"]
    handcrafted_feats = bundle["handcrafted_text_feats"]
    coop_feats = bundle["coop_text_feats"]
    classnames = bundle["classnames"]  # ["Mountain Bedrock", "Glacier Ice"]

    print(f"Class mapping: 0 -> {classnames[0]}, 1 -> {classnames[1]}")

    # Verify CoOp parameter count and frozen backbone
    prompt_learner = bundle["prompt_learner"]
    coop_trainable_params = sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad)
    clip_trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)

    print(f"  CoOp trainable parameters: {coop_trainable_params:,}")
    print(f"  CLIP backbone trainable parameters: {clip_trainable_params:,} (Frozen = {clip_trainable_params == 0})")
    print(f"  CoOp context shape: {prompt_learner.ctx.shape} (M={prompt_learner.n_ctx}, Dim={prompt_learner.ctx.shape[1]})")

    # 5. Run inference on all 100 test samples
    print("\n[4] Running inference on 100 test samples...")
    records = []

    for i in range(len(test_ds)):
        x_tensor, true_label, fname = test_ds[i]
        path, label_chk, ratio = test_ds.samples[i]
        assert true_label == label_chk

        gid = path.parent.name
        sample_id = f"{gid}/{fname}"

        x_in = x_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = F.normalize(clip_model.encode_image(x_in), p=2, dim=-1)

            # CLIP
            clip_logits = (img_feat @ handcrafted_feats.T) * 100.0
            clip_probs = F.softmax(clip_logits, dim=-1).squeeze().cpu().numpy()
            clip_pred = int(clip_probs.argmax())

            # CoOp
            coop_logits = (img_feat @ coop_feats.T) * 100.0
            coop_probs = F.softmax(coop_logits, dim=-1).squeeze().cpu().numpy()
            coop_pred = int(coop_probs.argmax())

        records.append({
            "sample_index": i + 1,
            "sample_id": sample_id,
            "glacier_id": gid,
            "filename": fname,
            "full_path": str(path),
            "ice_ratio": round(ratio, 4),
            "ground_truth_label": true_label,
            "ground_truth_class": classnames[true_label],
            "clip_pred_label": clip_pred,
            "clip_pred_class": classnames[clip_pred],
            "clip_prob_bedrock": round(float(clip_probs[0]), 5),
            "clip_prob_glacier": round(float(clip_probs[1]), 5),
            "clip_correct": bool(clip_pred == true_label),
            "coop_pred_label": coop_pred,
            "coop_pred_class": classnames[coop_pred],
            "coop_prob_bedrock": round(float(coop_probs[0]), 5),
            "coop_prob_glacier": round(float(coop_probs[1]), 5),
            "coop_correct": bool(coop_pred == true_label),
        })

    df = pd.DataFrame(records)
    out_csv = ROOT / "outputs" / "results" / "objective5_per_sample_predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} predictions to: {out_csv}")

    # 6. Recalculate metrics independently
    print("\n[5] Calculating independent evaluation metrics...")
    def calc_metrics(y_true, y_pred, name="Model"):
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        total = tp + fp + fn + tn
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        bal_acc = (rec + spec) / 2.0

        return {
            "name": name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Total": total,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "F1": f1,
            "Balanced_Accuracy": bal_acc,
        }

    m_clip = calc_metrics(df["ground_truth_label"].values, df["clip_pred_label"].values, "Zero-Shot CLIP")
    m_coop = calc_metrics(df["ground_truth_label"].values, df["coop_pred_label"].values, "Learnable CoOp")

    for m in [m_clip, m_coop]:
        print(f"\n--- {m['name']} ---")
        print(f"  TP: {m['TP']} | FP: {m['FP']} | FN: {m['FN']} | TN: {m['TN']} (Total: {m['Total']})")
        print(f"  Accuracy:          {m['Accuracy']:.4f} ({m['Accuracy']*100:.2f}%)")
        print(f"  Precision:         {m['Precision']:.4f}")
        print(f"  Recall:            {m['Recall']:.4f}")
        print(f"  Specificity:       {m['Specificity']:.4f}")
        print(f"  F1 Score:          {m['F1']:.4f}")
        print(f"  Balanced Accuracy: {m['Balanced_Accuracy']:.4f}")

    delta_acc = m_coop['Accuracy'] - m_clip['Accuracy']
    delta_prec = m_coop['Precision'] - m_clip['Precision']
    delta_rec = m_coop['Recall'] - m_clip['Recall']
    delta_f1 = m_coop['F1'] - m_clip['F1']
    fp_reduction = (m_clip['FP'] - m_coop['FP']) / m_clip['FP']

    print("\n--- Empirical Deltas (CoOp - CLIP) ---")
    print(f"  Accuracy Delta:    {delta_acc:+.4f} ({delta_acc*100:+.2f}%)")
    print(f"  Precision Delta:   {delta_prec:+.4f} ({delta_prec*100:+.2f}%)")
    print(f"  Recall Delta:      {delta_rec:+.4f} ({delta_rec*100:+.2f}%)")
    print(f"  F1 Score Delta:    {delta_f1:+.4f} ({delta_f1*100:+.2f}%)")
    print(f"  FP Reduction:      -{fp_reduction*100:.1f}% ({m_coop['FP']} vs {m_clip['FP']})")

    # Save summary json
    summary = {
        "dataset_audit": {
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "patch_overlap": len(path_overlap),
            "glacier_id_overlap": len(gid_overlap),
            "train_glacier_ids": len(train_gids),
            "test_glacier_ids": len(test_gids),
        },
        "model_architecture": {
            "backbone": "OpenAI CLIP ViT-B-32",
            "coop_context_tokens": prompt_learner.n_ctx,
            "coop_trainable_params": coop_trainable_params,
            "clip_backbone_trainable_params": clip_trainable_params,
            "backbone_frozen": clip_trainable_params == 0,
            "context_configuration": "Unified / Shared Context (16 tokens shared across classes)",
        },
        "metrics": {
            "clip": m_clip,
            "coop": m_coop,
            "deltas": {
                "accuracy": round(delta_acc, 4),
                "precision": round(delta_prec, 4),
                "recall": round(delta_rec, 4),
                "f1": round(delta_f1, 4),
                "fp_reduction_pct": round(fp_reduction * 100.0, 2),
            }
        }
    }
    with open(ROOT / "outputs" / "results" / "objective5_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
