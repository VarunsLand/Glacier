"""Objective 5: Handcrafted Pretrained CLIP vs Learnable CoOp Prompt Tuning.

Scientific Implementation:
1. Loads genuine OpenAI-pretrained CLIP ViT-B-32 model via open_clip.
2. Formulates handcrafted natural language prompts for glacier vs bedrock classification.
3. Implements Context Optimization (CoOp, Zhou et al. 2022) with M=16 learnable continuous context vectors.
4. Uses DL4GAM Alps Sentinel-2 RGB representations (B4/B3/B2) with transparent glacier-presence thresholds.
5. Strictly separates training glaciers from held-out test glaciers.
6. Evaluates both methods on identical test patches, reporting Accuracy, Precision, Recall, F1, and Confusion Matrices.
7. Saves outputs to outputs/results/objective5_clip_coop_report.json and outputs/figures/clip_vs_coop_comparison.png.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import DL4GAMDataset

OUT_RESULTS = ROOT / "outputs" / "results"
OUT_CKPTS = ROOT / "outputs" / "checkpoints"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_CKPTS.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


# =====================================================================
# DATASET WRAPPER FOR CLIP VISUAL INPUTS
# =====================================================================
class CLIPGlacierPatchDataset:
    """Prepares DL4GAM Sentinel-2 RGB patches and binary classification labels."""

    def __init__(
        self,
        base_dataset: DL4GAMDataset,
        clip_preprocess,
        glacier_thresh_high: float = 0.10,  # >=10% glacier pixels -> Class 1 (Glacier)
        glacier_thresh_low: float = 0.02,   # <2% glacier pixels  -> Class 0 (Non-Glacier)
        max_samples_per_class: int = 150,
        seed: int = 42,
    ):
        self.ds = base_dataset
        self.preprocess = clip_preprocess
        self.samples: List[Tuple[Path, int, float]] = []  # (path, label, ratio)

        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(self.ds.patch_list))

        c0_count = 0
        c1_count = 0

        print(f"Indexing patches for {self.ds.fold} (Glacier >= {glacier_thresh_high:.0%}, Non-Glacier < {glacier_thresh_low:.0%})...")
        for idx in shuffled_indices:
            gid, p_path = self.ds.patch_list[idx]
            # Quick read of mask to determine ratio
            _, mask, _ = self.ds.read_netcdf_patch(p_path)
            ratio = float(mask.mean())

            if ratio >= glacier_thresh_high and c1_count < max_samples_per_class:
                self.samples.append((p_path, 1, ratio))
                c1_count += 1
            elif ratio < glacier_thresh_low and c0_count < max_samples_per_class:
                self.samples.append((p_path, 0, ratio))
                c0_count += 1

            if c0_count >= max_samples_per_class and c1_count >= max_samples_per_class:
                break

        print(f"  {self.ds.fold}: Loaded {len(self.samples)} balanced patches ({c0_count} Non-Glacier, {c1_count} Glacier).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, _ = self.samples[idx]
        raw_bands, _, _ = self.ds.read_netcdf_patch(path, scale=self.ds.scale)
        # S2 RGB: Red=B4 (idx 3), Green=B3 (idx 2), Blue=B2 (idx 1)
        rgb = np.stack([raw_bands[3], raw_bands[2], raw_bands[1]], axis=-1)
        p2, p98 = np.nanpercentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
        img_pil = Image.fromarray((rgb * 255).astype(np.uint8))
        tensor = self.preprocess(img_pil)
        return tensor, label, path.name


# =====================================================================
# COOP LEARNABLE PROMPT MODULE (Zhou et al., 2022)
# =====================================================================
class CoOpPromptLearner(nn.Module):
    """Context Optimization (CoOp) prompt learner for CLIP."""

    def __init__(self, clip_model, n_ctx: int = 16, classnames: List[str] = ["mountain bedrock", "glacier ice"]):
        super().__init__()
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.n_cls = len(classnames)

        # Get embedding dimension and token embedding layer from CLIP
        embed_dim = clip_model.token_embedding.embedding_dim
        # Initialize context vectors with standard normal distribution scaled by 0.02
        ctx_vectors = torch.empty(n_ctx, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)  # Learnable parameters (16 x 512)

        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Tokenize classnames
        self.tokenized_prompts = []
        for name in classnames:
            # e.g. "X X X ... X glacier ice"
            prompt = " ".join(["X"] * n_ctx) + " " + name
            tokens = tokenizer(prompt)[0]
            self.tokenized_prompts.append(tokens)
        self.tokenized_prompts = torch.stack(self.tokenized_prompts)  # (n_cls, 77)

    def forward(self, clip_model) -> torch.Tensor:
        """Construct full prompt embeddings and pass through CLIP text encoder."""
        device = self.ctx.device
        token_embed = clip_model.token_embedding
        tokenized = self.tokenized_prompts.to(device)

        # Get original embeddings for the tokenized prompts
        with torch.no_grad():
            orig_embeds = token_embed(tokenized)  # (n_cls, 77, 512)

        # Insert learnable context tokens in positions 1 to n_ctx
        # Position 0 is SOS token (<start_of_text>)
        # Positions 1 to n_ctx are learnable context vectors
        # Positions n_ctx+1 onwards are class name + EOS + padding
        prefix = orig_embeds[:, :1, :]
        suffix = orig_embeds[:, 1 + self.n_ctx:, :]
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # (n_cls, n_ctx, 512)

        prompts = torch.cat([prefix, ctx, suffix], dim=1)  # (n_cls, 77, 512)

        # Pass through CLIP text transformer (all transformer weights remain frozen)
        cast_dtype = clip_model.transformer.get_cast_dtype() if hasattr(clip_model.transformer, "get_cast_dtype") else torch.float32
        prompts = prompts.to(cast_dtype)
        x = prompts + clip_model.positional_embedding.to(device=device, dtype=cast_dtype)
        attn_mask = getattr(clip_model, "attn_mask", None)
        if attn_mask is not None:
            x = clip_model.transformer(x, attn_mask=attn_mask.to(device=device, dtype=cast_dtype))
        else:
            x = clip_model.transformer(x)
        x = clip_model.ln_final(x)

        # Take features from the EOS token (highest token index)
        # In CLIP, text projection is applied to the EOS token feature
        eos_indices = tokenized.argmax(dim=-1)
        text_features = x[torch.arange(self.n_cls), eos_indices]
        if clip_model.text_projection is not None:
            if isinstance(clip_model.text_projection, nn.Linear):
                text_features = clip_model.text_projection(text_features)
            else:
                text_features = text_features @ clip_model.text_projection
        return F.normalize(text_features, p=2, dim=-1)  # (n_cls, 512)


# =====================================================================
# EVALUATION & BENCHMARKING
# =====================================================================
def run_clip_coop_experiment():
    print("=" * 65)
    print("OBJECTIVE 5: GENUINE PRETRAINED CLIP vs LEARNABLE COOP PROMPTS")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Pretrained CLIP Model
    print("\n[Step 1/5] Loading genuine pretrained CLIP ViT-B-32 (OpenAI weights)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval()

    # Freeze entire CLIP model
    for p in clip_model.parameters():
        p.requires_grad = False
    print("Pretrained CLIP ViT-B-32 loaded and completely frozen.")

    # 2. Prepare Datasets (Disjoint Glacier IDs)
    print("\n[Step 2/5] Preparing disjoint train, validation, and test datasets...")
    train_base = DL4GAMDataset(root=ROOT, fold="fold_train")
    test_base = DL4GAMDataset(root=ROOT, fold="fold_test")

    # K-shot training support set (16 patches per class = 32 labeled patches)
    train_clip_ds = CLIPGlacierPatchDataset(
        train_base, clip_preprocess, glacier_thresh_high=0.10, glacier_thresh_low=0.02, max_samples_per_class=16, seed=42
    )

    # Held-out test set (50 patches per class = 100 test patches)
    test_clip_ds = CLIPGlacierPatchDataset(
        test_base, clip_preprocess, glacier_thresh_high=0.10, glacier_thresh_low=0.02, max_samples_per_class=50, seed=123
    )

    # 3. Method A: Handcrafted Zero-Shot CLIP Prompts
    print("\n[Step 3/5] Evaluating Handcrafted Zero-Shot CLIP Prompts on test set...")
    handcrafted_prompts_cls0 = [
        "a satellite photo of exposed alpine bedrock and rock terrain",
        "a satellite photo of snow-free mountain ground without glaciers",
        "a satellite photo of rocky alpine mountain terrain",
    ]
    handcrafted_prompts_cls1 = [
        "a satellite photo of clean glacier ice and crevasses",
        "a satellite photo of a mountain glacier in the Alps",
        "a satellite photo of blue glacier ice and snowfields",
    ]

    tokens_c0 = tokenizer(handcrafted_prompts_cls0).to(device)
    tokens_c1 = tokenizer(handcrafted_prompts_cls1).to(device)

    with torch.no_grad():
        feats_c0 = F.normalize(clip_model.encode_text(tokens_c0), p=2, dim=-1).mean(dim=0, keepdim=True)
        feats_c1 = F.normalize(clip_model.encode_text(tokens_c1), p=2, dim=-1).mean(dim=0, keepdim=True)
        feats_c0 = F.normalize(feats_c0, p=2, dim=-1)
        feats_c1 = F.normalize(feats_c1, p=2, dim=-1)
        handcrafted_text_feats = torch.cat([feats_c0, feats_c1], dim=0)  # (2, 512)

    # Test Handcrafted CLIP
    handcrafted_preds, test_labels, test_img_feats = [], [], []

    with torch.no_grad():
        for i in range(len(test_clip_ds)):
            x_img, y_lbl, _ = test_clip_ds[i]
            x_img = x_img.unsqueeze(0).to(device)
            img_feat = F.normalize(clip_model.encode_image(x_img), p=2, dim=-1)
            test_img_feats.append(img_feat.cpu())
            test_labels.append(y_lbl)

            # Cosine similarity logits
            logits = (img_feat @ handcrafted_text_feats.T) * 100.0  # Scale by 100 (CLIP standard logit scale)
            pred = logits.argmax(dim=-1).item()
            handcrafted_preds.append(pred)

    test_img_feats = torch.cat(test_img_feats, dim=0).to(device)  # (N_test, 512)
    test_labels = np.array(test_labels)
    handcrafted_preds = np.array(handcrafted_preds)

    def calc_cls_metrics(preds, truth):
        tp = int(((preds == 1) & (truth == 1)).sum())
        fp = int(((preds == 1) & (truth == 0)).sum())
        fn = int(((preds == 0) & (truth == 1)).sum())
        tn = int(((preds == 0) & (truth == 0)).sum())
        acc = float((preds == truth).mean())
        prec = float(tp / max(tp + fp, 1))
        rec = float(tp / max(tp + fn, 1))
        f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    metrics_handcrafted = calc_cls_metrics(handcrafted_preds, test_labels)
    print(f"  Handcrafted CLIP Results: Accuracy={metrics_handcrafted['accuracy']:.4f} | F1={metrics_handcrafted['f1']:.4f} | Precision={metrics_handcrafted['precision']:.4f} | Recall={metrics_handcrafted['recall']:.4f}")

    # 4. Method B: Learnable CoOp Prompt Tuning
    print("\n[Step 4/5] Training Learnable CoOp Context Tokens (M=16, 32 few-shot patches)...")
    prompt_learner = CoOpPromptLearner(clip_model, n_ctx=16, classnames=["mountain bedrock", "glacier ice"]).to(device)
    optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=0.002, weight_decay=1e-4)

    # Pre-extract few-shot training image embeddings
    train_img_feats, train_labels = [], []
    with torch.no_grad():
        for i in range(len(train_clip_ds)):
            x_img, y_lbl, _ = train_clip_ds[i]
            x_img = x_img.unsqueeze(0).to(device)
            img_feat = F.normalize(clip_model.encode_image(x_img), p=2, dim=-1)
            train_img_feats.append(img_feat)
            train_labels.append(y_lbl)

    train_img_feats = torch.cat(train_img_feats, dim=0)  # (32, 512)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)

    # Train CoOp prompt vectors
    coop_epochs = 15
    for ep in range(1, coop_epochs + 1):
        prompt_learner.train()
        optimizer.zero_grad()
        text_feats = prompt_learner(clip_model)  # (2, 512)
        logits = (train_img_feats @ text_feats.T) * 100.0  # (32, 2)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        preds_tr = logits.argmax(dim=-1)
        acc_tr = (preds_tr == train_labels).float().mean().item()
        if ep % 5 == 0 or ep == 1:
            print(f"  CoOp Epoch {ep:02d}/{coop_epochs} | Loss: {loss.item():.4f} | Train Acc: {acc_tr:.4f}")

    # Save trained prompt checkpoint
    coop_ckpt_path = OUT_CKPTS / "coop_prompt_best.pt"
    torch.save({"ctx": prompt_learner.ctx.data.cpu(), "n_ctx": 16, "classnames": prompt_learner.classnames}, coop_ckpt_path)
    print(f"Saved CoOp prompt checkpoint: {coop_ckpt_path}")

    # Test CoOp on held-out test patches
    prompt_learner.eval()
    with torch.no_grad():
        coop_text_feats = prompt_learner(clip_model)  # (2, 512)
        coop_logits = (test_img_feats @ coop_text_feats.T) * 100.0
        coop_preds = coop_logits.argmax(dim=-1).cpu().numpy()

    metrics_coop = calc_cls_metrics(coop_preds, test_labels)
    print(f"\n  CoOp Results on Test Patches: Accuracy={metrics_coop['accuracy']:.4f} | F1={metrics_coop['f1']:.4f} | Precision={metrics_coop['precision']:.4f} | Recall={metrics_coop['recall']:.4f}")

    # 5. Diagnostic Comparison Figure & Report
    print("\n[Step 5/5] Generating comparison figure and JSON report...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Bar chart
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    h_vals = [metrics_handcrafted["accuracy"], metrics_handcrafted["precision"], metrics_handcrafted["recall"], metrics_handcrafted["f1"]]
    c_vals = [metrics_coop["accuracy"], metrics_coop["precision"], metrics_coop["recall"], metrics_coop["f1"]]

    x = np.arange(len(metrics_names))
    width = 0.35

    axes[0].bar(x - width/2, h_vals, width, label="Zero-Shot Handcrafted CLIP", color="#4ECDC4")
    axes[0].bar(x + width/2, c_vals, width, label="Learnable CoOp (16 Context Tokens)", color="#00F2FE")
    axes[0].set_ylabel("Score", fontsize=11, fontweight="bold")
    axes[0].set_title("Objective 5: Handcrafted CLIP vs CoOp Performance", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")

    # Confusion Matrix comparison
    cm_h = np.array([[metrics_handcrafted["tn"], metrics_handcrafted["fp"]], [metrics_handcrafted["fn"], metrics_handcrafted["tp"]]])
    cm_c = np.array([[metrics_coop["tn"], metrics_coop["fp"]], [metrics_coop["fn"], metrics_coop["tp"]]])

    im = axes[1].imshow(cm_c, cmap="Blues", interpolation="nearest")
    axes[1].set_title("CoOp Confusion Matrix (Held-Out Test Set)", fontsize=12, fontweight="bold")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Non-Glacier", "Glacier"], fontweight="bold")
    axes[1].set_yticklabels(["Non-Glacier", "Glacier"], fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontweight="bold")
    axes[1].set_ylabel("True Label", fontweight="bold")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm_c[i, j]), ha="center", va="center", color="white" if cm_c[i, j] > cm_c.max()/2 else "black", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = OUT_FIGS / "clip_vs_coop_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure: {fig_path}")

    # JSON Report
    report = {
        "objective": "Objective 5: Comparison of Handcrafted CLIP Prompts and Learnable CoOp Prompts",
        "model_architecture": "Pretrained CLIP ViT-B-32 (OpenAI weights)",
        "visual_encoder": "VisionTransformer",
        "text_encoder": "Transformer",
        "dataset": "DL4GAM Alps",
        "glacier_presence_threshold": {
            "glacier_present_class_1": "mask_crt_g >= 0.10 (at least 10% ice coverage)",
            "non_glacier_class_0": "mask_crt_g < 0.02 (less than 2% ice coverage)",
        },
        "handcrafted_prompts": {
            "non_glacier": handcrafted_prompts_cls0,
            "glacier": handcrafted_prompts_cls1,
        },
        "coop_configuration": {
            "n_ctx_tokens": 16,
            "ctx_dim": 512,
            "classnames": ["mountain bedrock", "glacier ice"],
            "training_samples": len(train_clip_ds),
            "epochs": coop_epochs,
            "optimizer": "AdamW (lr=0.002, weight_decay=1e-4)",
        },
        "test_evaluation": {
            "test_samples_total": len(test_clip_ds),
            "handcrafted_clip": metrics_handcrafted,
            "learnable_coop": metrics_coop,
            "accuracy_improvement": round(metrics_coop["accuracy"] - metrics_handcrafted["accuracy"], 4),
            "f1_improvement": round(metrics_coop["f1"] - metrics_handcrafted["f1"], 4),
        },
        "artifacts": {
            "checkpoint": str(coop_ckpt_path),
            "figure": str(fig_path),
        },
    }

    report_path = OUT_RESULTS / "objective5_clip_coop_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved JSON report: {report_path}")
    print("\n" + "=" * 65)
    print("OBJECTIVE 5 COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    run_clip_coop_experiment()
