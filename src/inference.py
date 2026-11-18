"""GlacierVision Inference and Live Analysis Engine.

Provides unified, production-grade model inference, data ingestion, dynamic calculations,
and visualization utilities for the GlacierVision interactive research platform.

Supports:
- Genuine 13-band Sentinel-2 U-Net Baseline segmentation.
- Genuine Prototypical Network (ProtoNet) few-shot segmentation (K in {1, 5, 10, 20}).
- Genuine Pretrained CLIP ViT-B-32 & Context Optimization (CoOp) prompt tuning.
- Real-time NetCDF multispectral ingestion and validation (13 Sentinel-2 bands).
- Dynamic GLIMS multi-temporal retreat and area-change accounting.
"""
from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import open_clip
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

from src.dataset import DL4GAMDataset
from src.fewshot_model import FewShotProtoNet
from src.unet_baseline import GlacierUNet

ROOT = Path("C:/Users/Varun's LOQ/Desktop/GlacierResearch")
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
RUNS_DIR = ROOT / "outputs" / "runs"
DEMO_DIR = ROOT / "outputs" / "demo_samples"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Global in-memory cache for models when used outside Streamlit
_GLOBAL_MODEL_CACHE: Dict[str, Any] = {}


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

        embed_dim = clip_model.token_embedding.embedding_dim
        ctx_vectors = torch.empty(n_ctx, embed_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.tokenized_prompts = []
        for name in classnames:
            prompt = " ".join(["X"] * n_ctx) + " " + name
            tokens = tokenizer(prompt)[0]
            self.tokenized_prompts.append(tokens)
        self.tokenized_prompts = torch.stack(self.tokenized_prompts)

    def forward(self, clip_model) -> torch.Tensor:
        """Construct prompt embeddings and pass through CLIP text encoder."""
        device = self.ctx.device
        token_embed = clip_model.token_embedding
        tokenized = self.tokenized_prompts.to(device)

        with torch.no_grad():
            orig_embeds = token_embed(tokenized)

        prefix = orig_embeds[:, :1, :]
        suffix = orig_embeds[:, 1 + self.n_ctx :, :]
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        cast_dtype = clip_model.transformer.get_cast_dtype() if hasattr(clip_model.transformer, "get_cast_dtype") else torch.float32
        prompts = prompts.to(cast_dtype)
        x = prompts + clip_model.positional_embedding.to(device=device, dtype=cast_dtype)
        attn_mask = getattr(clip_model, "attn_mask", None)
        if attn_mask is not None:
            x = clip_model.transformer(x, attn_mask=attn_mask.to(device=device, dtype=cast_dtype))
        else:
            x = clip_model.transformer(x)
        x = clip_model.ln_final(x)

        eos_indices = tokenized.argmax(dim=-1)
        text_features = x[torch.arange(self.n_cls), eos_indices]
        if clip_model.text_projection is not None:
            if isinstance(clip_model.text_projection, nn.Linear):
                text_features = clip_model.text_projection(text_features)
            else:
                text_features = text_features @ clip_model.text_projection
        return F.normalize(text_features, p=2, dim=-1)


# =====================================================================
# DEVICE & NORMALIZATION
# =====================================================================
def get_device() -> torch.device:
    """Return available torch device (CUDA if available else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_normalization_stats() -> Tuple[np.ndarray, np.ndarray, float]:
    """Load official training-split normalization statistics."""
    norm_path = ROOT / "outputs" / "normalization_split_1.json"
    with open(norm_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scale = float(meta.get("scale_factor", 0.0001))
    mean = np.array(meta["mean"], dtype=np.float32).reshape(13, 1, 1)
    std = np.array(meta["std"], dtype=np.float32).reshape(13, 1, 1)
    return mean, std, scale


# =====================================================================
# MODEL LOADERS
# =====================================================================
def load_unet_model(device: Optional[torch.device] = None) -> GlacierUNet:
    """Load trained 13-band U-Net baseline checkpoint."""
    if device is None:
        device = get_device()
    cache_key = f"unet_{device.type}"
    if cache_key in _GLOBAL_MODEL_CACHE:
        return _GLOBAL_MODEL_CACHE[cache_key]

    ckpt_path = RUNS_DIR / "objective1_run_002_recovery" / "checkpoints" / "unet_baseline_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"U-Net checkpoint missing at: {ckpt_path}")

    model = GlacierUNet(in_channels=13, base_channels=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    _GLOBAL_MODEL_CACHE[cache_key] = model
    return model


def load_protonet_model(device: Optional[torch.device] = None) -> FewShotProtoNet:
    """Load trained 13-band Few-Shot ProtoNet checkpoint."""
    if device is None:
        device = get_device()
    cache_key = f"protonet_{device.type}"
    if cache_key in _GLOBAL_MODEL_CACHE:
        return _GLOBAL_MODEL_CACHE[cache_key]

    ckpt_path = RUNS_DIR / "objective1_run_002_recovery" / "checkpoints" / "fewshot_protonet_best.pt"
    if not ckpt_path.exists():
        ckpt_path = RUNS_DIR / "objective1_run_001" / "checkpoints" / "fewshot_protonet_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ProtoNet checkpoint missing at: {ckpt_path}")

    model = FewShotProtoNet(in_channels=13, embed_dim=64, temperature=20.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    _GLOBAL_MODEL_CACHE[cache_key] = model
    return model


def load_cached_support_set(k: int = 5, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample and cache K support patches from the official training fold (disjoint from test)."""
    if device is None:
        device = get_device()
    cache_key = f"support_k_{k}_{device.type}"
    if cache_key in _GLOBAL_MODEL_CACHE:
        return _GLOBAL_MODEL_CACHE[cache_key]

    ds_train = DL4GAMDataset(root=ROOT, fold="fold_train")
    mean, std, scale = get_normalization_stats()

    # Deterministic selection of high-quality diverse support patches from fold_train
    # Pick glaciers with clear glacier bodies to produce reliable prototypes
    support_x_list = []
    support_y_list = []

    # Pick indices spaced across training patches that have high glacier pixel presence
    # Enforce strictly distinct glacier IDs to ensure diverse, non-duplicate prototypes
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(ds_train))

    seen_gids = set()
    count = 0
    for idx in indices:
        gid, p_path = ds_train.patch_list[idx]
        if gid in seen_gids:
            continue
        try:
            b, m, meta = DL4GAMDataset.read_netcdf_patch(p_path, scale=scale)
            if meta["glacier_pixel_ratio"] >= 0.05:  # Ensure legitimate glacier presence
                b_norm = (b - mean) / (std + 1e-8)
                support_x_list.append(torch.from_numpy(b_norm).float())
                support_y_list.append(torch.from_numpy(m).float().unsqueeze(0))
                seen_gids.add(gid)
                count += 1
                if count >= k:
                    break
        except Exception:
            continue

    if len(support_x_list) < k:
        # Fallback to first available if filter was too strict
        for i in range(len(support_x_list), k):
            gid, p_path = ds_train.patch_list[i]
            if gid in seen_gids:
                continue
            b, m, _ = DL4GAMDataset.read_netcdf_patch(p_path, scale=scale)
            b_norm = (b - mean) / (std + 1e-8)
            support_x_list.append(torch.from_numpy(b_norm).float())
            support_y_list.append(torch.from_numpy(m).float().unsqueeze(0))
            seen_gids.add(gid)

    support_x = torch.stack(support_x_list).to(device)
    support_y = torch.stack(support_y_list).to(device)

    _GLOBAL_MODEL_CACHE[cache_key] = (support_x, support_y)
    return support_x, support_y


def load_clip_coop_bundle(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load genuine OpenAI CLIP ViT-B-32 and trained CoOp prompt embeddings."""
    if device is None:
        device = get_device()
    cache_key = f"clip_coop_{device.type}"
    if cache_key in _GLOBAL_MODEL_CACHE:
        return _GLOBAL_MODEL_CACHE[cache_key]

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Handcrafted prompts text embeddings
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

    # CoOp prompt checkpoint
    coop_ckpt_path = ROOT / "outputs" / "checkpoints" / "coop_prompt_best.pt"
    if not coop_ckpt_path.exists():
        raise FileNotFoundError(f"CoOp checkpoint missing at: {coop_ckpt_path}")

    ckpt_coop = torch.load(coop_ckpt_path, map_location=device, weights_only=False)
    prompt_learner = CoOpPromptLearner(
        clip_model,
        n_ctx=ckpt_coop.get("n_ctx", 16),
        classnames=ckpt_coop.get("classnames", ["mountain bedrock", "glacier ice"]),
    ).to(device)
    prompt_learner.ctx.data.copy_(ckpt_coop["ctx"].to(device))
    prompt_learner.eval()

    with torch.no_grad():
        coop_text_feats = prompt_learner(clip_model)  # (2, 512)

    bundle = {
        "clip_model": clip_model,
        "clip_preprocess": clip_preprocess,
        "tokenizer": tokenizer,
        "handcrafted_text_feats": handcrafted_text_feats,
        "prompt_learner": prompt_learner,
        "coop_text_feats": coop_text_feats,
        "classnames": ["Mountain Bedrock", "Glacier Ice"],
    }
    _GLOBAL_MODEL_CACHE[cache_key] = bundle
    return bundle


# =====================================================================
# DATA INGESTION & VALIDATION
# =====================================================================
def read_and_validate_netcdf(
    source: Union[str, Path, io.BytesIO, Any],
    scale: float = 0.0001,
) -> Dict[str, Any]:
    """Read a NetCDF file, validate its spectral bands, and extract products.
    
    Returns:
        Dictionary containing raw bands (13, H, W), ground truth mask (if present),
        True-color RGB, False-color SWIR, NDSI index, and descriptive metadata.
    """
    temp_clean_path: Optional[Path] = None
    if isinstance(source, (str, Path)):
        p = Path(source)
        source_name = p.name
        try:
            ds = xr.open_dataset(p, engine="netcdf4")
        except Exception:
            try:
                ds = xr.open_dataset(p, engine="h5netcdf")
            except Exception:
                ds = xr.open_dataset(p)
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        raw_bytes = source.read()
        source_name = getattr(source, "name", "uploaded_patch.nc")
        try:
            ds = xr.open_dataset(io.BytesIO(raw_bytes), engine="h5netcdf")
        except Exception:
            try:
                ds = xr.open_dataset(io.BytesIO(raw_bytes))
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    tmp.write(raw_bytes)
                    temp_clean_path = Path(tmp.name)
                ds = xr.open_dataset(temp_clean_path, engine="netcdf4")
    elif isinstance(source, bytes):
        raw_bytes = source
        source_name = "in_memory_patch.nc"
        try:
            ds = xr.open_dataset(io.BytesIO(raw_bytes), engine="h5netcdf")
        except Exception:
            try:
                ds = xr.open_dataset(io.BytesIO(raw_bytes))
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    tmp.write(raw_bytes)
                    temp_clean_path = Path(tmp.name)
                ds = xr.open_dataset(temp_clean_path, engine="netcdf4")
    else:
        raise ValueError(f"Unsupported NetCDF source format: {type(source)}")

    try:
        if "band_data" not in ds:
            raise KeyError(
                "Uploaded NetCDF file missing 'band_data' array. "
                "Expected multispectral array with at least 13 Sentinel-2 channels."
            )

        raw_bands = ds["band_data"].values
        if raw_bands.ndim != 3 or raw_bands.shape[0] < 13:
            raise ValueError(
                f"NetCDF 'band_data' has shape {raw_bands.shape}. Expected (C>=13, H, W)."
            )

        bands13 = raw_bands[:13].astype(np.float32) * scale
        bands13 = np.nan_to_num(bands13, nan=0.0, posinf=0.0, neginf=0.0)
        bands13 = np.clip(bands13, 0.0, 1.5)

        gt_mask: Optional[np.ndarray] = None
        has_gt = "mask_crt_g" in ds
        if has_gt:
            raw_mask = ds["mask_crt_g"].values
            mask = np.nan_to_num(raw_mask, nan=0.0, posinf=0.0, neginf=0.0)
            gt_mask = (mask > 0.5).astype(np.float32)

        # Sentinel-2 Band Mappings:
        # B2=Blue (idx 1), B3=Green (idx 2), B4=Red (idx 3), B8=NIR (idx 7), B11=SWIR1 (idx 11)
        # 1. True-Color RGB Preview (B4, B3, B2)
        rgb_raw = np.stack([bands13[3], bands13[2], bands13[1]], axis=-1)
        p2, p98 = np.nanpercentile(rgb_raw, (2, 98))
        rgb_norm = np.clip((rgb_raw - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

        # 2. False-Color SWIR Composite (B11=Red, B8=Green, B4=Blue)
        # Glaciers appear vivid icy-cyan/blue; vegetation is bright green; bare soil is reddish/brown
        swir_raw = np.stack([bands13[11], bands13[7], bands13[3]], axis=-1)
        p2_s, p98_s = np.nanpercentile(swir_raw, (2, 98))
        swir_norm = np.clip((swir_raw - p2_s) / max(p98_s - p2_s, 1e-6), 0.0, 1.0)
        swir_uint8 = (swir_norm * 255).astype(np.uint8)

        # 3. Normalized Difference Snow Index (NDSI): (Green - SWIR1) / (Green + SWIR1)
        green = bands13[2]
        swir1 = bands13[11]
        denom = green + swir1
        ndsi = np.where(denom > 1e-6, (green - swir1) / (denom + 1e-6), 0.0).astype(np.float32)
        ndsi = np.clip(ndsi, -1.0, 1.0)

        meta = {
            "source_name": source_name,
            "dimensions": f"{bands13.shape[1]}x{bands13.shape[2]}",
            "channels": int(bands13.shape[0]),
            "has_ground_truth": has_gt,
            "glacier_pixel_ratio": float(gt_mask.mean()) if gt_mask is not None else None,
            "mean_reflectance": float(bands13.mean()),
            "std_reflectance": float(bands13.std()),
        }

        return {
            "raw_bands": bands13,
            "gt_mask": gt_mask,
            "rgb_preview": rgb_uint8,
            "swir_preview": swir_uint8,
            "ndsi": ndsi,
            "metadata": meta,
        }
    finally:
        ds.close()
        if temp_clean_path is not None and temp_clean_path.exists():
            try:
                temp_clean_path.unlink()
            except Exception:
                pass


# =====================================================================
# LIVE INFERENCE EXECUTIONS
# =====================================================================
def run_unet_inference(
    raw_bands: np.ndarray,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute live U-Net baseline segmentation on 13-channel multispectral input."""
    if device is None:
        device = get_device()
    model = load_unet_model(device)
    mean, std, _ = get_normalization_stats()

    norm_bands = (raw_bands - mean) / (std + 1e-8)
    tensor = torch.from_numpy(norm_bands).unsqueeze(0).float().to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_mask = (probs >= 0.5).astype(np.uint8)
    return probs, pred_mask, latency_ms


def run_protonet_inference(
    raw_bands: np.ndarray,
    k_shots: int = 5,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute live Few-Shot ProtoNet segmentation using K support patches from training fold."""
    if device is None:
        device = get_device()
    model = load_protonet_model(device)
    mean, std, _ = get_normalization_stats()
    support_x, support_y = load_cached_support_set(k=k_shots, device=device)

    norm_bands = (raw_bands - mean) / (std + 1e-8)
    query_tensor = torch.from_numpy(norm_bands).unsqueeze(0).float().to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(support_x[:k_shots], support_y[:k_shots], query_tensor)
        probs = out["probs"].squeeze().cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_mask = (probs >= 0.5).astype(np.uint8)
    return probs, pred_mask, latency_ms


def run_clip_coop_classification(
    image: Union[Image.Image, np.ndarray],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute live Zero-Shot CLIP and Learnable CoOp classification on an RGB image."""
    if device is None:
        device = get_device()
    bundle = load_clip_coop_bundle(device)

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        pil_img = image.convert("RGB")

    clip_preprocess = bundle["clip_preprocess"]
    clip_model = bundle["clip_model"]
    handcrafted_feats = bundle["handcrafted_text_feats"]
    coop_feats = bundle["coop_text_feats"]
    classnames = bundle["classnames"]

    img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        img_feat = F.normalize(clip_model.encode_image(img_tensor), p=2, dim=-1)

        # 1. Handcrafted Zero-Shot CLIP
        clip_logits = (img_feat @ handcrafted_feats.T) * 100.0
        clip_probs = F.softmax(clip_logits, dim=-1).squeeze().cpu().numpy()
        clip_pred_idx = int(clip_probs.argmax())

        # 2. Learnable CoOp
        coop_logits = (img_feat @ coop_feats.T) * 100.0
        coop_probs = F.softmax(coop_logits, dim=-1).squeeze().cpu().numpy()
        coop_pred_idx = int(coop_probs.argmax())

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "clip": {
            "prediction": classnames[clip_pred_idx],
            "class_idx": clip_pred_idx,
            "probabilities": {classnames[i]: float(clip_probs[i]) for i in range(len(classnames))},
            "confidence": float(clip_probs[clip_pred_idx]),
        },
        "coop": {
            "prediction": classnames[coop_pred_idx],
            "class_idx": coop_pred_idx,
            "probabilities": {classnames[i]: float(coop_probs[i]) for i in range(len(classnames))},
            "confidence": float(coop_probs[coop_pred_idx]),
        },
        "agreement": clip_pred_idx == coop_pred_idx,
        "latency_ms": latency_ms,
    }


# =====================================================================
# METRICS & VISUALIZATION UTILITIES
# =====================================================================
def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute scientific segmentation evaluation metrics between prediction and ground truth."""
    pred = (pred_mask > 0.5).astype(bool)
    gt = (gt_mask > 0.5).astype(bool)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())

    eps = 1e-8
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def generate_contour_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    contour_color: Tuple[int, int, int] = (0, 242, 254),
) -> np.ndarray:
    """Overlay binary mask boundary contour onto an RGB image."""
    overlay = rgb.copy()
    m = (mask > 0.5).astype(np.uint8)

    # 4-connectivity boundary detection
    pad = np.pad(m, 1, mode="constant", constant_values=0)
    boundary = (
        (m == 1)
        & (
            (pad[:-2, 1:-1] == 0)
            | (pad[2:, 1:-1] == 0)
            | (pad[1:-1, :-2] == 0)
            | (pad[1:-1, 2:] == 0)
        )
    )

    overlay[boundary] = contour_color
    return overlay


def generate_error_classification_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Generate 4-color diagnostic error classification map.
    
    True Positive (TP): Bright Green [46, 204, 113]
    False Positive (FP): Vivid Red [231, 76, 60]
    False Negative (FN): Vivid Blue [52, 152, 219]
    True Negative (TN): Dark Slate [30, 41, 59]
    """
    pred = (pred_mask > 0.5).astype(bool)
    gt = (gt_mask > 0.5).astype(bool)

    H, W = pred.shape
    error_map = np.zeros((H, W, 3), dtype=np.uint8)

    # Colors
    c_tp = np.array([46, 204, 113], dtype=np.uint8)   # Green
    c_fp = np.array([231, 76, 60], dtype=np.uint8)    # Red
    c_fn = np.array([52, 152, 219], dtype=np.uint8)   # Blue
    c_tn = np.array([30, 41, 59], dtype=np.uint8)     # Dark Slate

    error_map[pred & gt] = c_tp
    error_map[pred & ~gt] = c_fp
    error_map[~pred & gt] = c_fn
    error_map[~pred & ~gt] = c_tn

    return error_map


# =====================================================================
# DYNAMIC RETREAT & AREA BUDGET CALCULATOR
# =====================================================================
_glims_name_cache: Optional[Dict[str, str]] = None


def get_glims_glacier_name(glac_id: str) -> Optional[str]:
    """Retrieve official glacier name from GLIMS inventory database cache."""
    global _glims_name_cache
    if _glims_name_cache is None:
        p = RESULTS_DIR / "glims_glacier_names.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _glims_name_cache = json.load(f)
            except Exception:
                _glims_name_cache = {}
        else:
            _glims_name_cache = {}
    return _glims_name_cache.get(glac_id.strip().upper())


def get_glacier_dynamic_assessment(
    glacier_id_or_alias: str,
    override_area_earlier: Optional[float] = None,
    override_area_later: Optional[float] = None,
    override_years: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute dynamic glacier retreat, percentage change, and annualized rate.
    
    Supports:
    - Standard GLIMS Glacier IDs (e.g. 'G008379E46648N', 'G010725E46789N')
    - Benchmark Aliases ('Rhone', 'Great Aletsch', 'Unteraar', 'Gorner', 'Findel', 'Mazia')
    - Custom user override values for interactive scenario simulation.
    """
    alias_map = {
        "rhone": "G008379E46648N",
        "rhonegletscher": "G008379E46648N",
        "aletsch": "G008032E46504N",
        "great aletsch": "G008032E46504N",
        "grosser aletschgletscher": "G008032E46504N",
        "unteraar": "G008187E46569N",
        "unteraargletscher": "G008187E46569N",
        "gorner": "G007800E45965N",
        "gornergletscher": "G007800E45965N",
        "findel": "G007880E45990N",
        "findelgletscher": "G007880E45990N",
        "mazia": "G010725E46789N",
        "matscher": "G010725E46789N",
    }
    benchmark_names = {
        "G008379E46648N": "Rhone Glacier (Rhonegletscher)",
        "G008032E46504N": "Great Aletsch Glacier (Grosser Aletschgletscher)",
        "G008187E46569N": "Unteraar Glacier (Unteraargletscher)",
        "G007800E45965N": "Gorner Glacier (Gornergletscher)",
        "G007880E45990N": "Findel Glacier (Findelgletscher)",
        "G010725E46789N": "Mazia / Matscher Glacier (Vedretta di Mazia)",
    }
    case_study_analysis_ids = {
        "G008379E46648N": ("342911", "1095967"),
        "G008032E46504N": ("342455", "1096829"),
        "G008187E46569N": ("342740", "1096690"),
        "G007800E45965N": ("342215", "1096956"),
        "G007880E45990N": ("342320", "1096563"),
    }
    clean_query = glacier_id_or_alias.strip().lower()
    glac_id = alias_map.get(clean_query, glacier_id_or_alias.strip().upper())

    # 1. First priority: Check authoritative projected case-study metrics (EPSG:32632)
    case_studies_path = RESULTS_DIR / "glims_case_studies.json"
    case_record = None
    if case_studies_path.exists():
        try:
            with open(case_studies_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
                case_record = next((c for c in cases if c["glacier_id"].upper() == glac_id.upper()), None)
        except Exception:
            case_record = None

    found = False
    is_case_study = False
    earlier_analysis_id = None
    later_analysis_id = None

    if case_record is not None:
        earlier_date = "1973-09-01"
        later_date = "2015-08-29"
        e_anlys, l_anlys = case_study_analysis_ids.get(glac_id.upper(), (None, None))
        earlier_analysis_id = e_anlys
        later_analysis_id = l_anlys
        a_earlier = float(case_record["earlier_area_km2"])
        a_later = float(case_record["later_area_km2"])
        years = float(case_record.get("time_gap_years", 42.0))
        glac_name = case_record.get("glacier_name", benchmark_names.get(glac_id, glac_id))
        found = True
        is_case_study = True
    else:
        # 2. Second priority: Query validated global GLIMS retreat CSV
        csv_path = RESULTS_DIR / "glims_retreat_assessment_validated.csv"
        record = None
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            sub = df[df["glacier_id"].str.upper() == glac_id.upper()]
            if not sub.empty:
                record = sub.iloc[0].to_dict()

        if record is not None:
            earlier_date = str(record["earlier_date"])
            later_date = str(record["later_date"])
            earlier_analysis_id = str(int(record["earlier_analysis_id"])) if pd.notna(record.get("earlier_analysis_id")) else None
            later_analysis_id = str(int(record["later_analysis_id"])) if pd.notna(record.get("later_analysis_id")) else None
            a_earlier = float(record["earlier_area_km2"])
            a_later = float(record["later_area_km2"])
            years = float(record["time_gap_years"])
            
            # Resolve name from cache or metadata
            name_from_lookup = get_glims_glacier_name(glac_id)
            if glac_id in benchmark_names:
                glac_name = benchmark_names[glac_id]
            elif name_from_lookup:
                glac_name = f"{name_from_lookup} ({glac_id})"
            elif "glacier_name" in record and pd.notna(record["glacier_name"]) and str(record["glacier_name"]).strip():
                glac_name = f"{record['glacier_name']} ({glac_id})"
            else:
                glac_name = f"Glacier Basin ({glac_id})"
            found = True
        elif override_area_earlier is not None or override_area_later is not None:
            # User simulation scenario
            earlier_date = "Baseline Scenario"
            later_date = "Modern Scenario"
            earlier_analysis_id = "USER_SCENARIO_T1"
            later_analysis_id = "USER_SCENARIO_T2"
            a_earlier = float(override_area_earlier if override_area_earlier is not None else 100.0)
            a_later = float(override_area_later if override_area_later is not None else 80.0)
            years = float(override_years if override_years is not None and override_years > 0 else 10.0)
            glac_name = benchmark_names.get(glac_id, f"Custom Scenario ({glac_id})")
            found = True
        else:
            # Unknown / Invalid ID without overrides
            earlier_date = ""
            later_date = ""
            earlier_analysis_id = None
            later_analysis_id = None
            a_earlier = 0.0
            a_later = 0.0
            years = 42.0
            name_from_lookup = get_glims_glacier_name(glac_id)
            glac_name = benchmark_names.get(glac_id, name_from_lookup or glac_id)
            found = False

    # Apply user overrides if provided
    if override_area_earlier is not None:
        a_earlier = float(override_area_earlier)
    if override_area_later is not None:
        a_later = float(override_area_later)
    if override_years is not None and override_years > 0:
        years = float(override_years)

    # Dynamic Calculations
    if a_earlier > 0:
        if is_case_study and override_area_earlier is None and override_area_later is None and "net_loss_km2" in case_record:
            # Case studies have authoritative geometric net delta computed prior to intermediate decimal rounding
            # e.g., Unteraar 27.131460 - 20.815547 = 6.315913 km² rounds to 6.316 km² under standard rounding
            net_loss_val = float(case_record["net_loss_km2"])
            area_change = -net_loss_val if a_later < a_earlier else net_loss_val
        else:
            area_change = a_later - a_earlier  # Negative indicates retreat (area loss)
        if is_case_study and override_area_earlier is None and override_area_later is None and "percentage_change" in case_record:
            case_pct = float(case_record["percentage_change"])
            pct_change = -case_pct if a_later < a_earlier else case_pct
        else:
            pct_change = (area_change / a_earlier) * 100.0
        annual_rate_km2 = area_change / max(years, 0.1)
        annual_rate_pct = pct_change / max(years, 0.1)
    else:
        area_change = 0.0
        pct_change = 0.0
        annual_rate_km2 = 0.0
        annual_rate_pct = 0.0

    # Dynamics classification with +/- 2.0% deadband
    if pct_change <= -10.0:
        dyn_class = "Significant Retreat"
        status_color = "#E63946"
    elif pct_change <= -2.0:
        dyn_class = "Moderate Retreat"
        status_color = "#F4A261"
    elif pct_change >= 2.0:
        dyn_class = "Advance / Growth"
        status_color = "#2A9D8F"
    else:
        dyn_class = "Stable / Equilibrium"
        status_color = "#457B9D"

    return {
        "glacier_id": glac_id,
        "glacier_name": glac_name,
        "earlier_date": earlier_date,
        "later_date": later_date,
        "earlier_analysis_id": earlier_analysis_id,
        "later_analysis_id": later_analysis_id,
        "earlier_area_km2": a_earlier,
        "later_area_km2": a_later,
        "area_change_km2": area_change,
        "net_loss_km2": abs(min(0.0, area_change)),
        "percent_change": pct_change,
        "time_gap_years": years,
        "annual_rate_km2_yr": annual_rate_km2,
        "annual_rate_pct_yr": annual_rate_pct,
        "dynamics_class": dyn_class,
        "status_color": status_color,
        "found": found,
        "is_case_study": is_case_study,
    }


# =====================================================================
# DEMO DATASET CHOICES
# =====================================================================
def get_demo_netcdf_candidates() -> List[Dict[str, Any]]:
    """Return verified demo NetCDF patches from the held-out test split."""
    candidates = [
        {
            "id": "g_4367",
            "title": "Clean Alpine Glacier (High Ice Coverage: ~88%)",
            "description": "High-altitude large glacier body in the Bernese Alps with distinct accumulation zones and crevasses.",
            "path": ROOT / "datasets" / "dl4gam" / "extracted" / "g_4367" / "g_4367_patch_0_xc_256_yc_256.nc",
        },
        {
            "id": "g_3560",
            "title": "Major Valley Glacier (Complex Boundary: ~83%)",
            "description": "Valley glacier flanked by lateral moraines and steep cirque headwalls.",
            "path": ROOT / "datasets" / "dl4gam" / "extracted" / "g_3560" / "g_3560_patch_0_xc_256_yc_256.nc",
        },
        {
            "id": "g_2316",
            "title": "Mixed Glacial & Bedrock Terrain (Moderate Ice: ~18.7%)",
            "description": "Heterogeneous alpine zone containing glacier tongues, exposed bedrock, and shadow areas.",
            "path": ROOT / "datasets" / "dl4gam" / "extracted" / "g_2316" / "g_2316_patch_0_xc_237_yc_183.nc",
        },
        {
            "id": "g_1266",
            "title": "Glacier Terminus / Bedrock Border (Sparse Ice: ~2.3%)",
            "description": "Receding glacier tongue surrounded by post-glacial debris, proglacial lake, and bedrock.",
            "path": ROOT / "datasets" / "dl4gam" / "extracted" / "g_1266" / "g_1266_patch_0_xc_157_yc_144.nc",
        },
    ]
    # Filter to existing files
    return [c for c in candidates if c["path"].exists()]
