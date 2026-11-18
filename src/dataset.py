"""DL4GAM Dataset Loader and Preprocessing Pipeline.

Strict adherence to split_1 official partitions, 13 Sentinel-2 spectral bands,
target mask mask_crt_g, and verified normalization statistics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr


def get_default_root() -> Path:
    """Return default project root path."""
    return Path(r"C:\Users\Varun's LOQ\Desktop\GlacierResearch")


class DL4GAMDataset(Dataset):
    """Dataset loader for DL4GAM Alps NetCDF patches.
    
    Loads:
      - Spectral data: First 13 Sentinel-2 bands (B1..B12) from 'band_data'
      - Target mask: 'mask_crt_g' (binary glacier mask: 0=background, 1=glacier)
      - Quality verification: checks fill_mask and cloudless_mask
    """

    BAND_NAMES = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
    ]

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split_col: str = "split_1",
        fold: str = "fold_train",
        norm_path: Optional[Union[str, Path]] = None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.root = Path(root) if root else get_default_root()
        self.split_col = split_col
        self.fold = fold
        self.augment = augment
        self.seed = seed

        self.data_dir = self.root / "datasets" / "dl4gam"
        self.extracted_dir = self.data_dir / "extracted"
        self.map_csv = self.data_dir / "data" / "map_all_splits_all_folds.csv"

        if not self.map_csv.exists():
            # Fallback search if path is different
            cands = list(self.data_dir.rglob("map_all_splits_all_folds.csv"))
            if not cands:
                raise FileNotFoundError(f"Missing DL4GAM split metadata under {self.data_dir}")
            self.map_csv = cands[0]

        # Load partition map
        df = pd.read_csv(self.map_csv)
        if self.split_col not in df.columns:
            raise ValueError(f"Split column '{self.split_col}' not found in {self.map_csv}")

        # Filter glacier IDs for the requested fold
        valid_folds = ["fold_train", "fold_valid", "fold_test"]
        if self.fold not in valid_folds:
            raise ValueError(f"Fold '{self.fold}' must be one of {valid_folds}")

        self.glacier_df = df[df[self.split_col] == self.fold].copy()
        self.glacier_ids = sorted(self.glacier_df["entry_id"].tolist())
        self.glacier_id_set = set(self.glacier_ids)

        # Collect all patch paths belonging strictly to this fold's glacier IDs
        self.glacier_to_patches: Dict[str, List[Path]] = {}
        self.patch_list: List[Tuple[str, Path]] = []

        for gid in self.glacier_ids:
            g_folder = self.extracted_dir / gid
            if g_folder.exists():
                patches = sorted(g_folder.glob("*.nc"))
                if patches:
                    self.glacier_to_patches[gid] = patches
                    for p in patches:
                        self.patch_list.append((gid, p))

        # Load normalization statistics (computed from training fold only)
        if norm_path is None:
            norm_path = self.root / "outputs" / "normalization_split_1.json"
        self.norm_path = Path(norm_path)
        
        if not self.norm_path.exists():
            raise FileNotFoundError(
                f"Normalization file not found at {self.norm_path}. "
                "Ensure compute_normalization.py has run."
            )

        with open(self.norm_path, "r", encoding="utf-8") as f:
            norm_meta = json.load(f)

        self.scale = float(norm_meta.get("scale_factor", 0.0001))
        self.mean = np.array(norm_meta["mean"], dtype=np.float32).reshape(13, 1, 1)
        self.std = np.array(norm_meta["std"], dtype=np.float32).reshape(13, 1, 1)
        self.pos_weight = float(norm_meta.get("pos_weight", 3.18))

    def __len__(self) -> int:
        return len(self.patch_list)

    @staticmethod
    def read_netcdf_patch(
        path: Path,
        scale: float = 0.0001
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Read a single NetCDF patch and extract 13 bands + mask_crt_g."""
        try:
            ds = xr.open_dataset(path, engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(path)

        with ds:
            if "band_data" not in ds or "mask_crt_g" not in ds:
                raise KeyError(f"Expected 'band_data' and 'mask_crt_g' in {path}")

            raw_bands = ds["band_data"].values  # shape: (15, 256, 256)
            raw_mask = ds["mask_crt_g"].values  # shape: (256, 256)

            # Slicing the first 13 Sentinel-2 spectral bands
            bands13 = raw_bands[:13].astype(np.float32) * scale

            # Quality check using slices 13 (cloudless) and 14 (fill) if present
            has_fill = True
            if raw_bands.shape[0] >= 15:
                fill_mask = raw_bands[14]
                invalid_pixels = (fill_mask == 0) | np.isnan(fill_mask)
                if np.any(invalid_pixels):
                    bands13[:, invalid_pixels] = 0.0
                    has_fill = False

            # NaN and finite value guardrails
            bands13 = np.nan_to_num(bands13, nan=0.0, posinf=0.0, neginf=0.0)
            bands13 = np.clip(bands13, 0.0, 1.5)

            # Target mask guardrail: binary 0/1
            mask = np.nan_to_num(raw_mask, nan=0.0, posinf=0.0, neginf=0.0)
            mask = (mask > 0.5).astype(np.float32)

            meta = {
                "valid_fill": 1.0 if has_fill else 0.0,
                "glacier_pixel_ratio": float(mask.mean()),
            }

        return bands13, mask, meta

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        gid, patch_path = self.patch_list[idx]
        bands13, mask, meta = self.read_netcdf_patch(patch_path, scale=self.scale)

        # Standardize using training fold statistics
        norm_bands = (bands13 - self.mean) / np.maximum(self.std, 1e-6)
        norm_bands = np.clip(norm_bands, -8.0, 8.0)

        # Data augmentation (training only)
        if self.augment:
            if np.random.rand() > 0.5:
                norm_bands = norm_bands[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() > 0.5:
                norm_bands = norm_bands[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k > 0:
                norm_bands = np.rot90(norm_bands, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        x = torch.from_numpy(norm_bands.copy()).float()
        y = torch.from_numpy(mask[None, :, :].copy()).float()  # (1, 256, 256)

        info = {
            "glacier_id": gid,
            "patch_name": patch_path.name,
            "patch_path": str(patch_path),
            "fold": self.fold,
        }

        return x, y, info

    def get_patches_by_glacier(self, glacier_id: str) -> List[Path]:
        """Return all patches belonging to a specific glacier ID."""
        return self.glacier_to_patches.get(glacier_id, [])

    def get_all_glacier_ids(self) -> List[str]:
        """Return sorted list of all unique glacier IDs in this fold."""
        return list(self.glacier_ids)
