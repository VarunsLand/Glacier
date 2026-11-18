"""Few-Shot Episodic Sampler for Glacier Segmentation.

Guarantees strict glacier-level disjointness between support and query sets
within every episode, eliminating spatial autocorrelation leakage.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset import DL4GAMDataset


class Episode:
    """Represents a single Few-Shot Episode with disjoint support and query sets."""

    def __init__(
        self,
        support_x: torch.Tensor,       # (K, 13, H, W)
        support_y: torch.Tensor,       # (K, 1, H, W)
        support_glaciers: List[str],
        query_x: torch.Tensor,         # (Q, 13, H, W)
        query_y: torch.Tensor,         # (Q, 1, H, W)
        query_glaciers: List[str],
        episode_idx: int = 0,
    ):
        self.support_x = support_x
        self.support_y = support_y
        self.support_glaciers = support_glaciers
        self.query_x = query_x
        self.query_y = query_y
        self.query_glaciers = query_glaciers
        self.episode_idx = episode_idx

        # Enforce disjointness assertion
        overlap = set(support_glaciers) & set(query_glaciers)
        if overlap:
            raise ValueError(
                f"Data leakage detected in Episode {episode_idx}! "
                f"Glaciers in both support and query: {overlap}"
            )

    @property
    def k_shot(self) -> int:
        return self.support_x.shape[0]

    @property
    def q_queries(self) -> int:
        return self.query_x.shape[0]

    def to(self, device: torch.device) -> "Episode":
        """Move episode tensors to target device."""
        self.support_x = self.support_x.to(device)
        self.support_y = self.support_y.to(device)
        self.query_x = self.query_x.to(device)
        self.query_y = self.query_y.to(device)
        return self


class FewShotEpisodeSampler:
    """Generates few-shot segmentation episodes with strict glacier separation."""

    def __init__(
        self,
        dataset: DL4GAMDataset,
        k_shot: int = 5,
        q_queries: int = 2,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.seed = seed
        self.rng = random.Random(seed)

        # Build list of eligible glacier IDs that have at least 1 patch
        self.eligible_glaciers = [
            gid for gid, patches in self.dataset.glacier_to_patches.items() if len(patches) > 0
        ]

        if len(self.eligible_glaciers) < (self.k_shot + self.q_queries):
            raise ValueError(
                f"Not enough distinct glaciers ({len(self.eligible_glaciers)}) in fold "
                f"'{dataset.fold}' to sample {self.k_shot} support and {self.q_queries} query glaciers!"
            )

        # Pre-index patches per glacier
        self.glacier_to_indices: Dict[str, List[int]] = {}
        for idx, (gid, path) in enumerate(self.dataset.patch_list):
            self.glacier_to_indices.setdefault(gid, []).append(idx)

    def sample_episode(self, episode_idx: Optional[int] = None) -> Episode:
        """Sample a single K-shot, Q-query episode with zero glacier overlap."""
        if episode_idx is not None:
            # Deterministic seeding per episode index for exact reproducibility
            ep_rng = random.Random(self.seed + episode_idx * 10007)
        else:
            ep_rng = self.rng

        # 1. Randomly sample disjoint glacier IDs for support and query
        total_needed = self.k_shot + self.q_queries
        selected_glaciers = ep_rng.sample(self.eligible_glaciers, total_needed)

        support_gids = selected_glaciers[:self.k_shot]
        query_gids = selected_glaciers[self.k_shot:]

        # 2. Sample 1 patch per support glacier
        support_indices = []
        for gid in support_gids:
            patch_idx = ep_rng.choice(self.glacier_to_indices[gid])
            support_indices.append(patch_idx)

        # 3. Sample 1 patch per query glacier
        query_indices = []
        for gid in query_gids:
            patch_idx = ep_rng.choice(self.glacier_to_indices[gid])
            query_indices.append(patch_idx)

        # 4. Load tensors
        support_x_list, support_y_list = [], []
        for idx in support_indices:
            x, y, _ = self.dataset[idx]
            support_x_list.append(x)
            support_y_list.append(y)

        query_x_list, query_y_list = [], []
        for idx in query_indices:
            x, y, _ = self.dataset[idx]
            query_x_list.append(x)
            query_y_list.append(y)

        support_x = torch.stack(support_x_list, dim=0)  # (K, 13, 256, 256)
        support_y = torch.stack(support_y_list, dim=0)  # (K, 1, 256, 256)
        query_x = torch.stack(query_x_list, dim=0)      # (Q, 13, 256, 256)
        query_y = torch.stack(query_y_list, dim=0)      # (Q, 1, 256, 256)

        return Episode(
            support_x=support_x,
            support_y=support_y,
            support_glaciers=support_gids,
            query_x=query_x,
            query_y=query_y,
            query_glaciers=query_gids,
            episode_idx=episode_idx if episode_idx is not None else 0,
        )
