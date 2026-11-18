"""Prototypical Few-Shot Segmentation Network for Glacier Detection.

Features:
- Lightweight 13-channel multispectral convolutional encoder.
- Dense spatial feature embeddings.
- Masked average pooling for glacier foreground and background prototypes.
- Pixel-wise cosine similarity against query embeddings with temperature scaling.
- Bilinear upsampling to full spatial resolution (256x256).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FeatureEncoder(nn.Module):
    """13-Channel Multispectral Encoder for Dense Spatial Embeddings."""

    def __init__(self, in_channels: int = 13, embed_dim: int = 64):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBlock(in_channels, 32, stride=1),
            ConvBlock(32, 32, stride=1),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64, stride=2),  # Downsample 256 -> 128
            ConvBlock(64, 64, stride=1),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128, stride=2), # Downsample 128 -> 64
            ConvBlock(128, 128, stride=1),
        )
        # Dilated residual stage to expand receptive field without losing spatial resolution
        self.layer4 = nn.Sequential(
            ConvBlock(128, 128, dilation=2, padding=2),
            ConvBlock(128, 128, dilation=2, padding=2),
        )
        # Final projection to embedding space
        self.proj = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        return x  # Shape: (B, embed_dim, 64, 64)


class FewShotProtoNet(nn.Module):
    """Prototypical Few-Shot Segmentation Network."""

    def __init__(self, in_channels: int = 13, embed_dim: int = 64, temperature: float = 20.0):
        super().__init__()
        self.encoder = FeatureEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.scale = nn.Parameter(torch.tensor(float(temperature)))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial embeddings and L2-normalize along channel dimension."""
        feats = self.encoder(x)  # (B, D, H', W')
        return F.normalize(feats, p=2, dim=1)

    def compute_prototypes(
        self,
        support_feats: torch.Tensor,  # (K, D, H', W')
        support_masks: torch.Tensor   # (K, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute foreground (glacier) and background prototypes via Masked Average Pooling."""
        K, D, H_feat, W_feat = support_feats.shape

        # Downsample support masks to feature map dimensions (64, 64)
        masks_down = F.interpolate(
            support_masks, size=(H_feat, W_feat), mode="nearest"
        )  # (K, 1, H', W')

        fg_mask = masks_down  # (K, 1, H', W')
        bg_mask = 1.0 - masks_down  # (K, 1, H', W')

        eps = 1e-8

        # Weighted masked sum across all K support patches and spatial locations
        # support_feats: (K, D, H', W')
        # fg_mask: (K, 1, H', W')
        fg_weighted = (support_feats * fg_mask).sum(dim=(0, 2, 3))  # (D,)
        fg_count = fg_mask.sum() + eps
        proto_fg = fg_weighted / fg_count  # (D,)

        bg_weighted = (support_feats * bg_mask).sum(dim=(0, 2, 3))  # (D,)
        bg_count = bg_mask.sum() + eps
        proto_bg = bg_weighted / bg_count  # (D,)

        # L2 normalize prototype vectors
        proto_fg = F.normalize(proto_fg, p=2, dim=0)
        proto_bg = F.normalize(proto_bg, p=2, dim=0)

        return proto_fg, proto_bg

    def forward(
        self,
        support_x: torch.Tensor,  # (K, 13, 256, 256)
        support_y: torch.Tensor,  # (K, 1, 256, 256)
        query_x: torch.Tensor,    # (Q, 13, 256, 256)
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for an episode: extracts support prototypes and classifies query pixels."""
        # 1. Extract normalized feature embeddings
        support_feats = self.extract_features(support_x)  # (K, D, 64, 64)
        query_feats = self.extract_features(query_x)      # (Q, D, 64, 64)

        # 2. Compute class prototypes
        proto_fg, proto_bg = self.compute_prototypes(support_feats, support_y)

        # 3. Compute pixel-wise cosine similarity between query embeddings and prototypes
        # query_feats: (Q, D, H', W')
        # proto_fg: (D,) -> reshape to (1, D, 1, 1)
        sim_fg = (query_feats * proto_fg.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # (Q, 1, H', W')
        sim_bg = (query_feats * proto_bg.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # (Q, 1, H', W')

        # 4. Scale similarities by learnable temperature parameter
        temp = torch.clamp(self.scale, min=1.0, max=50.0)
        logits_lowres = temp * (sim_fg - sim_bg)  # (Q, 1, H', W')

        # 5. Upsample logits to original input resolution (256, 256)
        H_orig, W_orig = query_x.shape[-2:]
        logits = F.interpolate(
            logits_lowres, size=(H_orig, W_orig), mode="bilinear", align_corners=False
        )

        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probs": probs,
            "proto_fg": proto_fg,
            "proto_bg": proto_bg,
            "sim_fg": sim_fg,
            "sim_bg": sim_bg,
        }
