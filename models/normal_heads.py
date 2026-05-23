"""Surface-normal heads.

N1..N9: inherit from DepthHead (same per-patch feature construction as
        D1..D9), swap the 1-channel scalar head for a 3-channel XYZ head.

N_BN / N_DQ: official-style BNHead-equivalent baselines (parallel to S_BN
        / S_DQ for seg).  N_BN = BN + 2-layer conv to 3 channels.  N_DQ =
        N_BN + register-as-query cross-attn residual (out_channels=3).
        These cases bypass DepthHead inheritance because there is no
        corresponding D-variant.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_heads import DepthHead, EMBED_DIM, PATCH_GRID
from .seg_heads import NUM_REG


VARIANTS_NORMAL = ("N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9",
                   "N_BN", "N_DQ")


def _d_for(n_variant: str) -> str:
    return "D" + n_variant[1:]


class NormalHead(DepthHead):
    """N1..N9: thin shell over DepthHead with a 3-channel output head."""

    def __init__(self, variant: str, embed_dim: int = EMBED_DIM,
                 hidden: int = 128, grid: int = PATCH_GRID):
        assert variant in VARIANTS_NORMAL
        if variant in ("N_BN", "N_DQ"):
            raise ValueError(
                f"{variant} is implemented by NormalHeadBNDQ; "
                f"use make_normal_head(variant) instead.")
        d_variant = _d_for(variant)
        # Initialize as the depth variant — gets correct build_feature behavior,
        # routing module, optional random buffer, etc.
        super().__init__(variant=d_variant, embed_dim=embed_dim,
                         hidden=hidden, grid=grid)
        # Replace the 1-channel head with a 3-channel one
        D_in = self.head[0].in_channels
        self.head = nn.Sequential(
            nn.Conv2d(D_in, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),   # XYZ normal
        )

    def forward(self, batch: Dict[str, torch.Tensor], out_hw: int = 224):
        f = self.build_feature(batch)                                   # [B, HW, D_in]
        B, HW, D_in = f.shape
        f = f.transpose(1, 2).reshape(B, D_in, self.grid, self.grid)    # [B, D_in, H, W]
        normal = self.head(f)                                           # [B, 3, H, W]
        normal_full = F.interpolate(normal, size=(out_hw, out_hw),
                                    mode="bilinear", align_corners=False)
        out = {"normal": normal, "normal_full": normal_full}
        if self.variant in ("D4", "D6", "D7"):
            out["routing"] = self.routing.routing_weights()
        elif self.variant == "D5":
            out["routing"] = self.routing.routing_weights(patches=batch["patches"])
        return out


class NormalHeadBNDQ(nn.Module):
    """N_BN  = official BNHead-style baseline: BN(patches) + 2-layer conv → 3 ch.
    N_DQ  = N_BN + register-as-query cross-attn residual (out_channels=3).

    Same training interface as NormalHead: forward(batch, out_hw) returns
    dict with keys {'normal', 'normal_full'}.
    """

    def __init__(self, variant: str, embed_dim: int = EMBED_DIM,
                 hidden: int = 128, grid: int = PATCH_GRID,
                 dq_decoder_dim: int = 256):
        super().__init__()
        assert variant in ("N_BN", "N_DQ")
        self.variant = variant
        self.embed_dim = embed_dim
        self.grid = grid

        self.bn = nn.BatchNorm2d(embed_dim)
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

        self.use_dq = (variant == "N_DQ")
        if self.use_dq:
            from .dpt import RegQueryBranch
            self.reg_query = RegQueryBranch(
                embed_dim=embed_dim, decoder_dim=dq_decoder_dim,
                num_layers=2, num_heads=8, grid=grid,
                num_queries=NUM_REG, out_channels=3,
            )

    def forward(self, batch: Dict[str, torch.Tensor], out_hw: int = 224):
        patches = batch["patches"]                                # [B, HW, D]
        B, HW, D = patches.shape
        f = patches.transpose(1, 2).reshape(B, D, self.grid, self.grid)  # [B, D, H, W]
        f = self.bn(f)
        normal = self.head(f)                                     # [B, 3, H, W]
        if self.use_dq:
            residual = self.reg_query(batch["regs"], patches)     # [B, 3, H, W]
            normal = normal + residual
        normal_full = F.interpolate(normal, size=(out_hw, out_hw),
                                    mode="bilinear", align_corners=False)
        return {"normal": normal, "normal_full": normal_full}


def make_normal_head(variant: str, **kwargs):
    """Factory: returns the right class for the requested variant."""
    if variant in ("N_BN", "N_DQ"):
        return NormalHeadBNDQ(variant=variant, **kwargs)
    return NormalHead(variant=variant, **kwargs)
