"""Depth-regression heads for the NYU experiment.

Variants D1..D7 mirror S1..S7 exactly: same per-patch feature construction,
only the task head changes.  Instead of a 1x1 conv -> C classes for seg,
depth uses a 2-layer conv head -> 1 scalar (log-depth) -> bilinear upsample.

All variants are on cached frozen-backbone features; 2080 can train any of
them in a couple of minutes.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seg_heads import SpatialRoutingMLP, PATCH_GRID, NUM_REG, EMBED_DIM


VARIANTS_DEPTH = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9")


def _din(variant: str, D: int = EMBED_DIM) -> int:
    return {"D1": D, "D2": 2 * D, "D3": 2 * D, "D4": 2 * D, "D5": 2 * D,
            "D6": D, "D7": 3 * D,
            # controls: same dim as D7, different 3rd channel content
            "D8": 3 * D,   # [patch; cls; patch_copy]    — no new info
            "D9": 3 * D,   # [patch; cls; random_fixed]  — no signal at all
            }[variant]


class DepthHead(nn.Module):
    """Same feature construction as SegHead, different output head."""

    def __init__(self, variant: str, embed_dim: int = EMBED_DIM,
                 hidden: int = 128, grid: int = PATCH_GRID):
        super().__init__()
        assert variant in VARIANTS_DEPTH
        self.variant = variant
        self.embed_dim = embed_dim
        self.grid = grid

        D_in = _din(variant, embed_dim)
        # Depth regression head: 2-layer 1x1 conv -> 1 scalar (log-depth).
        self.head = nn.Sequential(
            nn.Conv2d(D_in, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        self.routing: Optional[SpatialRoutingMLP] = None
        if variant == "D4":
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=0)
        elif variant == "D5":
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=embed_dim,
                                             hidden=32)
        elif variant == "D6":
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=0)
        elif variant == "D7":
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=0)
        elif variant == "D9":
            # Fixed random Gaussian vector (same across all images and seeds),
            # L2-normed to ~register magnitude so the 3rd channel has comparable
            # statistics to D7's reg_ctx, but carries zero useful info.
            g = torch.Generator(); g.manual_seed(0)
            v = torch.randn(embed_dim, generator=g)
            v = v / v.norm().clamp(min=1e-9) * 10.0
            self.register_buffer("random_vec", v)

    # --- feature construction (mirrors SegHead) ---
    def build_feature(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cls     = batch["cls"]          # [B, 1, D]
        regs    = batch["regs"]         # [B, R, D]
        patches = batch["patches"]      # [B, HW, D]
        B, HW, D = patches.shape

        if self.variant == "D1":
            return patches

        if self.variant == "D2":
            return torch.cat([patches, cls[:, 0:1].expand(-1, HW, -1)], dim=-1)

        if self.variant == "D3":
            reg_mean = regs.mean(dim=1, keepdim=True)
            return torch.cat([patches, reg_mean.expand(-1, HW, -1)], dim=-1)

        if self.variant == "D4":
            reg_ctx, _ = self.routing(regs)
            return torch.cat([patches, reg_ctx], dim=-1)

        if self.variant == "D5":
            reg_ctx, _ = self.routing(regs, patches=patches)
            return torch.cat([patches, reg_ctx], dim=-1)

        if self.variant == "D6":
            reg_ctx, _ = self.routing(regs)
            return reg_ctx

        if self.variant == "D7":
            reg_ctx, _ = self.routing(regs)
            cls_b = cls[:, 0:1].expand(-1, HW, -1)
            return torch.cat([patches, cls_b, reg_ctx], dim=-1)

        if self.variant == "D8":
            # control: same 3D dim as D7, but third channel = patch copy
            # (no information beyond patches; tests whether bare param count helps)
            cls_b = cls[:, 0:1].expand(-1, HW, -1)
            return torch.cat([patches, cls_b, patches], dim=-1)

        if self.variant == "D9":
            # control: third channel = a fixed random vector broadcast to all patches
            # (same dim as D7, zero useful signal in the third channel)
            cls_b = cls[:, 0:1].expand(-1, HW, -1)
            rand = self.random_vec.view(1, 1, -1).expand(patches.shape[0], HW, -1)
            return torch.cat([patches, cls_b, rand], dim=-1)

        raise ValueError(self.variant)

    def forward(self, batch: Dict[str, torch.Tensor], out_hw: int = 224):
        f = self.build_feature(batch)                                # [B, HW, D_in]
        B, HW, D_in = f.shape
        f = f.transpose(1, 2).reshape(B, D_in, self.grid, self.grid)  # [B, D_in, H, W]
        log_d = self.head(f)                                         # [B, 1, H, W]
        log_d_full = F.interpolate(log_d, size=(out_hw, out_hw),
                                    mode="bilinear", align_corners=False)
        out = {"log_depth": log_d, "log_depth_full": log_d_full}
        if self.variant in ("D4", "D6", "D7"):
            out["routing"] = self.routing.routing_weights()
        elif self.variant == "D5":
            out["routing"] = self.routing.routing_weights(patches=batch["patches"])
        return out
