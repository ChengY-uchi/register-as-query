"""Segmentation heads for the 4 VOC variants.

All variants share a common forward template:

    1.  per-patch feature  f [B, P=256, D_input]
    2.  reshape            f -> [B, D_input, H=16, W=16]
    3.  1x1 conv           f -> logits [B, 21, 16, 16]
    4.  bilinear upsample  to [B, 21, 224, 224]

Only the construction of `f` differs across variants:

    S1 : f = patch                            D_input = D
    S2 : f = [patch; cls_broadcast]           D_input = 2D
    S3 : f = [patch; reg_mean_broadcast]      D_input = 2D
    S4 : f = [patch; reg_ctx_per_patch]       D_input = 2D
         where reg_ctx = sum_k(alpha_k(i,j) * reg_k),
         alpha = softmax(MLP(sinusoidal_pos(i, j))),
         i.e. per-patch routing over 4 registers that depends ONLY on
         position (image-invariant). This is the critical variant.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S_BN", "S_DQ")
# S_BN  = official-style BNHead baseline: BN(patches) + 1×1 cls_seg
# S_DQ  = S_BN + register-as-query cross-attn residual (out_channels=num_classes)
EMBED_DIM = 1024
PATCH_GRID = 16
NUM_REG = 4


# -------------------- spatial routing module (S4) --------------------

def sinusoidal_2d_pos(grid: int, dim: int) -> torch.Tensor:
    """Returns a [grid, grid, dim] 2D sinusoidal positional embedding."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin/cos embedding"
    d_each = dim // 4
    freqs = torch.exp(
        torch.arange(d_each).float() * (-math.log(10000.0) / max(d_each - 1, 1))
    )                                    # [d_each]
    pos = torch.arange(grid).float()
    sx = torch.sin(pos.unsqueeze(1) * freqs.unsqueeze(0))   # [grid, d_each]
    cx = torch.cos(pos.unsqueeze(1) * freqs.unsqueeze(0))
    emb_x = torch.cat([sx, cx], dim=-1)                      # [grid, 2*d_each]
    sy = torch.sin(pos.unsqueeze(1) * freqs.unsqueeze(0))
    cy = torch.cos(pos.unsqueeze(1) * freqs.unsqueeze(0))
    emb_y = torch.cat([sy, cy], dim=-1)                      # [grid, 2*d_each]
    # Outer-combine x and y (broadcast-concat)
    Ex = emb_x.unsqueeze(0).expand(grid, -1, -1)             # [grid, grid, 2*d_each]
    Ey = emb_y.unsqueeze(1).expand(-1, grid, -1)             # [grid, grid, 2*d_each]
    return torch.cat([Ey, Ex], dim=-1)                       # [grid, grid, dim]


class SpatialRoutingMLP(nn.Module):
    """Per-patch routing over R registers.

    content_dim is None (S4): position-only routing, image-invariant,
        weights shape [H, W, R].

    content_dim > 0 (S5): content-aware routing. Each patch contributes its
        own feature to the router, so weights vary per-image:
        weights shape [B, H, W, R].
    """

    def __init__(self, grid: int = PATCH_GRID, pos_dim: int = 128,
                 hidden: int = 64, num_regs: int = NUM_REG,
                 content_dim: int = 0):
        super().__init__()
        self.grid = grid
        self.num_regs = num_regs
        self.content_dim = content_dim
        in_dim = pos_dim + content_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_regs),
        )
        # zero-init output bias so softmax starts uniform (0.25 each)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

        pos_emb = sinusoidal_2d_pos(grid, pos_dim)                # [H, W, pos_dim]
        self.register_buffer("pos_emb", pos_emb)

    def routing_weights(self, patches: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.content_dim == 0:
            logits = self.mlp(self.pos_emb)                        # [H, W, R]
            return torch.softmax(logits, dim=-1)
        assert patches is not None and patches.shape[-1] == self.content_dim
        B, HW, _ = patches.shape
        assert HW == self.grid * self.grid
        pos_flat = self.pos_emb.reshape(HW, -1)                    # [HW, pos_dim]
        pos_flat = pos_flat.unsqueeze(0).expand(B, -1, -1)         # [B, HW, pos_dim]
        inp = torch.cat([pos_flat, patches], dim=-1)               # [B, HW, pos_dim + D]
        logits = self.mlp(inp)                                     # [B, HW, R]
        logits = logits.reshape(B, self.grid, self.grid, self.num_regs)
        return torch.softmax(logits, dim=-1)

    def forward(self, regs: torch.Tensor,
                patches: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """regs: [B, R, D]  ->  (reg_ctx [B, HW, D], weights [H,W,R] or [B,H,W,R])."""
        w = self.routing_weights(patches)
        if w.dim() == 3:                              # [H, W, R]  (S4 path)
            ctx = torch.einsum("hwk,bkd->bhwd", w, regs)
        else:                                         # [B, H, W, R] (S5 path)
            ctx = torch.einsum("bhwk,bkd->bhwd", w, regs)
        ctx = ctx.reshape(regs.shape[0], self.grid * self.grid, -1)
        return ctx, w


# -------------------- unified model --------------------

class SegHead(nn.Module):
    def __init__(self, variant: str, embed_dim: int = EMBED_DIM,
                 num_classes: int = 21, grid: int = PATCH_GRID,
                 dq_decoder_dim: int = 256):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        self.grid = grid
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        D_in = {"S1": embed_dim, "S2": 2 * embed_dim, "S3": 2 * embed_dim,
                "S4": 2 * embed_dim, "S5": 2 * embed_dim,
                "S6": embed_dim,
                "S7": 3 * embed_dim,
                "S_BN": embed_dim, "S_DQ": embed_dim}[variant]
        self.conv1x1 = nn.Conv2d(D_in, num_classes, kernel_size=1)

        # Official BNHead-style normalization (S_BN / S_DQ)
        self.use_bn = variant in ("S_BN", "S_DQ")
        if self.use_bn:
            self.bn = nn.BatchNorm2d(embed_dim)

        # Register-as-query residual branch (S_DQ only)
        self.use_dq = (variant == "S_DQ")
        if self.use_dq:
            # local import to avoid circular dependency through models/__init__.py
            from .dpt import RegQueryBranch
            self.reg_query = RegQueryBranch(
                embed_dim=embed_dim, decoder_dim=dq_decoder_dim,
                num_layers=2, num_heads=8, grid=grid,
                num_queries=NUM_REG, out_channels=num_classes,
            )

        self.routing: Optional[SpatialRoutingMLP] = None
        if variant == "S4":
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=0)
        elif variant == "S5":
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=embed_dim,
                                              hidden=32)
        elif variant == "S6":
            # register-only: position-routed reg_ctx, no patches in feature
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=0)
        elif variant == "S7":
            # [patch; cls; reg_routed]: CLS = global class, reg_routed = per-region
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=0)

    def build_feature(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns per-patch feature [B, HW, D_in] for the chosen variant."""
        cls     = batch["cls"]        # [B, 1, D]
        regs    = batch["regs"]       # [B, 4, D]
        patches = batch["patches"]    # [B, HW, D]
        B, HW, D = patches.shape

        if self.variant in ("S1", "S_BN", "S_DQ"):
            return patches

        if self.variant == "S2":
            return torch.cat([patches, cls[:, 0:1].expand(-1, HW, -1)], dim=-1)

        if self.variant == "S3":
            reg_mean = regs.mean(dim=1, keepdim=True)              # [B, 1, D]
            return torch.cat([patches, reg_mean.expand(-1, HW, -1)], dim=-1)

        if self.variant == "S4":
            assert self.routing is not None
            reg_ctx, _w = self.routing(regs)                       # [B, HW, D]
            return torch.cat([patches, reg_ctx], dim=-1)

        if self.variant == "S5":
            assert self.routing is not None
            reg_ctx, _w = self.routing(regs, patches=patches)      # [B, HW, D]
            return torch.cat([patches, reg_ctx], dim=-1)

        if self.variant == "S6":
            # register-only diagnostic: position-routed reg context, NO patches
            assert self.routing is not None
            reg_ctx, _w = self.routing(regs)                       # [B, HW, D]
            return reg_ctx

        if self.variant == "S7":
            # [patch; cls_broadcast; reg_routed]: decouple global class + per-region
            assert self.routing is not None
            reg_ctx, _w = self.routing(regs)                       # [B, HW, D]
            cls_b = cls[:, 0:1].expand(-1, HW, -1)                 # [B, HW, D]
            return torch.cat([patches, cls_b, reg_ctx], dim=-1)    # [B, HW, 3D]

        raise ValueError(self.variant)

    def forward(self, batch: Dict[str, torch.Tensor],
                out_hw: int = 224) -> Dict[str, torch.Tensor]:
        """Returns dict with 'logits_full' [B, C, H_out, H_out] and optionally
        'routing' [H, W, R] for S4 (image-invariant spatial routing)."""
        f = self.build_feature(batch)                              # [B, HW, D_in]
        B, HW, D_in = f.shape
        f = f.transpose(1, 2).reshape(B, D_in, self.grid, self.grid)   # [B, D_in, H, W]
        if self.use_bn:
            f = self.bn(f)
        logits = self.conv1x1(f)                                   # [B, C, H, W]
        if self.use_dq:
            # register-as-query cross-attn residual (init at 0 via combine zero-init)
            residual = self.reg_query(batch["regs"], batch["patches"])  # [B, C, H, W]
            logits = logits + residual
        logits_full = F.interpolate(
            logits, size=(out_hw, out_hw), mode="bilinear", align_corners=False,
        )
        out = {"logits": logits, "logits_full": logits_full}
        if self.variant in ("S4", "S6"):
            out["routing"] = self.routing.routing_weights()        # [H, W, R]
        elif self.variant == "S5":
            # per-image routing needs patches
            out["routing"] = self.routing.routing_weights(patches=batch["patches"])
        return out
