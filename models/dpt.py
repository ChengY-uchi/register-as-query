"""DPT (Dense Prediction Transformer, Ranftl et al. 2021) decoder on top of
frozen DINOv2-L-reg, with register-aware Readout modules for variants
D1..D7.

Closely follows Intel's reference implementation (isl-org/DPT):
    - Readout per layer (ProjectReadout style for variants with extra tokens)
    - Reassemble each layer to a target stride via bilinear resize
    - 4-stage RefineNet-like FeatureFusionBlock from deepest → shallowest
    - 3×3 → 2× upsample → 3×3 → 1×1 final head

Variant integration is at the **Readout** stage; the rest of the decoder is
shared so every variant has identical decoder capacity.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seg_heads import SpatialRoutingMLP


EMBED_DIM = 1024            # DINOv2-L
NUM_REG = 4
DECODER_DIM = 256           # standard DPT-Large hidden dim
PATCH_SIZE = 14
DEFAULT_GRID = 37           # 518 / 14
DPT_VARIANTS = ("D1", "D2", "D3", "D4", "D7", "DF", "DQ", "DQC", "DQ_LQ", "DQ_FRQ")
# DQ = D1 base + register-as-query branch.  Registers ARE queries,
#       patches are keys/values; 4 cross-attention spatial masks combined
#       into a residual added to D1 DPT output.  Fundamentally different
#       integration than concat-readout or FiLM modulation.


# ----------------------- residual blocks -----------------------

class ResidualConvUnit(nn.Module):
    def __init__(self, dim: int, use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn
        bias = not use_bn
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=bias)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(dim)
            self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        res = x
        out = self.act(x)
        out = self.conv1(out)
        if self.use_bn: out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.use_bn: out = self.bn2(out)
        return res + out


class FeatureFusionBlock(nn.Module):
    """RefineNet-style fusion: `forward(path_deep, [skip_shallow])` returns
    a feature at 2× the deeper resolution."""
    def __init__(self, dim: int, use_bn: bool = False):
        super().__init__()
        self.res_conf_unit1 = ResidualConvUnit(dim, use_bn=use_bn)
        self.res_conf_unit2 = ResidualConvUnit(dim, use_bn=use_bn)
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, *xs):
        out = xs[0]
        if len(xs) == 2:
            skip = xs[1]
            # patch_size=14 with input 518 yields strides {4,8,16,32} that
            # aren't a perfect 2x cascade — resize skip to match path's HxW.
            if skip.shape[-2:] != out.shape[-2:]:
                skip = F.interpolate(skip, size=out.shape[-2:],
                                      mode="bilinear", align_corners=False)
            out = out + self.res_conf_unit1(skip)
        out = self.res_conf_unit2(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.out_conv(out)
        return out


# ----------------------- variant-aware readout -----------------------

class ProjectReadout(nn.Module):
    """Map per-layer ViT tokens [B, 1+R+P, D] to per-patch features [B, P, D].

    Variant-specific:
        D1: identity on patches
        D2: project [patch ; cls]   2D → D
        D3: project [patch ; reg_mean]   2D → D
        D4: project [patch ; reg_routed]  2D → D  (position-only routing)
        D7: project [patch ; cls ; reg_routed]  3D → D
    """

    def __init__(self, variant: str, embed_dim: int = EMBED_DIM,
                 num_reg: int = NUM_REG, grid: int = DEFAULT_GRID):
        super().__init__()
        assert variant in DPT_VARIANTS
        self.variant = variant
        self.num_reg = num_reg

        in_mult = {"D1": 1, "D2": 2, "D3": 2, "D4": 2, "D7": 3, "DF": 1, "DQ": 1, "DQC": 2,
                   "DQ_LQ": 1, "DQ_FRQ": 1}[variant]
        if in_mult > 1:
            self.proj = nn.Sequential(
                nn.Linear(in_mult * embed_dim, embed_dim),
                nn.GELU(),
            )
        else:
            self.proj = None

        if variant in ("D4", "D7"):
            self.routing = SpatialRoutingMLP(grid=grid, content_dim=0)
        else:
            self.routing = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, 1+R+P, D]  →  patches [B, P, D]
        cls = tokens[:, 0]                                  # [B, D]
        regs = tokens[:, 1 : 1 + self.num_reg]              # [B, R, D]
        patches = tokens[:, 1 + self.num_reg :]             # [B, P, D]
        B, P, D = patches.shape

        if self.variant in ("D1", "DF", "DQ", "DQ_LQ", "DQ_FRQ"):
            # DF/DQ/DQ_LQ/DQ_FRQ take register elsewhere
            return patches
        if self.variant in ("D2", "DQC"):
            cls_b = cls.unsqueeze(1).expand(-1, P, -1)
            return self.proj(torch.cat([patches, cls_b], dim=-1))
        if self.variant == "D3":
            reg_mean = regs.mean(dim=1, keepdim=True).expand(-1, P, -1)
            return self.proj(torch.cat([patches, reg_mean], dim=-1))
        if self.variant == "D4":
            reg_ctx, _ = self.routing(regs)
            return self.proj(torch.cat([patches, reg_ctx], dim=-1))
        if self.variant == "D7":
            reg_ctx, _ = self.routing(regs)
            cls_b = cls.unsqueeze(1).expand(-1, P, -1)
            return self.proj(torch.cat([patches, cls_b, reg_ctx], dim=-1))
        raise ValueError(self.variant)


# ----------------------- Reassemble -----------------------

class Reassemble(nn.Module):
    """Token sequence → spatial feature at a target stride.

    Follows the official DINOv2 / DPT design (Ranftl 2021, Sect. 3.2):
        - Conv 1×1 to decoder dim
        - LEARNABLE resize per layer: ConvTranspose (shallow, upsample),
          Identity (mid), or strided Conv (deep, downsample)
        - 3×3 refine conv

    For DINOv2-L-reg @ 518 with patch_size=14, base grid = 37×37.
    Output spatial sizes after resize are 148 / 74 / 37 / 19 — the fusion
    chain handles non-exact 2× cascade via interpolation in its skip path.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, decoder_dim: int = DECODER_DIM,
                 target_stride: int = 4):
        super().__init__()
        self.target_stride = target_stride
        self.proj = nn.Conv2d(embed_dim, decoder_dim, 1)
        if target_stride == 4:                   # shallow: upsample ×4
            self.resize = nn.ConvTranspose2d(decoder_dim, decoder_dim, 4, stride=4)
        elif target_stride == 8:                 # upsample ×2
            self.resize = nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, stride=2)
        elif target_stride == 16:                # identity (already at this stride)
            self.resize = nn.Identity()
        elif target_stride == 32:                # deep: downsample ×2 via stride-2 conv
            self.resize = nn.Conv2d(decoder_dim, decoder_dim, 3, stride=2, padding=1)
        else:
            raise ValueError(f"unsupported target_stride: {target_stride}")
        self.refine = nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1)

    def forward(self, tokens: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
        # tokens: [B, P, D]
        B, P, D = tokens.shape
        grid = int(math.sqrt(P))
        x = tokens.transpose(1, 2).reshape(B, D, grid, grid)        # [B, D, g, g]
        x = self.proj(x)                                            # [B, dd, g, g]
        x = self.resize(x)                                          # learnable resize
        x = self.refine(x)
        return x


# ----------------------- FiLM modulation -----------------------

class FiLMBlock(nn.Module):
    """Per-channel affine modulation conditioned on a context vector.

        gamma, beta = Linear(ctx_dim → 2 * feat_dim)(ctx)
        out = x * (1 + gamma) + beta    (broadcast over H, W)

    Initialized with zero weights so the block is identity at init — any
    learned modulation is purely additive contribution from the register.
    """
    def __init__(self, ctx_dim: int, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(ctx_dim, 2 * feat_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W];  ctx: [B, ctx_dim]
        gb = self.proj(ctx)                                # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)                  # [B, C] each
        B, C = gamma.shape
        return x * (1.0 + gamma.view(B, C, 1, 1)) + beta.view(B, C, 1, 1)


# ----------------------- Register-as-Query branch (DQ) -----------------------

class RegQueryBranch(nn.Module):
    """Mask2Former-style cross-attention branch:
        registers are queries, patches are keys/values.
        4 register queries → 4 spatial masks (via Q @ K^T) → combine → residual.

    Output spatial size = patch grid (G × G). Caller upsamples to target HxW.
    """
    def __init__(self, embed_dim: int = EMBED_DIM,
                 decoder_dim: int = DECODER_DIM,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 grid: int = DEFAULT_GRID,
                 num_queries: int = NUM_REG,
                 out_channels: int = 1):
        super().__init__()
        self.grid = grid
        self.num_queries = num_queries
        self.out_channels = out_channels

        # Project tokens to decoder dim
        self.q_proj = nn.Linear(embed_dim, decoder_dim)
        self.kv_proj = nn.Linear(embed_dim, decoder_dim)

        # Learned spatial positional embedding for patches
        self.pos_emb = nn.Parameter(torch.zeros(1, grid * grid, decoder_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Transformer decoder layers (self-attn on queries + cross-attn to patches + FFN)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=decoder_dim, nhead=num_heads,
                dim_feedforward=4 * decoder_dim, batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])

        # Mask head: per-query → use refined Q to dot with patch K → spatial logit
        self.mask_head = nn.Linear(decoder_dim, decoder_dim)

        # Combine R register masks into `out_channels` residual maps
        # (depth: 1, normal: 3, seg: num_classes)
        self.combine = nn.Conv2d(num_queries, out_channels, kernel_size=1)
        # init to zero → branch outputs zero at init (residual not contributing)
        nn.init.zeros_(self.combine.weight)
        nn.init.zeros_(self.combine.bias)

    def forward(self, registers: torch.Tensor,
                patches: torch.Tensor) -> torch.Tensor:
        # registers: [B, R, D];   patches: [B, P, D]
        B, P, D = patches.shape
        Q = self.q_proj(registers)                        # [B, R, dd]
        K = self.kv_proj(patches) + self.pos_emb          # [B, P, dd]

        for layer in self.layers:
            Q = layer(Q, K)                               # self-attn + cross-attn + FFN

        Q_mask = self.mask_head(Q)                        # [B, R, dd]
        masks = torch.einsum("brd,bpd->brp", Q_mask, K)   # [B, R, P]
        masks = masks.view(B, self.num_queries, self.grid, self.grid)  # [B, R, g, g]

        residual = self.combine(masks)                    # [B, out_channels, g, g]
        return residual


# ----------------------- DPT head -----------------------

class DPTHead(nn.Module):
    """4-layer DPT decoder.

    Input forward:
        forward(tokens_per_layer, input_hw)
            tokens_per_layer: list of 4 tensors [B, 1+R+P, D] (shallow → deep)
            input_hw: original input image (H, W) — e.g., (518, 518)
        Returns:
            [B, 1, H_out, W_out] log-depth or raw-depth (caller decides).
    """

    def __init__(self, variant: str, embed_dim: int = EMBED_DIM,
                 decoder_dim: int = DECODER_DIM, num_reg: int = NUM_REG,
                 grid: int = DEFAULT_GRID, use_bn: bool = True):
        super().__init__()
        assert variant in DPT_VARIANTS
        self.variant = variant
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.num_reg = num_reg

        # one readout per layer (independent params so each layer can specialize)
        self.readouts = nn.ModuleList([
            ProjectReadout(variant, embed_dim=embed_dim, num_reg=num_reg,
                           grid=grid) for _ in range(4)
        ])

        # reassemble: shallow → deep with target strides 4, 8, 16, 32
        target_strides = [4, 8, 16, 32]
        self.reassembles = nn.ModuleList([
            Reassemble(embed_dim, decoder_dim, ts) for ts in target_strides
        ])

        # fusion blocks: applied deep → shallow
        self.fusions = nn.ModuleList([
            FeatureFusionBlock(decoder_dim, use_bn=use_bn) for _ in range(4)
        ])

        # FiLM modulation (variant DF only): one block per fusion stage,
        # conditioned on that layer's register mean.
        self.use_film = (variant == "DF")
        if self.use_film:
            self.films = nn.ModuleList([
                FiLMBlock(ctx_dim=embed_dim, feat_dim=decoder_dim) for _ in range(4)
            ])

        # Register-as-Query branch (variants DQ / DQC / DQ_LQ / DQ_FRQ).  Adds a
        # residual to the final DPT depth map via cross-attention with patches as K/V.
        # DQ:     queries = the 4 input register tokens at L23 (per-image content).
        # DQC:    same as DQ but base readout is patch+cls (D2-style).
        # DQ_LQ:  queries = 4 LEARNABLE parameter vectors (Mask2Former-style);
        #         ablation = "is the architecture useful even without registers?"
        #         (has SGD optimization advantage over DQ — not a clean isolate)
        # DQ_FRQ: queries = 4 FIXED random vectors (sampled at construction time,
        #         never updated).  Clean ablation: tests whether register CONTENT
        #         carries useful info, since both DQ and DQ_FRQ have frozen queries
        #         and no task-specific query optimization.
        # NOTE: residual starts at 0 via combine.weight zero-init inside RegQueryBranch.
        self.use_regquery = variant in ("DQ", "DQC", "DQ_LQ", "DQ_FRQ")
        self.use_learnable_q = (variant == "DQ_LQ")
        self.use_fixed_q = (variant == "DQ_FRQ")
        if self.use_regquery:
            self.reg_query = RegQueryBranch(
                embed_dim=embed_dim, decoder_dim=decoder_dim,
                num_layers=2, num_heads=8, grid=grid, num_queries=num_reg,
            )
        if self.use_learnable_q:
            # 4 learnable query embeddings
            self.learnable_q = nn.Parameter(torch.randn(1, num_reg, embed_dim) * 0.02)
        if self.use_fixed_q:
            # 4 fixed random query embeddings (saved in state_dict, never trained)
            self.register_buffer(
                "fixed_q", torch.randn(1, num_reg, embed_dim) * 0.02
            )

        # project: 3×3 conv + BN before the final head (matches official DPT)
        self.project = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=not use_bn),
            nn.BatchNorm2d(decoder_dim) if use_bn else nn.Identity(),
        )

        # final head: 3×3 → 2× upsample → 3×3 → ReLU → 1×1 → 1 channel
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_dim // 2, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, tokens_per_layer: List[torch.Tensor],
                input_hw: Tuple[int, int]) -> torch.Tensor:
        assert len(tokens_per_layer) == 4

        # per-layer readout → per-layer reassemble
        spatial = []
        for tokens, readout, reasm in zip(tokens_per_layer, self.readouts,
                                           self.reassembles):
            p = readout(tokens)              # [B, P, D]
            f = reasm(p, input_hw)           # [B, dd, H_t, W_t]
            spatial.append(f)
        # spatial[i] strides: 4, 8, 16, 32 (shallow → deep)

        # Pre-compute per-layer register means for FiLM (DF variant only)
        reg_summaries = None
        if self.use_film:
            reg_summaries = [
                tokens_per_layer[i][:, 1 : 1 + self.num_reg].mean(dim=1)
                for i in range(4)
            ]  # list of [B, D]

        # fuse deep → shallow, with optional per-stage FiLM modulation
        path = self.fusions[3](spatial[3])
        if self.use_film: path = self.films[3](path, reg_summaries[3])
        path = self.fusions[2](path, spatial[2])
        if self.use_film: path = self.films[2](path, reg_summaries[2])
        path = self.fusions[1](path, spatial[1])
        if self.use_film: path = self.films[1](path, reg_summaries[1])
        path = self.fusions[0](path, spatial[0])
        if self.use_film: path = self.films[0](path, reg_summaries[0])

        # project (3×3 conv + BN) then final head → stride 1 (full resolution)
        path = self.project(path)
        out = self.head(path)

        # DQ/DQC/DQ_LQ/DQ_FRQ variants: add register-as-query residual
        if self.use_regquery:
            L23_tokens = tokens_per_layer[-1]                 # deepest layer
            patches = L23_tokens[:, 1 + self.num_reg :]       # [B, P, D]
            B = patches.shape[0]
            if self.use_learnable_q:
                queries = self.learnable_q.expand(B, -1, -1)  # [B, R, D] learnable
            elif self.use_fixed_q:
                queries = self.fixed_q.expand(B, -1, -1)      # [B, R, D] frozen random
            else:
                queries = L23_tokens[:, 1 : 1 + self.num_reg]  # [B, R, D] register-derived
            residual = self.reg_query(queries, patches)       # [B, 1, g, g]
            residual = F.interpolate(residual, size=out.shape[-2:],
                                     mode="bilinear", align_corners=False)
            out = out + residual

        return out
