"""Frozen DINOv2-L-reg backbone with layer-K token extraction.

DINOv2-reg token layout (after prepare_tokens_with_masks):
    [CLS, reg_1, reg_2, reg_3, reg_4, patch_1, ..., patch_256]
    index 0,   1:1+R,                              1+R:1+R+P
With R=4 registers and P=256 patches (224/14)^2, total N = 261.

extract_layer_tokens(x, layer_idx) runs one forward pass, captures
blocks[layer_idx]'s output via a forward hook, applies model.norm (matching
DINOv2's get_intermediate_layers(norm=True) convention), and splits into
(cls, regs, patches) tensors.
"""

from __future__ import annotations

import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_REGISTERS = 4
NUM_PATCHES = 256      # (224 / 14) ** 2
EMBED_DIM = 1024
NUM_BLOCKS = 24


class FrozenDinoReg(nn.Module):
    """Wrapper around dinov2_vitl14_reg that exposes mid-layer token features."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        model = torch.hub.load(
            str(REPO_ROOT), "dinov2_vitl14_reg", source="local"
        ).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.device = device

        # sanity
        assert model.num_register_tokens == NUM_REGISTERS
        assert model.embed_dim == EMBED_DIM
        assert len(model.blocks) == NUM_BLOCKS

    @torch.no_grad()
    def extract_layer_tokens(self, x: torch.Tensor, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Run one forward, capture blocks[layer_idx] output, apply model.norm,
        return dict with cls [B,1,D], regs [B,R,D], patches [B,P,D]."""
        assert 0 <= layer_idx < NUM_BLOCKS
        cache: Dict[str, torch.Tensor] = {}

        def hook(module, inp, out):
            cache["out"] = out

        h = self.model.blocks[layer_idx].register_forward_hook(hook)
        try:
            _ = self.model(x)
        finally:
            h.remove()

        out = cache["out"]                       # [B, N, D] pre-final-norm
        normed = self.model.norm(out)            # [B, N, D] post-final-norm
        cls = normed[:, 0:1, :]                  # [B, 1, D]
        regs = normed[:, 1 : 1 + NUM_REGISTERS]  # [B, R, D]
        patches = normed[:, 1 + NUM_REGISTERS :] # [B, P, D]
        assert patches.shape[1] == NUM_PATCHES, patches.shape
        return {"cls": cls, "regs": regs, "patches": patches}
