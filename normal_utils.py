"""Surface normal helpers — derive GT normals from cached NYU depths via finite
differences, plus standard angular-error metrics.

NYU labeled-depth camera intrinsics (full 640×480, in pixels):
    fx_d = 582.62  fy_d = 582.69  cx_d = 313.05  cy_d = 238.44
After resize-shorter-side-to-256 (scale = 256/480 ≈ 0.5333) + center-crop to
224, the *effective* focal length on the model-input image becomes:
    fx_resize = 582.62 * 256/480 ≈ 310.7
This is the value we plug into the cross-product when forming GT normals.

For ablation purposes the absolute correctness of normals is irrelevant — we
only need a consistent GT shared across N1..N7.  The same depth-to-normal
recipe is applied to every example.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# NYU labeled intrinsics (depth-frame, native 640x480), scaled to 256-shorter-side
FX_RESIZED = 582.624 * 256.0 / 480.0
FY_RESIZED = 582.692 * 256.0 / 480.0


def depth_to_normal(depth: torch.Tensor,
                    fx: float = FX_RESIZED,
                    fy: float = FY_RESIZED,
                    min_depth: float = 0.5,
                    max_depth: float = 10.0,
                    grad_thresh: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """depth: [B, H, W] meters.

    Returns:
        normal : [B, H, W, 3] unit vectors, normalized.
        mask   : [B, H, W] bool — valid pixels (in-range depth, away from
                 image border, low local depth gradient = excludes object
                 boundaries / depth discontinuities).
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    B, H, W = depth.shape

    # central differences (pad zeros at borders; mask covers border anyway)
    dx = torch.zeros_like(depth)
    dy = torch.zeros_like(depth)
    dx[:, :, 1:-1] = (depth[:, :, 2:] - depth[:, :, :-2]) * 0.5
    dy[:, 1:-1, :] = (depth[:, 2:, :] - depth[:, :-2, :]) * 0.5

    # Normal direction in NYU/OpenCV convention (camera looks at +Z, image y down):
    #   n  ∝  ( -fx * dz/dx,  -fy * dz/dy,  1 )
    nx = -fx * dx
    ny = -fy * dy
    nz = torch.ones_like(depth)
    n = torch.stack([nx, ny, nz], dim=-1)                      # [B, H, W, 3]
    n_norm = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    # mask out edges, invalid depth, depth-discontinuity pixels
    valid = (depth > min_depth) & (depth < max_depth)
    border = torch.zeros_like(depth, dtype=torch.bool)
    border[:, 1:-1, 1:-1] = True
    grad_mag = torch.sqrt(dx * dx + dy * dy)
    flat = grad_mag < grad_thresh
    mask = valid & border & flat
    return n_norm, mask


def angular_metrics(pred_unit: torch.Tensor, gt_unit: torch.Tensor,
                    mask: torch.Tensor) -> dict:
    """pred_unit, gt_unit: [B, H, W, 3] unit-normed.  mask: [B, H, W] bool.

    Returns standard surface-normal metrics: mean / median angular error in
    degrees, and accuracy under {11.25°, 22.5°, 30°} thresholds.
    """
    cos = (pred_unit * gt_unit).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    ang_deg = torch.acos(cos) * (180.0 / np.pi)
    sel = ang_deg[mask]
    return {
        "mean_ang_err":   float(sel.mean().item()),
        "median_ang_err": float(sel.median().item()),
        "acc_11_25":      float((sel < 11.25).float().mean().item()),
        "acc_22_5":       float((sel < 22.5).float().mean().item()),
        "acc_30":         float((sel < 30.0).float().mean().item()),
    }


def cosine_loss(pred: torch.Tensor, gt_unit: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """pred: [B, 3, H, W] (unnormed). gt_unit: [B, H, W, 3]. mask: [B, H, W]."""
    pred = pred.permute(0, 2, 3, 1)
    pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    cos = (pred_n * gt_unit).sum(dim=-1)
    n = mask.float().sum().clamp(min=1.0)
    return ((1.0 - cos) * mask).sum() / n
