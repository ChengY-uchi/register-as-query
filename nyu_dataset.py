"""NYU Depth V2 labeled dataset + joint image/depth transforms.

.mat file (HDF5 v7.3) contains (indexed along leading dim N=1449):
    images  : (1449, 3, 640, 480) uint8    — note: (N, C, W, H), MATLAB col-major
    depths  : (1449, 640, 480)    float32  — meters, 0 = invalid / missing

Native image resolution is 640 × 480 (landscape). For DINOv2 eval we resize
the shorter side (480) to 256, centercrop 224 × 224. Depth mask is resized
with NEAREST, the float depth with BILINEAR (so small invalid blobs don't
expand).

Valid-depth mask: depth > 0.5 m and depth < 10.0 m (NYU standard clip).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


IMG_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_DEPTH = 0.5        # meters
MAX_DEPTH = 10.0


def _orient_image(img_arr: np.ndarray) -> np.ndarray:
    """(3, 640, 480) -> (480, 640, 3) uint8 HWC."""
    return img_arr.transpose(2, 1, 0)


def _orient_depth(d: np.ndarray) -> np.ndarray:
    """(640, 480) -> (480, 640) float32."""
    return d.T


def joint_eval_transform(image: Image.Image, depth: np.ndarray):
    """Apply DINOv2 resize(shorter=256)+centercrop(224) jointly to image and depth.
    depth input is numpy (H, W) float32 meters; output is tensor."""
    w, h = image.size
    scale = RESIZE_SIZE / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # depth: same resize via PIL (bilinear is acceptable for depth, we keep
    # nearest-neighbor for validity mask below)
    d_pil = Image.fromarray(depth.astype(np.float32))
    d_resized = d_pil.resize((new_w, new_h), Image.BILINEAR)

    # centercrop to IMG_SIZE
    left = (new_w - IMG_SIZE) // 2
    top  = (new_h - IMG_SIZE) // 2
    image = image.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    d_resized = d_resized.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    img_t = TF.to_tensor(image)
    img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
    depth_t = torch.from_numpy(np.asarray(d_resized, dtype=np.float32))
    return img_t, depth_t


class NyuDepthDataset(Dataset):
    """Loads NYU labeled depth samples with DINOv2 eval transform."""

    def __init__(self, mat_path: Path, indices: List[int],
                 preload: bool = True):
        self.mat_path = Path(mat_path)
        self.indices = list(indices)
        self.preload = preload
        self._images = None
        self._depths = None

        if preload:
            # Keep whole dataset in RAM: ~3 GB. Fast random access.
            with h5py.File(self.mat_path, "r") as f:
                self._images = np.asarray(f["images"][:])      # (1449, 3, 640, 480)
                self._depths = np.asarray(f["depths"][:])      # (1449, 640, 480)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        if self.preload:
            img = _orient_image(self._images[idx])
            depth = _orient_depth(self._depths[idx])
        else:
            with h5py.File(self.mat_path, "r") as f:
                img = _orient_image(f["images"][idx])
                depth = _orient_depth(f["depths"][idx])
        image_pil = Image.fromarray(img)
        img_t, depth_t = joint_eval_transform(image_pil, depth)
        return {"index": int(idx), "image": img_t, "depth": depth_t}


def collate(batch):
    return {
        "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "images":  torch.stack([b["image"] for b in batch]),
        "depths":  torch.stack([b["depth"] for b in batch]),
    }


def stratified_80_20(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * 0.8)
    return idx[:cut].tolist(), idx[cut:].tolist()


def valid_mask(depth: torch.Tensor) -> torch.Tensor:
    """Returns a bool mask where depth is in [MIN_DEPTH, MAX_DEPTH]."""
    return (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
