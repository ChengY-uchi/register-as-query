"""KITTI depth-selection val (1000 images) loader for outdoor depth experiments.

After unzipping `data_depth_selection.zip` the layout is:
    depth_selection/val_selection_cropped/
        image/             1000 RGB images (1216x352, .png)
        groundtruth_depth/ 1000 GT depth images (.png, uint16, mm)
        intrinsics/        per-image camera intrinsics (.txt)

Depth encoding: 16-bit PNG, depth_m = pixel_value / 256.0.  Pixels with
value 0 are INVALID (sparse LIDAR projection has many holes).

Pipeline:
    PIL load → center-crop to a square at the smaller side (352×352)
    → resize to 518 × 518 (bicubic for image, nearest for depth)
    → ImageNet normalize image
    → mask depth in [0.5, 80] meters (KITTI standard)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


IMG_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_DEPTH_K = 0.5
MAX_DEPTH_K = 80.0       # KITTI standard


def _list_kitti_pairs(root: Path):
    img_dir = root / "depth_selection" / "val_selection_cropped" / "image"
    dep_dir = root / "depth_selection" / "val_selection_cropped" / "groundtruth_depth"
    img_files = sorted(img_dir.glob("*.png"))
    pairs = []
    for f in img_files:
        # filenames match between image and depth folders (just different suffix)
        # image:  2011_09_26_drive_0002_sync_image_0000000005_image_02.png
        # depth:  2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png
        stem = f.name.replace("image", "groundtruth_depth", 1)
        d = dep_dir / stem
        if d.exists():
            pairs.append((f, d))
    return pairs


def joint_kitti_transform(img: Image.Image, dep_pil: Image.Image):
    """Center-crop image+depth to square at shorter side, then resize to
    518×518.  Image: bilinear; depth: nearest (preserves invalid 0s)."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    dep_pil = dep_pil.crop((left, top, left + s, top + s))

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    dep_pil = dep_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

    img_t = TF.to_tensor(img)
    img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
    dep_np = np.asarray(dep_pil, dtype=np.float32) / 256.0       # uint16 → meters
    dep_t = torch.from_numpy(dep_np)
    return img_t, dep_t


class KittiDepthDataset(Dataset):
    def __init__(self, root: Path, indices: List[int]):
        self.root = Path(root)
        self.pairs = _list_kitti_pairs(self.root)
        self.indices = list(indices)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        f_img, f_dep = self.pairs[idx]
        img = Image.open(f_img).convert("RGB")
        dep = Image.open(f_dep)
        assert dep.mode in ("I", "I;16"), f"unexpected depth mode {dep.mode}"
        img_t, dep_t = joint_kitti_transform(img, dep)
        return {"index": int(idx), "image": img_t, "depth": dep_t}


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


def valid_mask_k(depth: torch.Tensor) -> torch.Tensor:
    return (depth > MIN_DEPTH_K) & (depth < MAX_DEPTH_K)
