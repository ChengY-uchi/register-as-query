"""Pre-extract NYU Depth V2 features at 518×518 resolution from 4 ViT
layers of DINOv2-L-reg (frozen), for use with the DPT decoder.

Layers extracted: [5, 11, 17, 23]  (shallow → deep), as in DPT-Large
configuration. At 518 input we get 37×37 = 1369 patch tokens per layer,
plus 1 cls + 4 reg = 1374 tokens per layer.

Cache file (per split):
    indices       : Long [N]
    tokens_L5     : fp16 [N, 1374, 1024]
    tokens_L11    : fp16 [N, 1374, 1024]
    tokens_L17    : fp16 [N, 1374, 1024]
    tokens_L23    : fp16 [N, 1374, 1024]
    depths        : fp16 [N, 518, 518]   (meters, 0 = invalid)
    meta          : dict

Size estimate (per split, ~1160 images):
    4 × 1160 × 1374 × 1024 × 2  ≈  12.7 GB / split.
Per-layer cache could be split into separate files if total disk is tight.
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.backbone import FrozenDinoReg, NUM_REGISTERS, EMBED_DIM


# ----------------------- 518×518 dataset -----------------------

IMG_SIZE_518 = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def joint_eval_transform_518(image: Image.Image, depth: np.ndarray):
    """Resize-shorter-side=518 → center-crop 518; bilinear for both image
    and depth (depth is float — bilinear is acceptable; finer than NN at
    larger scale)."""
    w, h = image.size
    scale = IMG_SIZE_518 / min(w, h)
    new_w = int(round(w * scale)); new_h = int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)
    d_pil = Image.fromarray(depth.astype(np.float32))
    d_resized = d_pil.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - IMG_SIZE_518) // 2
    top = (new_h - IMG_SIZE_518) // 2
    image = image.crop((left, top, left + IMG_SIZE_518, top + IMG_SIZE_518))
    d_resized = d_resized.crop((left, top, left + IMG_SIZE_518, top + IMG_SIZE_518))

    import torchvision.transforms.functional as TF
    img_t = TF.to_tensor(image)
    img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
    depth_t = torch.from_numpy(np.asarray(d_resized, dtype=np.float32))
    return img_t, depth_t


class NyuDepth518Dataset(Dataset):
    def __init__(self, mat_path: Path, indices, preload: bool = True):
        self.mat_path = Path(mat_path)
        self.indices = list(indices)
        self.preload = preload
        self._images = self._depths = None
        if preload:
            with h5py.File(self.mat_path, "r") as f:
                self._images = np.asarray(f["images"][:])     # (1449, 3, 640, 480)
                self._depths = np.asarray(f["depths"][:])     # (1449, 640, 480)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        if self.preload:
            img = self._images[idx]; dep = self._depths[idx]
        else:
            with h5py.File(self.mat_path, "r") as f:
                img = f["images"][idx]; dep = f["depths"][idx]
        # (3, 640, 480) → (480, 640, 3)
        img = img.transpose(2, 1, 0)
        dep = dep.T            # (480, 640)
        img_pil = Image.fromarray(img)
        x, d = joint_eval_transform_518(img_pil, dep)
        return {"index": int(idx), "image": x, "depth": d}


def collate(batch):
    return {
        "indices": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "images":  torch.stack([b["image"] for b in batch]),
        "depths":  torch.stack([b["depth"] for b in batch]),
    }


# ----------------------- multi-layer hook extraction -----------------------

@torch.no_grad()
def extract_multilayer(model, x, layers):
    """Forward DINOv2 once with hooks on multiple layers; returns
    dict[layer] → tokens [B, N, D] (post `model.norm`)."""
    cache = {}
    handles = []
    for L in layers:
        def make_hook(LL):
            def hook(_m, _i, o): cache[LL] = o
            return hook
        handles.append(model.blocks[L].register_forward_hook(make_hook(L)))
    try:
        _ = model(x)
    finally:
        for h in handles: h.remove()
    out = {}
    for L in layers:
        out[L] = model.norm(cache[L])
    return out


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 11, 17, 23])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    from nyu_dataset import stratified_80_20

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx = stratified_80_20(1449, seed=args.split_seed)
    print(f"[518-feat] split:  train={len(train_idx)}  test={len(test_idx)}",
          flush=True)

    backbone = FrozenDinoReg(device=args.device)
    layers = sorted(args.layers)
    NUM_TOK = 1 + NUM_REGISTERS + (518 // 14) ** 2     # 1374

    for split_name, idxs in [("train", train_idx), ("test", test_idx)]:
        out_path = out_dir / f"nyu_features_{split_name}_518_L{'_'.join(str(L) for L in layers)}.pt"
        if out_path.exists():
            print(f"[518-feat] cache exists: {out_path}  (skip)", flush=True)
            continue

        ds = NyuDepth518Dataset(args.mat, idxs, preload=True)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=True)

        N = len(ds)
        tok = {L: torch.empty((N, NUM_TOK, EMBED_DIM), dtype=torch.float16)
               for L in layers}
        dep = torch.empty((N, 518, 518), dtype=torch.float16)
        idx_buf = torch.empty((N,), dtype=torch.long)

        cur = 0
        t0 = time.time()
        for step, batch in enumerate(dl):
            x = batch["images"].to(args.device, non_blocking=True)
            tokens = extract_multilayer(backbone.model, x, layers)
            b = x.shape[0]
            for L in layers:
                tok[L][cur:cur+b] = tokens[L].to("cpu", dtype=torch.float16)
            dep[cur:cur+b] = batch["depths"].to(torch.float16)
            idx_buf[cur:cur+b] = batch["indices"]
            cur += b
            if step % 5 == 0:
                el = time.time() - t0
                eta = el / max(step + 1, 1) * (len(dl) - step - 1)
                print(f"[518-feat] {split_name}  {step+1}/{len(dl)}  cur={cur}/{N}  "
                      f"el={el:.1f}s  eta={eta:.1f}s", flush=True)
        assert cur == N

        cache = {
            "indices": idx_buf,
            "depths":  dep,
            "meta":    {"backbone": "dinov2_vitl14_reg", "layers": layers,
                        "img_size": 518, "split": split_name,
                        "split_seed": args.split_seed},
        }
        for L in layers:
            cache[f"tokens_L{L}"] = tok[L]
        torch.save(cache, out_path)
        sz = out_path.stat().st_size / 1e9
        print(f"[518-feat] wrote {out_path}  ({sz:.2f} GB)  in {time.time()-t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
