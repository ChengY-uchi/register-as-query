"""Pre-extract L23 token features + keep 224x224 masks for VOC 2012 val set.

Saves one .pt per split: data/voc_features_{split}_L23.pt
containing:
    image_ids : list[str]          (length N)
    cls       : fp16 [N, 1, 1024]
    regs      : fp16 [N, 4, 1024]
    patches   : fp16 [N, 256, 1024]
    masks     : int16 [N, 224, 224]   (values 0..20, 255 as -1 if re-encoded;
                                       here we keep 0..20 and 255 as int16)
    meta      : dict (layer, backbone, img_size)

One-time cost, reused by S1..S4 and every seed.
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.backbone import FrozenDinoReg, NUM_REGISTERS, NUM_PATCHES, EMBED_DIM
from voc_dataset import (
    VocSegDataset, dataset_collate, load_voc_split, stratified_80_20,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc-root",
                    default="datasets/voc2012/VOCdevkit/VOC2012")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    voc_root = Path(args.voc_root)
    if not voc_root.exists():
        raise SystemExit(f"VOC root not found at {voc_root}. Download first.")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val_ids = load_voc_split(voc_root, "val")
    print(f"[voc-feat] VOC val: {len(val_ids)} images", flush=True)
    train_ids, test_ids = stratified_80_20(val_ids, seed=args.split_seed)
    print(f"[voc-feat]   80/20 split:  train={len(train_ids)}  test={len(test_ids)}",
          flush=True)

    backbone = FrozenDinoReg(device=args.device)

    for split_name, ids in [("train", train_ids), ("test", test_ids)]:
        out_path = out_dir / f"voc_features_{split_name}_L{args.layer}.pt"
        if out_path.exists():
            print(f"[voc-feat] cache exists: {out_path} (skip)", flush=True)
            continue

        ds = VocSegDataset(voc_root, ids)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=dataset_collate,
                        pin_memory=True)

        N = len(ds)
        cls_all     = torch.empty((N, 1, EMBED_DIM), dtype=torch.float16)
        regs_all    = torch.empty((N, NUM_REGISTERS, EMBED_DIM), dtype=torch.float16)
        patches_all = torch.empty((N, NUM_PATCHES, EMBED_DIM), dtype=torch.float16)
        masks_all   = torch.empty((N, 224, 224), dtype=torch.int16)
        ids_out: list = []

        cur = 0
        t0 = time.time()
        for step, batch in enumerate(dl):
            x = batch["images"].to(args.device, non_blocking=True)
            toks = backbone.extract_layer_tokens(x, layer_idx=args.layer)
            b = x.shape[0]
            cls_all[cur:cur+b]     = toks["cls"].to("cpu", dtype=torch.float16)
            regs_all[cur:cur+b]    = toks["regs"].to("cpu", dtype=torch.float16)
            patches_all[cur:cur+b] = toks["patches"].to("cpu", dtype=torch.float16)
            masks_all[cur:cur+b]   = batch["masks"].to(torch.int16)
            ids_out.extend(batch["image_ids"])
            cur += b
            if step % 10 == 0:
                el = time.time() - t0
                eta = el / max(step + 1, 1) * (len(dl) - step - 1)
                print(f"[voc-feat] {split_name}  {step+1}/{len(dl)}  cur={cur}/{N}  "
                      f"elapsed={el:.1f}s  eta={eta:.1f}s", flush=True)
        assert cur == N

        cache = {
            "image_ids": ids_out,
            "cls":       cls_all,
            "regs":      regs_all,
            "patches":   patches_all,
            "masks":     masks_all,
            "meta":      {"backbone": "dinov2_vitl14_reg", "layer": args.layer,
                          "img_size": 224, "num_classes": 21,
                          "split": split_name, "split_seed": args.split_seed},
        }
        torch.save(cache, out_path)
        sz = out_path.stat().st_size / 1e9
        print(f"[voc-feat] wrote {out_path}  ({sz:.2f} GB)  in {time.time()-t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
