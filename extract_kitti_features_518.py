"""Extract 518×518 4-layer DINOv2-L-reg features for KITTI depth_selection val.

Same structure as extract_nyu_features_518.py but for KITTI dataset.
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.backbone import FrozenDinoReg, NUM_REGISTERS, EMBED_DIM
from kitti_dataset import KittiDepthDataset, collate, stratified_80_20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/kitti")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 11, 17, 23])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1000 KITTI val_selection images
    ds_probe = KittiDepthDataset(args.root, list(range(1000)))
    N_total = len(ds_probe.pairs)
    print(f"[kitti-feat] found {N_total} KITTI image/depth pairs", flush=True)

    train_idx, test_idx = stratified_80_20(N_total, seed=args.split_seed)
    print(f"[kitti-feat] split:  train={len(train_idx)}  test={len(test_idx)}",
          flush=True)

    backbone = FrozenDinoReg(device=args.device)
    layers = sorted(args.layers)
    NUM_TOK = 1 + NUM_REGISTERS + (518 // 14) ** 2

    for split_name, idxs in [("train", train_idx), ("test", test_idx)]:
        out_path = out_dir / f"kitti_features_{split_name}_518_L{'_'.join(str(L) for L in layers)}.pt"
        if out_path.exists():
            print(f"[kitti-feat] cache exists: {out_path}  (skip)", flush=True)
            continue

        ds = KittiDepthDataset(args.root, idxs)
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
            cache: dict = {}
            handles = []
            for L in layers:
                def make_hook(LL):
                    def hook(_m, _i, o): cache[LL] = o
                    return hook
                handles.append(backbone.model.blocks[L].register_forward_hook(make_hook(L)))
            try:
                with torch.no_grad():
                    _ = backbone.model(x)
            finally:
                for h in handles: h.remove()
            b = x.shape[0]
            for L in layers:
                normed = backbone.model.norm(cache[L])
                tok[L][cur:cur+b] = normed.to("cpu", dtype=torch.float16)
            dep[cur:cur+b] = batch["depths"].to(torch.float16)
            idx_buf[cur:cur+b] = batch["indices"]
            cur += b
            if step % 5 == 0:
                el = time.time() - t0
                eta = el / max(step + 1, 1) * (len(dl) - step - 1)
                print(f"[kitti-feat] {split_name}  {step+1}/{len(dl)}  cur={cur}/{N}  "
                      f"el={el:.1f}s  eta={eta:.1f}s", flush=True)
        assert cur == N

        meta = {"backbone": "dinov2_vitl14_reg", "layers": layers,
                "img_size": 518, "split": split_name, "split_seed": args.split_seed,
                "dataset": "KITTI_depth_selection"}
        torch.save({"indices": idx_buf, "depths": dep, "meta": meta,
                    **{f"tokens_L{L}": tok[L] for L in layers}}, out_path)
        sz = out_path.stat().st_size / 1e9
        print(f"[kitti-feat] wrote {out_path}  ({sz:.2f} GB)  in "
              f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
