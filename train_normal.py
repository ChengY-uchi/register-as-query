"""Train one NYU surface-normal variant (N1..N7) on cached frozen-backbone
features. GT normals are derived from cached depths via finite differences.

    python train_normal.py --variant N1 --seed 42
    python train_normal.py --variant N7 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.normal_heads import NormalHead, VARIANTS_NORMAL, make_normal_head
from normal_utils import depth_to_normal, angular_metrics, cosine_loss


# ------------------------- cached features + GT normals -------------------------

class NyuNormalFeatDataset(Dataset):
    def __init__(self, cache_path: Path):
        c = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        self.cls = c["cls"].float()
        self.regs = c["regs"].float()
        self.patches = c["patches"].float()
        depths = c["depths"].float()                # [N, 224, 224]
        # Pre-compute GT normals + valid mask once.
        with torch.no_grad():
            n, m = depth_to_normal(depths)          # [N, H, W, 3], [N, H, W]
        self.normals = n.half()                     # save half precision
        self.valid_mask = m
        self.indices = c["indices"]

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        return {
            "cls":     self.cls[i],
            "regs":    self.regs[i],
            "patches": self.patches[i],
            "normal":  self.normals[i].float(),     # [H, W, 3]
            "mask":    self.valid_mask[i],          # [H, W]
            "index":   int(self.indices[i]),
        }


# ------------------------- utils -------------------------

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_schedule(optim, total, warm):
    def f(step):
        if step < warm: return (step + 1) / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optim, f)


# ------------------------- train / eval -------------------------

def run_epoch(model, loader, optim, sched, device, train: bool):
    (model.train if train else model.eval)()
    tot_loss = 0.0; tot_n = 0
    preds, gts, masks = [], [], []
    for batch in loader:
        cls     = batch["cls"].to(device, non_blocking=True)
        regs    = batch["regs"].to(device, non_blocking=True)
        patches = batch["patches"].to(device, non_blocking=True)
        normal  = batch["normal"].to(device, non_blocking=True)     # [B, H, W, 3]
        mask    = batch["mask"].to(device, non_blocking=True)        # [B, H, W]

        feat_in = {"cls": cls, "regs": regs, "patches": patches}
        with torch.set_grad_enabled(train):
            out = model(feat_in, out_hw=normal.shape[1])
            n_pred = out["normal_full"]                              # [B, 3, H, W]
            loss = cosine_loss(n_pred, normal, mask)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
        tot_loss += loss.item() * normal.shape[0]
        tot_n    += normal.shape[0]
        if not train:
            with torch.no_grad():
                pn = n_pred.permute(0, 2, 3, 1)
                pn = pn / pn.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            preds.append(pn.cpu())
            gts.append(normal.cpu())
            masks.append(mask.cpu())

    out = {"loss": tot_loss / max(tot_n, 1)}
    if preds:
        p = torch.cat(preds); g = torch.cat(gts); m = torch.cat(masks)
        out.update(angular_metrics(p, g, m))
    return out


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS_NORMAL, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-cache", default="data/nyu_features_train_L23.pt")
    ap.add_argument("--test-cache",  default="data/nyu_features_test_L23.pt")
    ap.add_argument("--out-dir", default="results/nyu_normal/{variant}/seed{seed}")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir.format(variant=args.variant, seed=args.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    print(f"[normal] variant={args.variant}  seed={args.seed}  out={out_dir}",
          flush=True)

    t0 = time.time()
    tr = NyuNormalFeatDataset(args.train_cache)
    te = NyuNormalFeatDataset(args.test_cache)
    print(f"[normal] loaded caches + computed normals in {time.time()-t0:.1f}s  "
          f"train={len(tr)}  test={len(te)}", flush=True)

    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    te_dl = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = make_normal_head(args.variant).to(args.device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[normal] trainable params: {n_trainable:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    total = args.epochs * len(tr_dl)
    warm  = args.warmup_epochs * len(tr_dl)
    sched = make_schedule(optim, total, warm)

    history = []
    best = {"mean_ang_err": float("inf"), "epoch": -1}
    t_start = time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        trm = run_epoch(model, tr_dl, optim, sched, args.device, train=True)
        tem = run_epoch(model, te_dl, optim, sched, args.device, train=False)
        el = time.time() - t0
        history.append({"epoch": ep+1, "train_loss": trm["loss"],
                        "test": tem, "seconds": el})
        print(f"[ep {ep+1:02d}/{args.epochs}]  "
              f"tr_loss={trm['loss']:.3f}  "
              f"te_mean_ang={tem['mean_ang_err']:.2f}  "
              f"te_med_ang={tem['median_ang_err']:.2f}  "
              f"te_acc11.25={tem['acc_11_25']:.3f}  "
              f"te_acc30={tem['acc_30']:.3f}  ({el:.1f}s)", flush=True)
        if tem["mean_ang_err"] < best["mean_ang_err"]:
            best = {"mean_ang_err": tem["mean_ang_err"],
                    "median_ang_err": tem["median_ang_err"],
                    "acc_11_25": tem["acc_11_25"],
                    "acc_22_5":  tem["acc_22_5"],
                    "acc_30":    tem["acc_30"],
                    "epoch": ep + 1}
            torch.save({"model": model.state_dict(), "variant": args.variant,
                        "seed": args.seed, "epoch": ep+1},
                       out_dir / "best.pt")
    total_elapsed = time.time() - t_start

    routing = None
    if args.variant in ("N4", "N6", "N7"):
        with torch.no_grad():
            routing = model.routing.routing_weights().cpu().tolist()

    summary = {
        "variant": args.variant, "seed": args.seed, "task": "normal",
        "config": vars(args), "n_trainable": n_trainable,
        "best": best, "history": history, "routing": routing,
        "total_seconds": total_elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[normal] wrote {out_dir/'summary.json'}", flush=True)
    print(f"[normal] best: ep={best['epoch']}  mean_ang={best['mean_ang_err']:.2f}°  "
          f"acc@11.25={best['acc_11_25']:.3f}", flush=True)


if __name__ == "__main__":
    main()
