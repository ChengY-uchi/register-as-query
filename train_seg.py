"""Train one VOC 2012 segmentation variant (S1 | S2 | S3 | S4) on cached
frozen-backbone features. Prints per-epoch train loss, test mIoU, and
per-class IoU breakdown at the end. Saves checkpoint + summary JSON.

Usage:
    python train_seg.py --variant S1 --seed 42
    python train_seg.py --variant S4 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.seg_heads import SegHead, VARIANTS
from voc_dataset import NUM_CLASSES, IGNORE_INDEX, VOC_CLASSES


# ------------------------- cached dataset -------------------------

class VocSegFeatDataset(Dataset):
    def __init__(self, cache_path: Path):
        c = torch.load(Path(cache_path), map_location="cpu", weights_only=False)
        self.cls = c["cls"].float()
        self.regs = c["regs"].float()
        self.patches = c["patches"].float()
        self.masks = c["masks"]                       # int16, [N, 224, 224]
        self.image_ids = c["image_ids"]

    def __len__(self): return len(self.image_ids)

    def __getitem__(self, i):
        return {
            "cls":     self.cls[i],
            "regs":    self.regs[i],
            "patches": self.patches[i],
            "mask":    self.masks[i].long(),
            "image_id": self.image_ids[i],
        }


# ------------------------- metrics -------------------------

class IoUMeter:
    """Multi-class confusion-matrix tracker; computes per-class IoU + mIoU."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.C = num_classes
        self.ignore = ignore_index
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred, target: [B, H, W] int
        valid = target != self.ignore
        p = pred[valid].view(-1)
        t = target[valid].view(-1)
        idx = t * self.C + p
        bc = torch.bincount(idx, minlength=self.C * self.C)
        self.mat += bc.to(self.mat.device).reshape(self.C, self.C)

    def compute(self):
        mat = self.mat.double()
        inter = mat.diag()
        union = mat.sum(0) + mat.sum(1) - inter
        iou = (inter / union.clamp(min=1e-9)).cpu().numpy()
        valid = (union > 0).cpu().numpy()
        per_class = {VOC_CLASSES[c]: float(iou[c]) for c in range(self.C)}
        present = [iou[c] for c in range(self.C) if valid[c]]
        miou = float(np.mean(present)) if present else 0.0
        return {"miou": miou, "per_class": per_class,
                "classes_present": int(valid.sum())}


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
    meter = IoUMeter(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    tot_loss = 0.0; tot_n = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    for batch in loader:
        cls = batch["cls"].to(device, non_blocking=True)
        regs = batch["regs"].to(device, non_blocking=True)
        patches = batch["patches"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        feat_in = {"cls": cls, "regs": regs, "patches": patches}
        with torch.set_grad_enabled(train):
            out = model(feat_in, out_hw=mask.shape[-1])
            logits = out["logits_full"]                          # [B, C, H, W]
            loss = loss_fn(logits, mask)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
        tot_loss += loss.item() * mask.shape[0]; tot_n += mask.shape[0]
        pred = logits.argmax(dim=1)
        meter.update(pred.cpu(), mask.cpu())
    m = meter.compute()
    m["loss"] = tot_loss / max(tot_n, 1)
    return m


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-cache", default="data/voc_features_train_L23.pt")
    ap.add_argument("--test-cache",  default="data/voc_features_test_L23.pt")
    ap.add_argument("--out-dir", default="results/voc_seg/{variant}/seed{seed}")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir.format(variant=args.variant, seed=args.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print(f"[seg] variant={args.variant}  seed={args.seed}  out={out_dir}", flush=True)
    t0 = time.time()
    tr = VocSegFeatDataset(args.train_cache)
    te = VocSegFeatDataset(args.test_cache)
    print(f"[seg] loaded caches in {time.time()-t0:.1f}s  train={len(tr)}  test={len(te)}",
          flush=True)

    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    te_dl = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = SegHead(args.variant).to(args.device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seg] trainable params: {n_trainable:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    steps_per_epoch = len(tr_dl)
    total = args.epochs * steps_per_epoch
    warm = args.warmup_epochs * steps_per_epoch
    sched = make_schedule(optim, total, warm)

    history = []
    best = {"miou": -1.0, "epoch": -1}
    t_start = time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        trm = run_epoch(model, tr_dl, optim, sched, args.device, train=True)
        tem = run_epoch(model, te_dl, optim, sched, args.device, train=False)
        el = time.time() - t0
        history.append({"epoch": ep + 1, "train": {"loss": trm["loss"], "miou": trm["miou"]},
                        "test":  {"loss": tem["loss"], "miou": tem["miou"]},
                        "seconds": el})
        print(f"[ep {ep+1:02d}/{args.epochs}]  "
              f"tr_loss={trm['loss']:.3f}  tr_miou={trm['miou']:.4f}  "
              f"te_loss={tem['loss']:.3f}  te_miou={tem['miou']:.4f}  ({el:.1f}s)",
              flush=True)
        if tem["miou"] > best["miou"]:
            best = {"miou": tem["miou"], "epoch": ep + 1,
                    "per_class": tem["per_class"]}
            torch.save({"model": model.state_dict(), "variant": args.variant,
                        "seed": args.seed, "epoch": ep + 1},
                       out_dir / "best.pt")

    total_elapsed = time.time() - t_start
    print(f"[seg] done in {total_elapsed:.1f}s.  best: ep={best['epoch']} "
          f"miou={best['miou']:.4f}", flush=True)

    # routing snapshot (S4 = image-invariant; S5 = content-aware mean across test)
    routing = None
    routing_std = None
    if args.variant in ("S4", "S6", "S7"):
        with torch.no_grad():
            routing = model.routing.routing_weights().cpu().tolist()   # [H, W, R]
    elif args.variant == "S5":
        # average routing across the test set
        model.eval()
        all_w = []
        with torch.no_grad():
            for batch in te_dl:
                feat_in = {
                    "cls":     batch["cls"].to(args.device),
                    "regs":    batch["regs"].to(args.device),
                    "patches": batch["patches"].to(args.device),
                }
                w = model.routing.routing_weights(patches=feat_in["patches"])   # [B, H, W, R]
                all_w.append(w.cpu())
        all_w = torch.cat(all_w, dim=0)                     # [N, H, W, R]
        routing = all_w.mean(dim=0).tolist()                # [H, W, R]
        routing_std = all_w.std(dim=0).tolist()             # [H, W, R]

    summary = {
        "variant": args.variant, "seed": args.seed,
        "config": vars(args),
        "n_trainable": n_trainable,
        "best": best,
        "history": history,
        "routing": routing,
        "routing_std": routing_std,
        "total_seconds": total_elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[seg] wrote {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
