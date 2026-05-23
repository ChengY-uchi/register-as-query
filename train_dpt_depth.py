"""Train DPT-decoded depth on cached 518×518 4-layer DINOv2-L-reg features.

Variant differs only at the per-layer Readout module (see models/dpt.py).
Pipeline: 4-layer tokens → ProjectReadout × variant → Reassemble × 4
        → FeatureFusion (deep → shallow) → output head → log-depth

Loss: SILog with NYU mask (depth in [0.5, 10] m).
Metrics: RMSE, AbsRel, δ₁, δ₂, δ₃, log_rmse.

    python train_dpt_depth.py --variant D1 --seed 42
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.dpt import DPTHead, DPT_VARIANTS
from nyu_dataset import MIN_DEPTH, MAX_DEPTH, valid_mask


# ----------------------- cache loader -----------------------

class NyuDpt518Dataset(Dataset):
    """Reads cached fp16 features; keeps them as fp16 in RAM and upcasts
    only the requested sample to fp32 in __getitem__.
    """
    def __init__(self, cache_path: Path, layers=(5, 11, 17, 23),
                 max_n: int = None, subsample_seed: int = 0):
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.layers = list(layers)
        self.depths = c["depths"]
        self.indices = c["indices"]
        self.tokens = {L: c[f"tokens_L{L}"] for L in self.layers}

        # Optional subsample for data-efficiency studies.
        if max_n is not None and max_n > 0 and max_n < len(self.indices):
            rng = np.random.default_rng(subsample_seed)
            sel = torch.from_numpy(rng.permutation(len(self.indices))[:max_n]).long()
            self.depths = self.depths[sel]
            self.indices = self.indices[sel]
            self.tokens = {L: self.tokens[L][sel] for L in self.layers}

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        # upcast only the i-th sample
        out = {
            "depth":  self.depths[i].float(),
            "index":  int(self.indices[i]),
        }
        for L in self.layers:
            out[f"tok_L{L}"] = self.tokens[L][i].float()
        return out


# ----------------------- loss / metrics -----------------------

def silog_loss(pred_log: torch.Tensor, gt: torch.Tensor,
               mask: torch.Tensor, lam: float = 0.85, alpha: float = 10.0):
    gt_log = torch.log(gt.clamp(min=MIN_DEPTH))
    g = (pred_log - gt_log) * mask
    n = mask.float().sum().clamp(min=1.0)
    Dg = (g ** 2).sum() / n
    Dg_ = (g.sum() / n) ** 2
    return alpha * torch.sqrt((Dg - lam * Dg_).clamp(min=1e-9))


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    p = pred[mask].clamp(min=MIN_DEPTH, max=MAX_DEPTH)
    g = gt[mask].clamp(min=MIN_DEPTH, max=MAX_DEPTH)
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    absrel = ((p - g).abs() / g).mean().item()
    ratio = torch.maximum(p / g, g / p)
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()
    log_rmse = torch.sqrt(((torch.log(p) - torch.log(g)) ** 2).mean()).item()
    return {"rmse": rmse, "absrel": absrel,
            "delta1": d1, "delta2": d2, "delta3": d3,
            "log_rmse": log_rmse}


# ----------------------- utils -----------------------

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_schedule(optim, total, warm):
    def f(step):
        if step < warm: return (step + 1) / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optim, f)


# ----------------------- epoch loop -----------------------

def run_epoch(model, loader, optim, sched, device, train: bool,
              layers, input_hw, depth_hw):
    (model.train if train else model.eval)()
    tot_loss = 0.0; tot_n = 0
    preds, gts, masks = [], [], []
    for batch in loader:
        tokens_per_layer = [batch[f"tok_L{L}"].to(device, non_blocking=True)
                             for L in layers]
        depth = batch["depth"].to(device, non_blocking=True)
        mask = valid_mask(depth)

        with torch.set_grad_enabled(train):
            log_d = model(tokens_per_layer, input_hw)             # [B, 1, H_out, W_out]
            # Bring to depth GT resolution
            log_d = F.interpolate(log_d, size=depth_hw,
                                   mode="bilinear", align_corners=False)
            log_d = log_d[:, 0]                                   # [B, H, W]
            loss = silog_loss(log_d, depth, mask)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                sched.step()
        tot_loss += loss.item() * depth.shape[0]
        tot_n    += depth.shape[0]
        if not train:
            with torch.no_grad():
                preds.append(torch.exp(log_d).cpu())
                gts.append(depth.cpu())
                masks.append(mask.cpu())

    out = {"loss": tot_loss / max(tot_n, 1)}
    if preds:
        p = torch.cat(preds); g = torch.cat(gts); m = torch.cat(masks)
        out.update(depth_metrics(p, g, m))
    return out


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=DPT_VARIANTS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-cache",
                    default="data/nyu_features_train_518_L5_11_17_23.pt")
    ap.add_argument("--test-cache",
                    default="data/nyu_features_test_518_L5_11_17_23.pt")
    ap.add_argument("--out-dir",
                    default="results/nyu_dpt/{variant}/seed{seed}")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)         # DPT decoder uses smaller LR
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--decoder-dim", type=int, default=256)
    ap.add_argument("--layers", nargs="+", type=int, default=[5, 11, 17, 23])
    ap.add_argument("--max-train", type=int, default=-1,
                    help="subsample training set; -1 = use all")
    ap.add_argument("--subsample-seed", type=int, default=0,
                    help="seed for which training images are chosen (independent of train seed)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir.format(variant=args.variant, seed=args.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    print(f"[dpt] variant={args.variant}  seed={args.seed}  out={out_dir}", flush=True)

    t0 = time.time()
    max_n = None if args.max_train <= 0 else args.max_train
    tr = NyuDpt518Dataset(args.train_cache, layers=args.layers,
                           max_n=max_n, subsample_seed=args.subsample_seed)
    te = NyuDpt518Dataset(args.test_cache, layers=args.layers)
    print(f"[dpt] loaded caches in {time.time()-t0:.1f}s  "
          f"train={len(tr)}  test={len(te)}  "
          f"(max_train={args.max_train})", flush=True)

    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    te_dl = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    grid = int(math.sqrt(tr.tokens[args.layers[0]].shape[1] - 1 - 4))
    print(f"[dpt] inferred grid = {grid}×{grid}  (518/14 = 37)", flush=True)
    input_hw = (518, 518)
    depth_hw = (518, 518)

    model = DPTHead(variant=args.variant, decoder_dim=args.decoder_dim,
                    grid=grid).to(args.device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dpt] trainable params: {n_trainable:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    total = args.epochs * len(tr_dl)
    warm  = args.warmup_epochs * len(tr_dl)
    sched = make_schedule(optim, total, warm)

    history = []
    best = {"rmse": float("inf"), "epoch": -1}
    t_start = time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        trm = run_epoch(model, tr_dl, optim, sched, args.device, train=True,
                        layers=args.layers, input_hw=input_hw, depth_hw=depth_hw)
        tem = run_epoch(model, te_dl, optim, sched, args.device, train=False,
                        layers=args.layers, input_hw=input_hw, depth_hw=depth_hw)
        el = time.time() - t0
        history.append({"epoch": ep + 1, "train_loss": trm["loss"],
                        "test": tem, "seconds": el})
        print(f"[ep {ep+1:02d}/{args.epochs}]  "
              f"tr_loss={trm['loss']:.3f}  "
              f"te_rmse={tem['rmse']:.3f}  te_absrel={tem['absrel']:.3f}  "
              f"te_d1={tem['delta1']:.3f}  ({el:.1f}s)", flush=True)
        if tem["rmse"] < best["rmse"]:
            best = {"rmse": tem["rmse"], "absrel": tem["absrel"],
                    "delta1": tem["delta1"], "delta2": tem["delta2"],
                    "delta3": tem["delta3"], "log_rmse": tem["log_rmse"],
                    "epoch": ep + 1}
            torch.save({"model": model.state_dict(), "variant": args.variant,
                        "seed": args.seed, "epoch": ep + 1},
                       out_dir / "best.pt")
    total_elapsed = time.time() - t_start

    summary = {
        "variant": args.variant, "seed": args.seed, "task": "dpt_depth",
        "config": vars(args), "n_trainable": n_trainable,
        "best": best, "history": history,
        "total_seconds": total_elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[dpt] wrote {out_dir/'summary.json'}", flush=True)
    print(f"[dpt] best: ep={best['epoch']}  rmse={best['rmse']:.3f}  "
          f"d1={best['delta1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
