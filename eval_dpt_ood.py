"""OOD evaluation of DPT-trained D1/D2/D7 on corrupted NYU test images.

For each corruption type, re-extract 518×518 4-layer DINOv2-L-reg features
from corrupted images, then run each trained DPTHead. Tests whether register
concat provides robustness under distribution shift (analogous to seg OOD-1).

Output: results/nyu_dpt/ood_table.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
os.environ.setdefault("XFORMERS_DISABLED", "1")

import sys
import time
from pathlib import Path
from typing import List, Dict

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.backbone import FrozenDinoReg
from models.dpt import DPTHead
from extract_nyu_features_518 import joint_eval_transform_518
from nyu_dataset import stratified_80_20, valid_mask, MIN_DEPTH, MAX_DEPTH


LAYERS = [5, 11, 17, 23]


# ----------------------- corruptions (PIL → PIL) -----------------------

def make_corruption(kind: str):
    if kind == "clean":
        return lambda img: img
    if kind.startswith("gauss_noise_"):
        sigma = float(kind.split("_")[-1]) * 255.0
        def corrupt(img):
            arr = np.array(img, dtype=np.float32)
            arr += np.random.normal(0, sigma, arr.shape)
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return corrupt
    if kind.startswith("gauss_blur_"):
        k = int(kind.split("_")[-1])
        sigma = k / 3.0
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=sigma))
    if kind.startswith("jpeg_"):
        q = int(kind.split("_")[-1])
        def corrupt(img):
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
            return Image.open(buf).convert("RGB")
        return corrupt
    if kind.startswith("dark_"):
        alpha = float(kind.split("_")[-1])
        def corrupt(img):
            arr = np.array(img, dtype=np.float32) * alpha
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return corrupt
    if kind.startswith("defocus_blur_"):
        r = int(kind.split("_")[-1])
        y, x = np.ogrid[-r:r+1, -r:r+1]
        kernel = (x*x + y*y <= r*r).astype(np.float32)
        kernel /= kernel.sum()
        def corrupt(img):
            arr = np.array(img, dtype=np.float32)
            out = cv2.filter2D(arr, -1, kernel)
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        return corrupt
    if kind.startswith("motion_blur_"):
        L = int(kind.split("_")[-1])
        kernel = np.zeros((L, L), dtype=np.float32)
        kernel[L // 2, :] = 1.0 / L
        def corrupt(img):
            arr = np.array(img, dtype=np.float32)
            out = cv2.filter2D(arr, -1, kernel)
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        return corrupt
    if kind.startswith("contrast_"):
        alpha = float(kind.split("_")[-1])
        def corrupt(img):
            arr = np.array(img, dtype=np.float32)
            mean = arr.mean(axis=(0, 1), keepdims=True)
            out = mean + (arr - mean) * alpha
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        return corrupt
    if kind.startswith("fog_"):
        density = float(kind.split("_")[-1])
        def corrupt(img):
            arr = np.array(img, dtype=np.float32)
            fog = np.full_like(arr, 220.0)
            out = arr * (1 - density) + fog * density
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        return corrupt
    if kind.startswith("shot_noise_"):
        lam = float(kind.split("_")[-1])  # smaller = more noise
        def corrupt(img):
            arr = np.array(img, dtype=np.float32) / 255.0
            noisy = np.random.poisson(arr * lam) / lam
            return Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype(np.uint8))
        return corrupt
    raise ValueError(kind)


# ----------------------- backbone forward with hooks -----------------------

@torch.no_grad()
def extract_multilayer(model, x: torch.Tensor, layers: List[int]) -> Dict[int, torch.Tensor]:
    cache = {}; handles = []
    for L in layers:
        def make_hook(LL):
            def hook(_m, _i, o): cache[LL] = o
            return hook
        handles.append(model.blocks[L].register_forward_hook(make_hook(L)))
    try:
        _ = model(x)
    finally:
        for h in handles: h.remove()
    return {L: model.norm(cache[L]) for L in layers}


# ----------------------- metrics -----------------------

def depth_metrics(pred, gt, mask):
    p = pred[mask].clamp(min=MIN_DEPTH, max=MAX_DEPTH)
    g = gt[mask].clamp(min=MIN_DEPTH, max=MAX_DEPTH)
    rmse = torch.sqrt(((p - g) ** 2).mean()).item()
    absrel = ((p - g).abs() / g).mean().item()
    ratio = torch.maximum(p / g, g / p)
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()
    return {"rmse": rmse, "absrel": absrel, "delta1": d1, "delta2": d2, "delta3": d3}


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    ap.add_argument("--results-root", default="results/nyu_dpt")
    ap.add_argument("--variants", nargs="+", default=["D1", "D2", "D7"])
    ap.add_argument("--corruptions", nargs="+",
                    default=["clean", "gauss_noise_0.08", "gauss_blur_9",
                             "jpeg_20", "dark_0.5"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--noise-seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="training seeds whose checkpoints to evaluate (multi-seed)")
    ap.add_argument("--out-suffix", default="",
                    help="optional suffix for output .md/.json filenames")
    ap.add_argument("--perm-registers", action="store_true",
                    help="ablation: shuffle the 4 register tokens across batch "
                         "before head inference (tests whether register CONTENT "
                         "carries image-specific guidance; affects DQ/DQC/D7/DF)")
    ap.add_argument("--perm-seed", type=int, default=12345,
                    help="seed for the cross-batch register permutation")
    args = ap.parse_args()

    np.random.seed(args.noise_seed); torch.manual_seed(args.noise_seed)

    # test split (same as training)
    _, test_idx = stratified_80_20(1449, seed=args.split_seed)
    N = len(test_idx)
    print(f"[ood-dpt] test split: {N} images", flush=True)

    print("[ood-dpt] loading DINOv2-L-reg backbone ...", flush=True)
    backbone = FrozenDinoReg(device=args.device)

    print("[ood-dpt] loading trained heads ...", flush=True)
    # heads is keyed by (variant, seed) tuple
    heads = {}
    for v in args.variants:
        for s in args.seeds:
            ckpt_path = Path(args.results_root) / v / f"seed{s}" / "best.pt"
            if not ckpt_path.exists():
                print(f"[ood-dpt] SKIP {v} seed={s}: {ckpt_path} not found")
                continue
            ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            m = DPTHead(variant=v).to(args.device).eval()
            m.load_state_dict(ck["model"])
            heads[(v, s)] = m
            print(f"[ood-dpt]   loaded {v} seed={s}")

    print("[ood-dpt] loading NYU mat into RAM ...", flush=True)
    with h5py.File(args.mat, "r") as f:
        all_images = np.asarray(f["images"][:])
        all_depths = np.asarray(f["depths"][:])

    results: Dict[str, Dict[str, dict]] = {}

    for corr in args.corruptions:
        print(f"\n[ood-dpt] === corruption = {corr} ===", flush=True)
        np.random.seed(args.noise_seed); torch.manual_seed(args.noise_seed)
        cfn = make_corruption(corr)

        # buffer multi-layer features for the whole test set (in CPU RAM, fp16)
        feats = {L: [] for L in LAYERS}
        deps = []
        t0 = time.time()
        for bstart in range(0, N, args.batch_size):
            bi = test_idx[bstart : bstart + args.batch_size]
            xs, ds = [], []
            for idx in bi:
                img = all_images[idx].transpose(2, 1, 0)        # (H, W, 3)
                dep = all_depths[idx].T                         # (H, W)
                pil = Image.fromarray(img)
                pil = cfn(pil)
                x_t, d_t = joint_eval_transform_518(pil, dep)
                xs.append(x_t); ds.append(d_t)
            x = torch.stack(xs).to(args.device, non_blocking=True)
            tokens = extract_multilayer(backbone.model, x, LAYERS)
            for L in LAYERS:
                feats[L].append(tokens[L].to("cpu", dtype=torch.float16))
            deps.append(torch.stack(ds).to(torch.float16))
        feats = {L: torch.cat(feats[L], dim=0) for L in LAYERS}
        deps = torch.cat(deps, dim=0)
        print(f"[ood-dpt]   features done in {time.time()-t0:.1f}s", flush=True)

        # ---- DQ_PERMQ ablation: shuffle register tokens across the batch ----
        # Tests whether the 4 register tokens carry image-specific guidance.
        # If perf is unchanged, registers are interchangeable / content-free
        # and DQ's gain comes from architecture, not register content.
        if args.perm_registers:
            NUM_REG = 4
            g = torch.Generator().manual_seed(args.perm_seed)
            perm = torch.randperm(N, generator=g)
            for L in LAYERS:
                feats[L][:, 1:1 + NUM_REG, :] = feats[L][perm][:, 1:1 + NUM_REG, :]
            print(f"[ood-dpt]   PERMUTED registers across batch "
                  f"(perm_seed={args.perm_seed})", flush=True)

        results[corr] = {}
        for key, model in heads.items():
            v, s = key
            key_str = f"{v}_s{s}"
            preds, gts, masks = [], [], []
            with torch.no_grad():
                for i in range(0, N, args.batch_size):
                    sl = slice(i, i + args.batch_size)
                    tokens_layer = [feats[L][sl].float().to(args.device, non_blocking=True)
                                     for L in LAYERS]
                    dep = deps[sl].float().to(args.device)
                    log_d = model(tokens_layer, input_hw=(518, 518))
                    log_d = F.interpolate(log_d, size=dep.shape[-2:],
                                          mode="bilinear", align_corners=False)
                    log_d = log_d[:, 0]
                    pred = torch.exp(log_d)
                    mask = valid_mask(dep)
                    preds.append(pred.cpu()); gts.append(dep.cpu()); masks.append(mask.cpu())
            p = torch.cat(preds); g = torch.cat(gts); m = torch.cat(masks)
            r = depth_metrics(p, g, m)
            results[corr][key_str] = r
            print(f"[ood-dpt]   {key_str}: rmse={r['rmse']:.3f}  d1={r['delta1']:.3f}",
                  flush=True)
        del feats, deps
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---------- aggregate across seeds ----------
    # Per (corruption, variant): mean ± std of metrics across seeds
    agg = {}    # corruption -> variant -> metric -> {mean, std, n}
    for c in args.corruptions:
        agg[c] = {}
        for v in args.variants:
            vals = {}
            for s in args.seeds:
                key = f"{v}_s{s}"
                if key not in results[c]: continue
                for mk, mv in results[c][key].items():
                    vals.setdefault(mk, []).append(mv)
            if not vals: continue
            agg[c][v] = {}
            for mk, lst in vals.items():
                agg[c][v][mk] = {
                    "mean": float(np.mean(lst)),
                    "std":  float(np.std(lst, ddof=1)) if len(lst) > 1 else 0.0,
                    "n":    len(lst),
                    "raw":  lst,
                }

    # ---------- write outputs ----------
    suf = args.out_suffix
    out_md = Path(args.results_root) / f"ood_table{suf}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# DPT — OOD evaluation on corrupted NYU test (multi-seed)",
             "",
             f"Trained checkpoints from seeds {args.seeds}. N = {N} images per "
             f"corruption. Backbone features re-extracted from corrupted images.",
             ""]
    variants_present = [v for v in args.variants
                         if any((v, s) in heads for s in args.seeds)]

    def fmt_pm(v, mk, ndigits=3, pct=False):
        if v not in agg.get(c, {}) or mk not in agg[c][v]: return "—"
        m, sd = agg[c][v][mk]["mean"], agg[c][v][mk]["std"]
        if pct:
            return f"{100*m:.2f} ± {100*sd:.2f}"
        return f"{m:.{ndigits}f} ± {sd:.{ndigits}f}"

    lines += ["## RMSE (mean ± std)  (lower is better)", "",
              "| variant | " + " | ".join(args.corruptions) + " |",
              "|" + "|".join(["---"] * (len(args.corruptions) + 1)) + "|"]
    for v in variants_present:
        row = [v]
        for c in args.corruptions:
            row.append(fmt_pm(v, "rmse"))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## δ₁ (mean ± std)  (higher is better)", "",
              "| variant | " + " | ".join(args.corruptions) + " |",
              "|" + "|".join(["---"] * (len(args.corruptions) + 1)) + "|"]
    for v in variants_present:
        row = [v]
        for c in args.corruptions:
            row.append(fmt_pm(v, "delta1"))
        lines.append("| " + " | ".join(row) + " |")

    # D7 - D2 with significance
    if "D2" in variants_present and "D7" in variants_present:
        lines += ["", "## D7 − D2 Δδ₁ — register's incremental value (mean ± pooled std)",
                  "",
                  "| corruption | D7 mean δ₁ | D2 mean δ₁ | Δ (pp) | pooled std (pp) | ratio | confirmed? |",
                  "|---|---|---|---|---|---|---|"]
        for c in args.corruptions:
            if "D2" not in agg[c] or "D7" not in agg[c]: continue
            d2m = agg[c]["D2"]["delta1"]["mean"]; d2s = agg[c]["D2"]["delta1"]["std"]
            d7m = agg[c]["D7"]["delta1"]["mean"]; d7s = agg[c]["D7"]["delta1"]["std"]
            delta = (d7m - d2m) * 100
            pooled = 100 * np.sqrt(d2s**2 + d7s**2)
            ratio = delta / max(pooled, 1e-9)
            mark = "✓" if ratio > 2 else ("≈" if abs(ratio) > 1 else "✗")
            lines.append(
                f"| {c} | {d7m:.3f} | {d2m:.3f} | {delta:+.2f} | "
                f"{pooled:.2f} | {ratio:+.2f} | {mark} |"
            )

    # raw per-seed table
    lines += ["", "## Per-seed raw  δ₁",
              "",
              "| variant × seed | " + " | ".join(args.corruptions) + " |",
              "|" + "|".join(["---"] * (len(args.corruptions) + 1)) + "|"]
    for v in variants_present:
        for s in args.seeds:
            key = f"{v}_s{s}"
            row = [key]
            for c in args.corruptions:
                if key in results[c]:
                    row.append(f"{results[c][key]['delta1']:.3f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")

    out_md.write_text("\n".join(lines))
    (out_md.with_suffix(".json")).write_text(
        json.dumps({"per_run": results, "aggregated": agg}, indent=2))
    print(f"\n[ood-dpt] wrote {out_md}", flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
