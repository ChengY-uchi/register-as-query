"""Phase OOD-spatial: probe register's spatial-dependency claim under
(A) region-localized image blur, (B) random patch-token dropout.

(A) — same Gaussian blur as before, but applied only to one half/quarter of
each image. If r2 (top) / r1 (left) / r3 (center) really specialize spatially,
then under top-only blur, the register-augmented heads should recover the
top region from r2's surviving aggregate (while patch features in that region
are degraded).

(B) — at inference, zero out a fraction p of patch tokens (CLS + registers
kept intact). Tests whether register provides a backup global summary when
patch features themselves fail.

Output: results/nyu_dpt/ood_spatial.md (+ .json)
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
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.backbone import FrozenDinoReg, NUM_REGISTERS
from models.dpt import DPTHead
from extract_nyu_features_518 import joint_eval_transform_518
from nyu_dataset import stratified_80_20, valid_mask, MIN_DEPTH, MAX_DEPTH


LAYERS = [5, 11, 17, 23]
BLUR_SIGMA = 3.0


# ----------------------- (A) localized blur corruptions -----------------------

def _region_mask(H: int, W: int, region: str) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    if region == "top":    m[:H // 2] = True
    elif region == "bottom": m[H // 2:] = True
    elif region == "left":  m[:, :W // 2] = True
    elif region == "right": m[:, W // 2:] = True
    elif region == "center":
        m[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True
    else: raise ValueError(region)
    return m


def make_localized_blur(region: str):
    def corrupt(img: Image.Image) -> Image.Image:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))
        arr = np.array(img); barr = np.array(blurred)
        H, W = arr.shape[:2]
        m = _region_mask(H, W, region)
        out = np.where(m[..., None], barr, arr)
        return Image.fromarray(out.astype(np.uint8))
    return corrupt


def make_corruption(kind: str):
    if kind == "clean":
        return lambda img: img
    if kind.startswith("blur_"):
        return make_localized_blur(kind.split("_", 1)[1])
    if kind.startswith("gauss_blur_"):
        sigma = int(kind.split("_")[-1]) / 3.0
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=sigma))
    raise ValueError(kind)


# ----------------------- (B) patch-token dropout -----------------------

def patch_dropout_tokens(tokens: torch.Tensor, p: float,
                         num_reg: int = NUM_REGISTERS,
                         seed: int = 0) -> torch.Tensor:
    """Zero out p% of patch tokens.  CLS and registers untouched."""
    if p <= 0: return tokens
    B, N, D = tokens.shape
    P = N - 1 - num_reg
    g = torch.Generator(device=tokens.device); g.manual_seed(seed)
    keep = (torch.rand(B, P, 1, generator=g, device=tokens.device) > p).float()
    out = tokens.clone()
    out[:, 1 + num_reg :] = out[:, 1 + num_reg :] * keep
    return out


# ----------------------- backbone forward + heads -----------------------

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


def depth_metrics(pred, gt, mask):
    p = pred[mask].clamp(min=MIN_DEPTH, max=MAX_DEPTH)
    g = gt[mask].clamp(min=MIN_DEPTH, max=MAX_DEPTH)
    ratio = torch.maximum(p / g, g / p)
    return {
        "rmse": torch.sqrt(((p - g) ** 2).mean()).item(),
        "absrel": ((p - g).abs() / g).mean().item(),
        "delta1": (ratio < 1.25).float().mean().item(),
        "delta2": (ratio < 1.25 ** 2).float().mean().item(),
        "delta3": (ratio < 1.25 ** 3).float().mean().item(),
    }


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    ap.add_argument("--results-root", default="results/nyu_dpt")
    ap.add_argument("--variants", nargs="+", default=["D1", "D2", "D7", "DQ"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regions", nargs="+",
                    default=["clean", "blur_top", "blur_bottom",
                             "blur_left", "blur_right", "blur_center",
                             "gauss_blur_9"])
    ap.add_argument("--dropout-rates", nargs="+", type=float,
                    default=[0.0, 0.10, 0.30, 0.50])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    _, test_idx = stratified_80_20(1449, seed=args.split_seed)
    N = len(test_idx)
    print(f"[ood-sp] test split: {N} images", flush=True)

    print("[ood-sp] loading backbone ...", flush=True)
    backbone = FrozenDinoReg(device=args.device)

    print("[ood-sp] loading trained DPT heads ...", flush=True)
    heads = {}
    for v in args.variants:
        ckpt = Path(args.results_root) / v / f"seed{args.seed}" / "best.pt"
        if not ckpt.exists():
            print(f"[ood-sp] SKIP {v}: {ckpt} not found"); continue
        s = torch.load(ckpt, map_location=args.device, weights_only=False)
        m = DPTHead(variant=v).to(args.device).eval()
        m.load_state_dict(s["model"])
        heads[v] = m
        print(f"[ood-sp]   loaded {v}")

    print("[ood-sp] loading NYU mat ...", flush=True)
    with h5py.File(args.mat, "r") as f:
        all_images = np.asarray(f["images"][:])
        all_depths = np.asarray(f["depths"][:])

    # ---------- (A) localized blur corruptions ----------
    a_results: Dict[str, Dict[str, dict]] = {}
    for corr in args.regions:
        print(f"\n[ood-sp][A] === {corr} ===", flush=True)
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        cfn = make_corruption(corr)

        feats = {L: [] for L in LAYERS}; deps = []
        t0 = time.time()
        for bstart in range(0, N, args.batch_size):
            bi = test_idx[bstart : bstart + args.batch_size]
            xs, ds = [], []
            for idx in bi:
                img = all_images[idx].transpose(2, 1, 0)
                dep = all_depths[idx].T
                pil = Image.fromarray(img)
                pil = cfn(pil)
                x_t, d_t = joint_eval_transform_518(pil, dep)
                xs.append(x_t); ds.append(d_t)
            x = torch.stack(xs).to(args.device, non_blocking=True)
            tokens = extract_multilayer(backbone.model, x, LAYERS)
            for L in LAYERS:
                feats[L].append(tokens[L].to("cpu", dtype=torch.float16))
            deps.append(torch.stack(ds).to(torch.float16))
        feats = {L: torch.cat(feats[L]) for L in LAYERS}
        deps = torch.cat(deps)
        print(f"[ood-sp][A]   features in {time.time()-t0:.1f}s", flush=True)

        a_results[corr] = {}
        for v, model in heads.items():
            preds, gts, masks = [], [], []
            with torch.no_grad():
                for i in range(0, N, args.batch_size):
                    sl = slice(i, i + args.batch_size)
                    tk = [feats[L][sl].float().to(args.device) for L in LAYERS]
                    dep = deps[sl].float().to(args.device)
                    log_d = model(tk, input_hw=(518, 518))
                    log_d = F.interpolate(log_d, size=dep.shape[-2:],
                                          mode="bilinear", align_corners=False)[:, 0]
                    pred = torch.exp(log_d)
                    mask = valid_mask(dep)
                    preds.append(pred.cpu()); gts.append(dep.cpu()); masks.append(mask.cpu())
            r = depth_metrics(torch.cat(preds), torch.cat(gts), torch.cat(masks))
            a_results[corr][v] = r
            print(f"[ood-sp][A]   {v}: rmse={r['rmse']:.3f}  d1={r['delta1']:.3f}",
                  flush=True)
        del feats, deps
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------- (B) patch token dropout ----------
    # Extract CLEAN features once, then dropout in token space for each p.
    print(f"\n[ood-sp][B] extracting clean features once ...", flush=True)
    feats_clean = {L: [] for L in LAYERS}; deps_clean = []
    cfn = make_corruption("clean")
    for bstart in range(0, N, args.batch_size):
        bi = test_idx[bstart : bstart + args.batch_size]
        xs, ds = [], []
        for idx in bi:
            img = all_images[idx].transpose(2, 1, 0)
            dep = all_depths[idx].T
            x_t, d_t = joint_eval_transform_518(Image.fromarray(img), dep)
            xs.append(x_t); ds.append(d_t)
        x = torch.stack(xs).to(args.device, non_blocking=True)
        tokens = extract_multilayer(backbone.model, x, LAYERS)
        for L in LAYERS:
            feats_clean[L].append(tokens[L].to("cpu", dtype=torch.float16))
        deps_clean.append(torch.stack(ds).to(torch.float16))
    feats_clean = {L: torch.cat(feats_clean[L]) for L in LAYERS}
    deps_clean = torch.cat(deps_clean)

    b_results: Dict[float, Dict[str, dict]] = {}
    for p in args.dropout_rates:
        print(f"\n[ood-sp][B] === patch dropout p={p} ===", flush=True)
        b_results[p] = {}
        for v, model in heads.items():
            preds, gts, masks = [], [], []
            with torch.no_grad():
                for i in range(0, N, args.batch_size):
                    sl = slice(i, i + args.batch_size)
                    tk = [feats_clean[L][sl].float().to(args.device) for L in LAYERS]
                    if p > 0:
                        tk = [patch_dropout_tokens(t, p, seed=args.seed + L)
                              for t, L in zip(tk, LAYERS)]
                    dep = deps_clean[sl].float().to(args.device)
                    log_d = model(tk, input_hw=(518, 518))
                    log_d = F.interpolate(log_d, size=dep.shape[-2:],
                                          mode="bilinear", align_corners=False)[:, 0]
                    pred = torch.exp(log_d)
                    mask = valid_mask(dep)
                    preds.append(pred.cpu()); gts.append(dep.cpu()); masks.append(mask.cpu())
            r = depth_metrics(torch.cat(preds), torch.cat(gts), torch.cat(masks))
            b_results[p][v] = r
            print(f"[ood-sp][B]   {v}: rmse={r['rmse']:.3f}  d1={r['delta1']:.3f}",
                  flush=True)

    # ---------- write outputs ----------
    out_md = Path(args.results_root) / "ood_spatial.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# DPT — spatial-dependency OOD probe",
             "",
             f"All evals use seed={args.seed} checkpoints.  N = {N} test imgs. "
             f"Backbone features re-extracted per condition (for (A)).",
             ""]

    # ---- (A) ----
    variants = [v for v in args.variants if v in heads]
    lines += ["## (A) Region-localized Gaussian blur — δ₁ (higher is better)", "",
              "| variant | " + " | ".join(args.regions) + " |",
              "|" + "|".join(["---"] * (len(args.regions) + 1)) + "|"]
    for v in variants:
        row = [v]
        for c in args.regions:
            row.append(f"{a_results[c][v]['delta1']:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    if "D2" in heads:
        lines += ["", "## (A) D7/DQ vs D2 Δδ₁ (pp) under localized blur", "",
                  "| variant | " + " | ".join(args.regions) + " |",
                  "|" + "|".join(["---"] * (len(args.regions) + 1)) + "|"]
        for v in [vv for vv in variants if vv != "D2"]:
            row = [v]
            for c in args.regions:
                d2 = a_results[c]["D2"]["delta1"]; dv = a_results[c][v]["delta1"]
                row.append(f"{100*(dv - d2):+.2f}")
            lines.append("| " + " | ".join(row) + " |")

    # ---- (B) ----
    lines += ["", "## (B) Random patch-token dropout — δ₁ (higher is better)", "",
              "| variant | " + " | ".join(f"p={p:.2f}" for p in args.dropout_rates) + " |",
              "|" + "|".join(["---"] * (len(args.dropout_rates) + 1)) + "|"]
    for v in variants:
        row = [v]
        for p in args.dropout_rates:
            row.append(f"{b_results[p][v]['delta1']:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    if "D2" in heads:
        lines += ["", "## (B) D7/DQ vs D2 Δδ₁ (pp) under patch dropout", "",
                  "| variant | " + " | ".join(f"p={p:.2f}" for p in args.dropout_rates) + " |",
                  "|" + "|".join(["---"] * (len(args.dropout_rates) + 1)) + "|"]
        for v in [vv for vv in variants if vv != "D2"]:
            row = [v]
            for p in args.dropout_rates:
                d2 = b_results[p]["D2"]["delta1"]; dv = b_results[p][v]["delta1"]
                row.append(f"{100*(dv - d2):+.2f}")
            lines.append("| " + " | ".join(row) + " |")

    out_md.write_text("\n".join(lines))
    (out_md.with_suffix(".json")).write_text(json.dumps(
        {"A_localized_blur": a_results, "B_patch_dropout": b_results}, indent=2))
    print(f"\n[ood-sp] wrote {out_md}", flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
