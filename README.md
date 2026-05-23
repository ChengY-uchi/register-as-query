# Register-as-Query: Cross-Attention Integration of DINOv2 Register Tokens for Dense Prediction

This repository studies whether the **4 register tokens** introduced by
[DINOv2-reg (Darcet et al., 2023)](https://arxiv.org/abs/2309.16588) carry
useful information for **downstream dense prediction** when integrated as
**queries in a cross-attention residual branch**, in the spirit of
Mask2Former.

## Research question

DINOv2-reg adds 4 learnable register tokens to absorb high-norm "artifact"
activations in the patch grid. The registers help internal feature
quality, but their *downstream* utility is unclear:

> **Q.** Given a frozen DINOv2-L-reg backbone, can the 4 register tokens
> be used as cross-attention queries (with patches as K/V) to produce a
> residual prediction map that improves over a standard readout?

We answer this on three dense prediction tasks (NYU depth, KITTI depth,
VOC segmentation, NYU surface normals) with multi-seed evaluation and
OOD-corruption robustness analysis.

## Method: variants

All variants share a **frozen DINOv2-L-reg backbone** and differ only in
the **decoder head / readout**. The DPT variants follow Ranftl et al.
2021; the per-task BN-style variants are simple linear probes.

### DPT-based variants (NYU + KITTI depth)

| Variant | Readout | Register integration |
|---------|---------|----------------------|
| `D1` | patch only | none |
| `D2` | `[patch ; cls]` | none (cls only) — equivalent to official DPT readout |
| `D7` | `[patch ; cls ; reg_routed]` | concat (position-routed mean of 4 regs) |
| `DF` | patch | FiLM modulation conditioned on register mean |
| **`DQ`** | patch | **register-as-query cross-attention residual** |
| **`DQC`** | `[patch ; cls]` | **DQ residual + cls readout** (D2 base + DQ) |
| `DQ_LQ` | patch | learnable queries (not register-derived) — *transfer ablation* |
| `DQ_FRQ` | patch | fixed random queries (no training, no register) — *clean control* |

### Per-task BN-style variants (VOC seg + NYU normal)

For VOC segmentation and NYU surface normals, we use simple BN + linear
heads (matching the official DINOv2 linear-probe style, single-layer
features), so the cross-task signal is comparable.

| Variant | Base | DQ residual |
|---------|------|------------|
| `S1` / `N1` | patch only | no |
| `S_BN` / `N_BN` | BN + patch | no |
| **`S_DQ` / `N_DQ`** | **BN + patch** | **yes** (out_channels = num_classes / 3) |

## File structure

```
register-as-query/
├── README.md
├── models/
│   ├── backbone.py            # FrozenDinoReg wrapper around DINOv2-L-reg
│   ├── dpt.py                 # DPT decoder + RegQueryBranch + variants
│   ├── depth_heads.py         # single-layer D1..D9 (NYU depth simple probe)
│   ├── normal_heads.py        # N1..N9 (inherit DepthHead) + N_BN/N_DQ
│   └── seg_heads.py           # S1..S7 + S_BN/S_DQ; SpatialRoutingMLP for D4/D7
├── nyu_dataset.py             # NYU depth_v2 labeled (1449) loader
├── kitti_dataset.py           # KITTI depth_selection val (1000) loader
├── normal_utils.py            # depth_to_normal, angular metrics, cosine loss
├── extract_nyu_features_518.py     # 4-layer DINOv2 features @ 518×518 for DPT
├── extract_kitti_features_518.py   # same for KITTI
├── extract_voc_features.py         # single-layer L23 features for VOC seg
├── train_dpt_depth.py         # NYU depth DPT training (any DPT variant)
├── train_dpt_kitti.py         # KITTI depth DPT training
├── train_seg.py               # VOC segmentation training (S* variants)
├── train_normal.py            # NYU surface normal training (N* variants)
├── eval_dpt_ood.py            # OOD corruption eval on NYU + --perm-registers
├── eval_dpt_spatial.py        # Spatial OOD eval (localized blur + patch dropout)
└── results/
    ├── nyu_dpt/{variant}/seed{n}/summary.json    # NYU depth DPT runs
    ├── kitti_dpt/{variant}/seed{n}/summary.json  # KITTI depth runs
    ├── voc_seg/{variant}/seed{n}/summary.json    # VOC seg runs
    ├── nyu_normal/{variant}/seed{n}/summary.json # NYU normal runs
    ├── SUMMARY.md                                 # consolidated multi-task results
    └── nyu_dpt/ood_table_fixed.md                 # full 5-variant × 10-corruption OOD table
```

The full checkpoints (`best.pt`, ~100 MB each) and the per-run
`summary.json` (loss curves, every-epoch metrics) are **not** committed.
All headline numbers live in `results/SUMMARY.md` (best-epoch mean ± std
across seeds for every variant on every task). Re-train any variant in
~10 min (linear) – 25 min (DPT) on a single H100/L40S after running the
feature extraction once.

## Setup

```bash
# Python dependencies
pip install torch torchvision numpy h5py pillow opencv-python

# DINOv2 backbone code (vendored as a sibling repo)
git clone https://github.com/facebookresearch/dinov2.git ../dinov2
```

### Datasets

| Task | Dataset | Where to put it |
|------|---------|----------------|
| Depth (NYU) | NYU Depth V2 labeled (1449 images, `.mat`) | `datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat` |
| Depth (KITTI) | KITTI `data_depth_selection.zip` val_selection_cropped (1000) | `datasets/kitti/depth_selection/...` |
| Seg (VOC) | Pascal VOC 2012 trainval | `datasets/voc/VOCdevkit/VOC2012/` |
| Normal | derived from NYU depth on the fly (no extra data) | — |

We use an 80/20 split of each labeled set (NYU 1159/290, KITTI 800/200,
VOC 1159/290). This is intentionally a **small-data, frozen-backbone
study** so the comparison isolates the head/readout effect rather than
the backbone or dataset scale. Reproducing the official DINOv2 numbers
(NYU RMSE ~0.342) would require the full ~24K NYU raw-video depth pairs
and is orthogonal to our ablation question.

## How to run

### 1. Extract frozen-backbone features (once)

```bash
# NYU depth + normal: 4-layer features at 518×518 (~17 GB cache)
python extract_nyu_features_518.py --mat datasets/nyu_depth_v2/nyu_depth_v2_labeled.mat

# KITTI depth: 4-layer features at 518×518 (~12 GB)
python extract_kitti_features_518.py --root datasets/kitti

# VOC seg: single-layer L23 features at 224×224 (~1 GB)
python extract_voc_features.py --voc-root datasets/voc/VOCdevkit/VOC2012
```

### 2. Train a variant

```bash
# NYU DPT depth (50 epochs ≈ 25 min on H100)
python train_dpt_depth.py --variant DQ  --seed 42
python train_dpt_depth.py --variant DQC --seed 42
python train_dpt_depth.py --variant D1  --seed 42   # baseline

# KITTI DPT depth (same script, KITTI cache)
python train_dpt_kitti.py --variant DQ  --seed 42

# VOC seg (20 epochs ≈ 1-2 min)
python train_seg.py    --variant S_DQ --seed 42
python train_seg.py    --variant S_BN --seed 42   # baseline

# NYU normal (60 epochs ≈ 3-5 min)
python train_normal.py --variant N_DQ --seed 42
python train_normal.py --variant N_BN --seed 42   # baseline
```

### 3. OOD evaluation on NYU depth

```bash
python eval_dpt_ood.py \
  --variants D1 D2 DQ DQC --seeds 42 0 1 \
  --corruptions clean gauss_blur_9 defocus_blur_9 motion_blur_15 \
                contrast_0.5 fog_0.5 shot_noise_30 gauss_noise_0.08 \
                jpeg_20 dark_0.5 \
  --out-suffix _fixed

# Ablation: shuffle register tokens across the batch (tests image-specific
# guidance — if DQ-perf collapses, registers really do encode per-image info)
python eval_dpt_ood.py --variants DQ DQC \
  --seeds 42 0 1 --perm-registers --out-suffix _permq
```

## Current findings (as of 2026-05-23)

### Cross-task summary (the headline)

| task | metric | base | DQ variant | Δ | σ ratio | confirmed? |
|------|--------|------|-----------|----|---------|-----------|
| **VOC seg** | mIoU ↑ | S_BN 0.4421 ± .002 | **S_DQ 0.4743 ± .003** | **+3.22 pp (+7.3 %)** | **≈ 9.5σ** | ✓ |
| **NYU normal** | mean ang err ↓ | N_BN 54.37 ± .07 | **N_DQ 52.96 ± .09** | **−1.41° (−2.6 %)** | **≈ 12σ** | ✓ |
| NYU depth | RMSE ↓ | D1 0.4191 ± .001 | DQ 0.4173 ± .003 | −0.4 % (≈ −0.6σ) | not 2σ | ≈ |
| NYU depth | δ₁ ↑ | D2 0.9356 ± .001 | DQC 0.9383 ± .001 | +0.27 pp | ≈ 2σ | ≈ |

**Headline:** the register-as-query cross-attention residual gives
**clear, statistically confirmed improvements on segmentation and surface
normal**, while on depth it is marginal but in the same direction. The
cross-task replication is the strongest argument that the architectural
choice (cross-attention + register-derived queries) carries real signal
on dense prediction.

> **⚠ Critical caveat for the seg/normal numbers:** the `S_DQ` / `N_DQ`
> heads have ≈ 2.7 M extra parameters from the cross-attention branch on
> top of a ≈ 23 k (seg) / 134 k (normal) `S_BN`/`N_BN` baseline. Part of
> the +7.3 % mIoU / −2.6 % ang-err improvement could be pure capacity.
> The **first planned experiment** (below) is a parameter-matched
> baseline that adds equivalent capacity *without* cross-attention so we
> can isolate the architecture's contribution from raw capacity.

### NYU depth (clean, multi-seed)

| Variant | RMSE ↓ | δ₁ ↑ | vs D1 RMSE | vs D2 RMSE |
|---------|--------|------|-----------|-----------|
| D1 | 0.4191 ± 0.001 | 0.9336 | — | — |
| D2 | 0.4207 ± 0.001 | 0.9356 | +0.4% | — |
| **DQ** | **0.4173 ± 0.003** | 0.9363 | **−0.4%** (not 2σ) | −0.8% (1.5σ) |
| **DQC** | 0.4179 ± 0.003 | **0.9383** | −0.3% | −0.7% |

The clean-test gain is modest (≤1% RMSE, not 2σ confirmed). DQC has the
best δ₁ across all variants.

### NYU OOD-corruption (DQ vs D2, RMSE-based)

| Corruption family | DQ vs D2 | Confirmed? |
|-------------------|----------|-----------|
| Motion blur | **−7.0% RMSE** | ✓ (7.1σ) |
| Defocus blur | −5.4% | ≈ (1.4σ) |
| Gaussian blur | −4.1% | ✓ (2.3σ) |
| Fog | −4.1% | ✓ (5.0σ) |
| Contrast | −1.4% | ✓ (2.7σ) |
| Jpeg | −0.5% | ≈ |
| Dark | −1.2% | ≈ |
| **Gaussian noise** | **+9.3%** (DQ worse) | ✓ (2.7σ) |
| **Shot noise** | **+11.6%** (DQ worse) | ✓ (3.4σ) |

DQ produces a clean **blur+/noise−** dichotomy. Cross-attention with
register-derived queries is helpful when local information is degraded
*coherently* (blur, fog, contrast) but harmful when high-frequency noise
gets amplified through attention.

**However**, the same comparison **vs D1** (no cls readout) is much weaker:
the blur-positive Δ shrinks or reverses, and one corruption (motion blur)
is 2.2σ *worse* for DQ. The DQ "blur improvement" partly reflects D2's
cls-readout weakness rather than DQ's strength.

### KITTI depth (outdoor)

All 5 DPT variants (D1/D2/D7/DF/DQ) on KITTI 800/200 give RMSE ≈ 1.83–2.03
with D2 best. **No register variant beats D2** on KITTI clean RMSE.
Cross-task replication of the OOD blur+/noise− pattern not yet tested on
KITTI.

### What did NOT work

- **DQC** (DQ + cls readout) was hoped to dominate D2 by combining cls's
  noise robustness with DQ's blur robustness. Result: DQC ≈ D2 across all
  corruptions, with no significant difference. With cls present, the
  cross-attention residual is squeezed out.
- **D7** (concat readout with register routing) gave a buggy + 6 % RMSE
  improvement that turned out to be a **double zero-init dead-gate**
  artifact (see `docs/`-equivalent notes in commit history). After
  fixing, D7 ≈ D2.
- **DQ_LQ** (learnable queries instead of registers) actually matched DQ
  on clean — but has an *optimization* advantage (SGD finds task-optimal
  queries) and so is not a clean ablation. `DQ_FRQ` (fixed random
  queries, no learning) is the cleaner control.

## Planned experiments (~2-day budget)

We have ≈ 2 days of compute left. The cross-task signal on seg + normal
is statistically very strong (≥ 9σ), so we are no longer trying to
*find* a signal — we are pressure-testing it against the most obvious
reviewer concerns.

### Priority 1 — parameter-matched capacity control (≈ 4 h, both tasks)

Add an `S_BNW` / `N_BNW` variant: same BN + linear head as `S_BN`/`N_BN`
but **widened with an MLP block of ≈ 2.7 M parameters** so the total
parameter count matches `S_DQ`/`N_DQ` without cross-attention. Train ×
3 seeds on VOC seg + NYU normal.

- If `S_BNW ≈ S_BN ≪ S_DQ`: capacity alone does not explain the gain →
  the cross-attention architecture (and the register-derived queries)
  is responsible. **Story holds.**
- If `S_BNW ≈ S_DQ`: most of the improvement is just capacity → register
  content is irrelevant. **Story collapses to a capacity finding.**

This is the single most-important next experiment because the seg/normal
gain is confounded with a 100× parameter increase.

### Priority 2 — `DQ_FRQ` register-content ablation (≈ 2 h, both tasks)

Cross-attention branch with **fixed random query vectors** (not
register-derived, not learnable; `register_buffer`). Already implemented
as `DQ_FRQ` in `models/dpt.py`; needs analogue `S_FRQ` / `N_FRQ` in
seg/normal heads.

- If `S_DQ ≈ S_FRQ`: cross-attention architecture is what matters, the
  register *content* does not.
- If `S_DQ > S_FRQ`: register content carries useful per-image guidance.

### Priority 3 — `--perm-registers` inference-time ablation (≈ 30 min, free)

`eval_dpt_ood.py --perm-registers` shuffles register tokens across the
batch at inference time (uses the existing trained DQ checkpoint). This
is the cheapest possible test of "does image-specific register content
matter".

### Priority 4 — KITTI cross-task replication (≈ half day)

Re-train DQ/DQC on KITTI (one seed minimum) and re-eval OOD. Confirms
whether the depth-only marginal signal holds outdoors, and whether KITTI
shows the same blur+/noise− pattern as NYU.

### Schedule

| Day | Morning | Afternoon |
|-----|---------|-----------|
| 1   | Param-matched control (P1)  | DQ_FRQ ablation (P2) |
| 2   | Perm-registers eval (P3) + KITTI re-eval (P4) | Compile final results, write paper draft |

## Possible outcomes and how we will write up

The experiments are designed to be **falsifiable**. We commit to writing
up the result whichever way it goes:

- **Outcome A — positive across tasks (DQ > base on depth + seg + normal,
  blur+/noise− replicates):** mechanism paper. Title: *Register
  Cross-Attention Reveals a Blur/Noise Frequency Trade-off in Frozen
  DINOv2 Dense Prediction*.
- **Outcome B — mixed (positive on some tasks, null on others):**
  publish as a calibration study. Title: *When Do DINOv2 Register Tokens
  Help Downstream Dense Prediction? A Multi-Task, Multi-Corruption
  Study*. Honest about which task / corruption family the gain depends
  on.
- **Outcome C — null / negative (DQ ≈ base everywhere, or
  ablation says register content is irrelevant):** publish as a negative
  result. Title: *Frozen DINOv2 Register Tokens Are Not a Free Lunch:
  Architecture-Confounded Gains Vanish Under Clean Ablation*. Documents
  the dead-gate trap (which is reproducible community knowledge) and the
  init-order confound across 4 register-integration variants.

All three outcomes are publishable; we are not biased toward A.

## Caveats and limitations

- **Frozen backbone, small training subset** (1149 NYU labeled, 1149 VOC
  trainval-80%, 800 KITTI val_selection-80%). Absolute numbers are
  ~10–20% above the SOTA full-data setup. We rely on **relative
  comparison across variants with the same setup** for our conclusions.
- We use the **NYU labeled 1449 split** for both depth and normal (with
  normals computed from depth via finite differences), not the official
  NYU 654 test split. The 80/20 split is fixed by seed = 42 throughout.
- Single-resolution (518×518 square) features for DPT. Official DPT uses
  multi-scale and aspect-preserving crops.
- All seeds use the same data shuffle order (controlled by
  `--seed` only).

## Publishing to GitHub

```bash
cd register-as-query/
git init
git add .
git commit -m "Initial commit: register-as-query for dense prediction"

# create a new empty repo on github.com, then:
git remote add origin git@github.com:<your-user>/register-as-query.git
git branch -M main
git push -u origin main
```

The `.gitignore` already excludes the heavy `data/` cache (~17 GB) and
the per-run `best.pt` checkpoints (~100 MB each). Only code +
`summary.json` outputs are committed.

## Acknowledgements

- DINOv2 backbone code from Meta's [official DINOv2 repo](https://github.com/facebookresearch/dinov2).
- DPT decoder architecture follows
  [Ranftl et al. 2021](https://arxiv.org/abs/2103.13413) and the official
  DINOv2 `dinov2/eval/depth/` reference implementation.
- Cross-attention with N queries (register-as-query design) inspired by
  [Mask2Former](https://arxiv.org/abs/2112.01527).

