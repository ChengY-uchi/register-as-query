# Results Summary

Consolidated multi-seed results across the four dense prediction tasks.
All numbers are from the **frozen DINOv2-L-reg backbone**, with only the
head/readout differing per variant.

**Stats convention:** every cell is `mean ± std` over **seeds 42, 0, 1**
unless stated otherwise. A "confirmed" Δ has `|Δ| / pooled std ≥ 2.0`.

---

## 1. NYU depth (1159 train / 290 test, 50 epochs DPT)

### 1.1 Clean test (multi-seed)

| variant | RMSE ↓ | δ₁ ↑ | AbsRel ↓ | n_seeds |
|---------|--------|------|----------|--------:|
| D1 (patch only) | 0.4191 ± 0.0006 | 0.9336 ± 0.0012 | — | 3 |
| D2 (patch + cls) | 0.4207 ± 0.0008 | 0.9356 ± 0.0005 | — | 3 |
| D7 (patch + cls + reg_routed) | 0.4225 ± 0.0003 | 0.9337 ± 0.0006 | — | 3 |
| DF (FiLM modulation) | 0.4198 | 0.9355 | — | 1 |
| **DQ (register cross-attn)** | **0.4173 ± 0.0021** | 0.9363 ± 0.0007 | — | 3 |
| **DQC (D2 base + DQ residual)** | 0.4179 ± 0.0027 | **0.9383 ± 0.0002** | — | 3 |

**Reading:** DQ has the lowest mean RMSE; DQC has the highest mean δ₁.
None of the gains over D1 reach the 2σ confirmation bar on clean.

### 1.2 OOD corruption (DQ vs D2, RMSE-based, 3 seeds, multi-seed)

See `nyu_dpt/ood_table_fixed.md` for the full 5-variant × 10-corruption
table. Headline rows:

| corruption family | DQ vs D2 ΔRMSE | σ ratio | confirmed |
|-------------------|----------------|---------|:---------:|
| motion_blur_15 | **−7.0 %** | −7.1σ | ✓ |
| defocus_blur_9 | −5.4 % | −1.4σ | ≈ |
| gauss_blur_9 | −4.1 % | −2.3σ | ✓ |
| fog_0.5 | −4.1 % | −5.0σ | ✓ |
| contrast_0.5 | −1.4 % | −2.7σ | ✓ |
| jpeg_20 | −0.5 % | −1.4σ | ≈ |
| dark_0.5 | −1.2 % | −1.6σ | ≈ |
| gauss_noise_0.08 | **+9.3 %** (worse) | +2.7σ | ✓ |
| shot_noise_30 | **+11.6 %** (worse) | +3.4σ | ✓ |

**Pattern:** cross-attention with register-derived queries is a clean
**blur+/noise−** trade-off: low-frequency degradations (blur, fog,
contrast) are helped; high-frequency noise is hurt.

### 1.3 Important caveat for the depth story

The same comparison **vs D1** (no cls in readout) is much weaker — the
blur-positive Δ shrinks, and motion_blur is 2.2σ *worse* for DQ. Part of
the "DQ improvement on blur" reflects D2's cls-readout weakness, not DQ's
strength. See `nyu_dpt/ood_table_fixed.md` for the per-baseline tables.

---

## 2. KITTI depth (800 train / 200 test, 50 epochs DPT, seed 42 only)

| variant | RMSE ↓ | AbsRel ↓ | δ₁ ↑ |
|---------|--------|----------|------|
| D1 | 1.957 | 0.0332 | 0.9898 |
| **D2** | **1.831** | **0.0290** | **0.9915** |
| D7 | 1.891 | 0.0334 | 0.9913 |
| DF | 2.028 | 0.0390 | 0.9904 |
| DQ (buggy ckpt removed; multi-seed retrain pending) | — | — | — |

D2 dominates on KITTI; no register variant beats D2 at single seed. The
DQ multi-seed re-eval after the dead-gate fix is a P4 planned experiment.

---

## 3. VOC 2012 segmentation (1159 train / 290 test, 20 epochs, single-layer L23)

### 3.1 Main comparison (S1 / S_BN / S_DQ, 3 seeds)

| variant | mIoU ↑ | Δ vs S_BN | params |
|---------|--------|-----------|-------:|
| S1 (patch only, no BN)            | 0.4055 ± 0.0007 | −0.037 | 21.5 k |
| **S_BN (patch + BN, official-style)** | 0.4421 ± 0.0023 | — | 23.6 k |
| **S_DQ (S_BN + register cross-attn)** | **0.4743 ± 0.0025** | **+0.0322 (+7.3 %, ≈ 9.5σ ✓)** | 2.79 M |

`S_DQ` improves over the BN baseline by **+3.22 mIoU**, statistically
strong (≈9.5σ across 3 seeds). **Confounder:** the DQ branch adds ~120 ×
more parameters, so part of the gain is plausibly capacity, not cross
attention specifically. Priority-1 experiment (param-matched `S_BNW`)
will isolate this.

### 3.2 Older single-layer variants (S2–S7, kept for context)

These were exploratory variants from earlier phases; the main story is
the S_BN / S_DQ comparison. (All 3 seeds; mIoU mean only for brevity.)

| variant | feature construction | mIoU mean |
|---------|----------------------|-----------|
| S2 | [patch ; cls] (2D)              | ≈ 0.408 |
| S3 | [patch ; reg_mean] (2D)         | ≈ 0.378 |
| S4 | [patch ; reg_routed (pos-only)] | ≈ 0.366 |
| S5 | [patch ; reg_routed (content)]  | ≈ 0.364 |
| S6 | reg_routed only (no patches)    | ≈ 0.184 (diagnostic) |
| S7 | [patch ; cls ; reg_routed]      | ≈ 0.378 |

Among the concat-readout variants, none of the register integrations
beat S2 (patch + cls) — only the **cross-attention** integration in S_DQ
does.

---

## 4. NYU surface normal (derived from depth, 1159/290, 60 epochs, single-layer L23)

### 4.1 Main comparison (N1 / N_BN / N_DQ, 3 seeds)

| variant | mean ang err ↓ | median ang err ↓ | acc@11.25° ↑ | acc@22.5° ↑ | acc@30° ↑ | params |
|---------|----------------|------------------|--------------|-------------|-----------|-------:|
| N1 (patch only)                  | 54.13 ± 0.08 | 49.96 ± 0.25 | 0.0557 | 0.1725 | 0.2610 | 131.6 k |
| N_BN (patch + BN)                | 54.37 ± 0.07 | 49.94 ± 0.23 | 0.0484 | 0.1615 | 0.2524 | 133.6 k |
| **N_DQ (N_BN + register cross-attn)** | **52.96 ± 0.09** | **47.69 ± 0.17** | **0.0542** | **0.1779** | **0.2745** | 2.90 M |

`N_DQ` improves mean angular error by **−1.41°** over N_BN
(−2.6 %, ≈ 12σ across 3 seeds). Same capacity caveat as seg: N_DQ has
≈ 22 × more parameters than N_BN.

### 4.2 Older single-layer variants (N1–N7)

| variant | mean ang err mean |
|---------|-------------------|
| N1 | 54.13 |
| N2 | ≈ 54.5 |
| N3 | ≈ 54.2 |
| N4 | ≈ 54.3 |
| N7 | 53.30 (best of N1–N7) |

Within concat-style register integration (N7), the improvement over N1
is small (−0.83°); the cross-attention N_DQ is dramatically larger.

---

## 5. Cross-task headline

| task | metric | base | DQ variant | Δ | σ ratio | confirmed | param gap |
|------|--------|------|-----------|----|---------|:---------:|----------:|
| VOC seg | mIoU ↑ | S_BN 0.4421 | S_DQ 0.4743 | +0.032 (+7.3 %) | 9.5σ | ✓ | 120× |
| NYU normal | mean ang err ↓ | N_BN 54.37 | N_DQ 52.96 | −1.41° (−2.6 %) | 12σ | ✓ | 22× |
| NYU depth | RMSE ↓ | D1 0.4191 | DQ 0.4173 | −0.002 (−0.4 %) | 0.6σ | ≈ | 1.4× |
| NYU depth | δ₁ ↑ | D2 0.9356 | DQ_C 0.9383 | +0.003 | 2σ | ≈ | — |

The cross-task replication is strong on seg + normal but weak on depth.
Two interpretations to disambiguate via P1 (param-matched control):
1. **Capacity hypothesis:** seg/normal gains are mostly extra params; DQ
   architecture per se does not matter. Depth gain is true-but-small
   because the DPT baseline already has high capacity in the readout.
2. **Architecture hypothesis:** cross-attention with register-derived
   queries is genuinely better, and the depth gain is small because the
   readout space is already saturated by the DPT decoder's other
   parameters.

---

## 6. What is in this directory

```
results/
├── SUMMARY.md                            # this file
├── nyu_dpt/
│   ├── ood_table_fixed.md/.json         # 5-variant × 10-corruption OOD table
│   └── (summary.json per run intentionally omitted; re-train to regen)
├── kitti_dpt/ … voc_seg/ … nyu_normal/  # (same: cleaned, re-train to regen)
```

Per-run `summary.json` files (loss curves, per-epoch metrics) and
checkpoints (`best.pt`, ~100 MB each) are **not** committed. Re-train
any variant in 1–25 minutes on a single H100/L40S to regenerate them.
