# RPMS — Regime-Partitioned Multi-Statistic Kernel Bank (Haptics, seed 42)

## 1. Objective

Test whether combining three complementary pooled views of the same fixed
MiniROCKET kernel responses — canonical global PPV (G), the validated R2
regime-PPV heterogeneity (H), and a NEW regime-conditional Hydra
competitive dispersion (HydraH) — improves classification over the audited
R2 representation under an identical 9996-feature budget, on Haptics,
seed 42, single dataset / single seed / single official test evaluation.

## 2. Scientific hypothesis

Hydra's competitive (order/competition) signal independently beat
MiniROCKET in the project's ensemble test, while R2's H_m is the one
validated regime-conditioned statistic. If competition and
magnitude-thresholding carry complementary information about the same
learned temporal regimes, then aggregating per-regime win-rate dispersion
with the SAME safe H-form aggregation should add predictive information
beyond G + H at equal budget.

## 3. RPMS architecture

```
z-normed series (T=1092)
        |
   +----+------------------+
   |    |                  |
   v    v                  v
MiniROCKET  frozen R2     Hydra bank (_HydraInternal
(9996)      encoder->VQ   k=8, g=64, re-seeded, frozen)
   |    K=8 regimes k(t)      |
   |    |                     | winners per (group, t)
   +----+---------------------+
   |    |                     |
   v    v                     v
G: PPV   H: sum_k q_k      HydraH: sum_r q_r (W_m,r - W_m)^2
        (PPV_m,k-PPV_m)^2      (win-rate dispersion)
   \_____|_____________________/
         v
   RidgeClassifierCV (logspace(-4,4,20)), train+val fit
```

No raw-response modulation, no r̃_m(t), no gates, no raw Hydra counts, no
SparseScaler, no output-level combination. Everything entering Ridge is a
pooled per-sample scalar.

## 4. Equal-budget design

Total budget 9996 = 3332 (G) + 3332 (H) + 3332 (HydraH). Allocation is
deterministic and performance-blind: first 3332 canonical MiniROCKET
features (G), first 3332 het features (H), first 3332 Hydra units in
canonical (dilation, diff-pass, group, kernel) order (HydraH, out of 4096
available units; g=64 chosen exactly so 8×2×32×8 ≥ 3332). No validation-
or test-based redistribution; the allocation was fixed before any fit.

## 5. Three branch definitions

- **G (Branch 1)**: canonical PPV_m over the per-feature valid region
  [padding_m, T−padding_m), exact canonical thresholds b_m. Verified
  identical to aeon MiniROCKET output (audit diff = 0.0).
- **H (Branch 2)**: the exact audited R2 statistic
  H_m = Σ_k q_k (PPV_{m,k} − PPV_m)², computed via the audited
  `heterogeneity_features` (valid-region grouped, min-occupancy 0.01·T,
  q renormalized over selected regimes). Independent recompute diff
  3.49e-09 (established float32/64 tolerance).
- **HydraH (Branch 3, NEW)**: for each Hydra unit u = (dilation d,
  diff-pass j, group g, kernel k): winner series w_u(t) =
  argmax_k Z[g, k, t] over the unit's valid window (input centers
  [4d, L_in−1−4d], the exact analogue of MiniRocket's padding convention),
  regime-conditioned win rates W_{m,r} = c_{u,k,r}/occ_r, global win rate
  W_m = c_{u,k}/n_valid, then
  **H^Hydra_u = Σ_r q_r (W_{m,r} − W_m)²** with the identical
  min-occupancy + renormalization semantics as H_m. Winner extraction
  uses aeon's exact `_HydraInternal` competitive rule (argmax over the
  k-axis = Hydra's count_max winner); the documented deviation is hard
  0/1 wins instead of Hydra's magnitude-weighted scatter, chosen because
  the dispersion statistic needs rates, not magnitudes.

## 6. Training procedure

No temporal-context training: RPMS reuses the **frozen audited R2
checkpoint** `results/rcmkn_haptics_seed42/context_model_seed42.pt`
(sha256[:16] `2dbb0cf4f3db3df8`, gated against the manifest; its
load+extract chain is the same one whose R2.2 B0 reproduced R2=0.5500
exactly). Regimes are extracted deterministically (byte-identical
re-extraction verified). HydraH has no learned parameters by design
(counts + closed-form dispersion). RidgeClassifierCV(np.logspace(-4,4,20))
fit on train+validation for the official model; Ridge alpha selection
used validation only.

## 7. Scientific information tracking

All nine tracks computed on train/validation only (n=155; test never
touched by any diagnostic). See §8-§10 and
`information_diagnostics.json` / `branch_metrics.json`.

## 8. Validation branch decomposition (Track 3 + Track 4)

| Model | val Macro-F1 | alpha | dim |
|---|---|---|---|
| G | 0.4633 | 4.28 | 3332 |
| H | 0.5600 | 0.234 | 3332 |
| HydraH | 0.2587 | 1.62 | 3332 |
| G+H | **0.6767** | 1.62 | 6664 |
| G+HydraH | 0.5660 | 4.28 | 6664 |
| H+HydraH | 0.3756 | 0.616 | 6664 |
| **G+H+HydraH (official)** | 0.5444 | 1.62 | 9996 |

Incremental validation gains: Δ(HydraH | G+H) = **−0.1323**;
Δ(H | G+HydraH) = −0.0216; Δ(G | H+HydraH) = +0.1688.

Reading: H alone already beats G alone; G+H is the best validation model
by a wide margin; **adding HydraH to G+H degrades validation by 13.2
points** — HydraH is not merely uninformative here, it is harmful at
equal budget (its 3332 dilution displaces G/H features inside the fixed
9996).

## 9. Redundancy analysis (Tracks 1, 2, 8)

- Scale/rank: G mean 0.501 std 0.293 eff-rank 69; H mean 0.0385 eff-rank
  136; HydraH mean 0.0095, 23.9% near-zero entries, eff-rank 137.
- Between-branch similarity: G~H pearson −0.258 / CKA 0.346;
  G~HydraH pearson −0.015 / CKA 0.225; H~HydraH pearson +0.006 / CKA
  0.353.
- Track 8 (same-kernel-set correlation): corr(H_m, HydraH_m) mean −0.007,
  median −0.006, 0.1% strongly correlated.

HydraH is **statistically novel, not redundant** — it measures something
the other two blocks do not. The failure is predictive, not informational
overlap: its novel signal does not help (and at 1/3 budget, hurts).

## 10. VQ diagnostics (Tracks 6, 7)

- 8/8 codes active, 0 dead; normalized entropy 0.935; perplexity 6.99;
  dominant-code fraction 0.236 (counts [39972, 1565, 24202, 19122, 23230,
  20936, 17687, 22546]). Code 1 is rare but alive. Revival count: N/A by
  design (frozen context — no VQ training occurs inside RPMS).
- Per-regime contribution to H: dominant k=6 (0.00776); all regimes
  contribute materially except rare k=1 (0.00027).
- Per-regime contribution to HydraH: dominant k=0 (0.002386); regime 1
  ≈ 0 (6e-06, consistent with its 1.6% occupancy).

## 11. Official test result

Exactly ONE official test evaluation (see provenance note in
`report.json`):

| Model | val MF1 | test Macro-F1 | accuracy |
|---|---|---|---|
| RPMS = G+H+HydraH (3332/3332/3332) | 0.5444 | **0.4765** | 0.4903 |

Class F1s: [0.333, 0.475, 0.500, 0.519, 0.556].

## 12. Comparison to canonical M0 and R2 (equal 9996 budget)

| Reference | test Macro-F1 | RPMS − ref |
|---|---|---|
| canonical M0 (historical) | 0.4974 | −0.0209 |
| audited R2 (frozen context, same budget) | 0.5500 | **−0.0735** |

RPMS falls **below both** references — below even canonical MiniROCKET.
Note the same-encoder advantage of R2: identical frozen context, so the
−0.0735 gap isolates exactly the effect of swapping 1666 G features + 1666
H features for 3332 HydraH features.

## 13. Which branch contributed what

- **H is the workhorse**: best single branch on validation (0.5600 vs
  G 0.4633, HydraH 0.2587); G+H validation (0.6767) is the strongest
  model observed anywhere in this experiment.
- **G is a strong second** (largest LOBO increment, +0.169), consistent
  with its canonical role.
- **HydraH contributed negatively at equal budget** on every validation
  combination it joined (G+HydraH < G? no — 0.5660 > 0.4633, but
  G+HydraH < H alone; H+HydraH 0.3756 < H 0.5600; G+H+HydraH 0.5444 <
  G+H 0.6767). As a standalone block it is weak (0.2587).

## 14. Whether branches appear complementary

No — not in the predictive sense, despite statistical novelty. Q1: RPMS
does NOT beat R2 (0.4765 < 0.5500). Q2: HydraH does NOT add information
beyond G+H on validation (−0.132). Q3: HydraH is *statistically*
complementary (corr ≈ 0, CKA 0.35) but *predictively* antagonistic in
this budget regime. Q4: the one-third allocation is not justified: the
full-budget G+H (6664) beat the official 9996 RPMS on validation, and
R2's 4998 H allocation beat RPMS's 3332 H allocation on test. On
Haptics' tiny train set (132), the marginal information HydraH adds is
outweighed by the budget it consumes.

## 15. Limitations

- Single dataset, single seed, single official test number; no variance
  estimate.
- Haptics is extreme n≪p (132 train rows / 9996 features); the Ridge
  shrinkage regime is exactly where a weak extra block hurts most. The
  negative result is a *budget under scarcity* verdict, not proof that
  HydraH carries no signal anywhere.
- HydraH hard-win counting deviates from Hydra's magnitude-weighted
  count_max; the "safe" H-aggregation may also discard magnitude
  information that makes Hydra work in its native form.
- Kernel/unit allocation is the first-3332 canonical prefix; other
  deterministic allocations were not explored (by design — no selection).
- The frozen R2 context couples RPMS to one regime partition; a jointly
  trained context could partition differently (out of scope).

## 16. Reproducibility

```bash
cd ECG_Benchmark
python -m pytest tests/test_rpms_haptics_seed42.py -q      # 16 tests
python -m pytest tests/ -q -k "rpms or rcmkn or drtn or conditioned"  # 239
python -m experiments.rpms_haptics_seed42.runner           # official run
python -m experiments.rpms_haptics_seed42.figures          # 9 figures
```

Seed 42 everywhere (torch/numpy/RandomState; Hydra bank re-seeded
independently, frozen; context checkpoint frozen). Artifacts: `report.json`,
`config.json`, `branch_metrics.json`, `information_diagnostics.json`,
`validation_ablation.json`, `predictions/`, `diagnostics/`, `checkpoints/`,
`figures/f1..f9`. Runtime ≈ 37 s (CPU).

**Bottom line**: the RPMS hypothesis is NOT supported. HydraH is a
genuinely novel, statistically complementary view of the same learned
regimes, but at the fixed 9996 budget on Haptics it degrades both
validation and test performance; the validated R2 (G 4998 + H 4998)
remains the strongest equal-budget representation. Per the stop rule:
no further datasets, seeds, or variants.
