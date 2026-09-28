# HERAMBA-CCA — Haptics, Seed 42 — Evidence Report

**Experiment**: replace the validation-searched mixing ratio rho with an
explicit shared-subspace decomposition. Model = `[MiniROCKET G || H_unique]`
where `H_unique` = the part of the HERAMBA/R2 representation H linearly
uncorrelated with the G-H shared canonical subspace. **No rho anywhere.**

---

## 1. Setup and identity gates

- Frozen canonical banks (seed 42, 155 dev / 308 test, 4998 + 4998):
  G = MiniROCKET global PPV features; H = R2 regime-conditioned
  heterogeneity (the exact R2 representation, reused unchanged).
- **Identity gate passed**: `[G||H]` + canonical RidgeClassifierCV on
  train+val reproduces the stored R2 result **exactly** (test Macro-F1
  = 0.5500, alpha = 4.2813). So the inputs to this experiment are the
  canonical R2 representations, bit-faithful.
- M0 reference reproduced on the same banks: 0.5037 on G-only
  (dev-fit convention of this experiment; the canonical 132/23/308
  MiniROCKET M0 = 0.4974 from the external-stack run; G-only here is
  fit on train+val with the G bank = first 4998 MiniROCKET features,
  hence the small difference is expected and documented).

## 2. CCA analysis (train-only, label-free)

- Predeclared reduction: numerical-rank PCA (energy 1-1e-10) fitted on
  TRAIN only -> k_G = k_H = 131 of 132 (no real information discarded;
  the rule is the most conservative possible).
- CCA via whitened cross-covariance SVD on the reduced views.
- Result: **all 131 canonical correlations are 1.000000** (up to
  numerical precision).

Interpretation: when CCA is allowed the full statistical rank of the
training data (n=132, both views effectively 131-dimensional), the two
representations span **identical linear subspaces on the training set**.
This is the expected geometry of two n-limited feature banks: with more
features than samples, both banks span R^n (almost surely), so CCA finds
a perfect linear alignment. This is a statement about the *sample
geometry*, NOT evidence that H's information content equals G's.

## 3. Shared-subspace rule and H_unique

- Predeclared rule (fixed before any evaluation): component j is shared
  iff rho_j >= 0.5. With rho_j = 1.0 everywhere, n_shared = 131.
- H_unique = OLS residual of H (reduced space) on the 131 shared
  canonical scores, fitted on TRAIN, frozen, applied unchanged to
  val/test. Numerical orthogonality verified: **max |cos| = 0.0**
  (exact, by OLS construction).
- Consequence: H_unique has **zero variance in every direction** —
  effective rank 1.0 (degenerate), fraction of H variance unique = 0.0,
  shared = 1.0.

## 4. Diagnostics

| Representation | dim | numerical rank | effective rank |
|---|---|---|---|
| G (reduced) | 131 | 131 | 68.16 |
| H (reduced) | 131 | 131 | 105.31 |
| H_unique    | 131 | 0   | 1.0 (degenerate) |

- Variance partition: shared 1.0, unique 0.0.
- The CCA-perfect geometry means the residualizer removes *everything*;
  this is correct behavior given the rule + geometry, and it
  demonstrates why a rank-limited CCA (not full-rank) would be needed
  to leave a non-degenerate H_unique.

## 5. Label-aware complementarity (dev-only)

- Paired 5-fold CV on the dev set (155 samples): G-only CV Macro-F1
  = 0.4575; [G, H_unique] = 0.4575. **Increment = +0.0000** — trivially,
  since H_unique is identically ~0.
- Permutation null (2000 column permutations, seed 42042): null mean
  0.0000, empirical p = 1.0. The test is uninformative here because the
  augmentation adds nothing (zero-variance block) — reported for
  completeness, not as evidence of absence.

## 6. Final comparison (single test evaluation each)

| Representation | Features | Macro-F1 | Accuracy | alpha |
|---|---|---|---|---|
| MiniROCKET (G only) | 4998 | 0.5037 | 0.5162 | 4.281 |
| R2/R5/HERAMBA [G||H] | 9996 | **0.5500** | 0.5649 | 4.281 |
| CCA-HERAMBA [G||H_unique] | 5129 | 0.5037 | 0.5162 | 4.281 |

CCA-HERAMBA degenerates to G-only because H_unique is empty. H itself
carries +0.0463 Macro-F1 over G alone (0.5500 vs 0.5037) — consistent
with the earlier inference study (G-only 0.5037 / H-only 0.4686 /
G+H 0.5500).

## 7. Scientific interpretation (outcome classification)

Per the predeclared interpretation categories, the raw result pattern is
closest to **outcome 6-adjacent / non-informative decomposition**: not an
implementation failure (all gates and tests pass; the math is exact) but
a **specification degeneracy**: with n=132 and full-rank CCA, "the part
of H not linearly aligned with G on the training sample" is the empty
set.

What this experiment does establish:
1. G and H have **perfectly overlapping sample-level linear geometry**
   on the training data (all CCA rhos = 1) — full-rank CCA cannot
   isolate a complementary subspace at this n.
2. H still contains label-relevant information beyond G (G+H 0.5500 >
   G 0.5037; permutation p = 0.0117 on the paired test in the earlier
   inference study), so the complementary information exists
   predictively but is *not separable* by linear full-rank CCA at n=132.

To make the decomposition informative, the shared-subspace definition
must be rank-limited (e.g., restrict CCA to the top-k variance
components of each bank with a predeclared k < n, so "shared" means
"aligned with the dominant, reliable directions of G" and the residual
retains H's low-variance directions). That is a specification change,
deliberately NOT made mid-experiment (Phase 12 rule: no redesign after
seeing results).

## 8. Answers to the evidence questions (Phase 13)

1. How similar are G and H? — Sample-geometry: identical linear span
   (CCA rho = 1.0 for all 131 components). Energy profile differs
   (eff rank 68 vs 105).
2. CCA spectrum? — Flat at 1.000 (see figures/cca_spectrum.png).
3. Shared dimensions? — 131 (all), by the predeclared tau=0.5 rule.
4. Fraction of H removed as shared? — 1.0 (all).
5. Effective rank H? — 105.31 (reduced), 67.0 (raw, earlier analysis).
6. Effective rank H_unique? — 1.0 (degenerate/empty).
7. Is H_unique predictive of Y? — Untestable: it has zero variance.
8. Incremental information beyond G? — None from H_unique (+0.0000, p=1.0).
9. Does [G,H_unique] improve over G alone? — No (identical).
10. Does it improve over R2/R5? — No (0.5037 vs 0.5500).
11. Seed uncertainty? — NOT established: single seed 42, single split.

## 9. Sanity checks (Phase 16) — all pass

- CCA fitted on TRAIN only (132 rows) — asserted in code.
- No test/val samples in any fit; transforms frozen then applied.
- No labels in any decomposition fit (verified by label-permutation
  invariance unit test).
- H_unique numerically orthogonal to shared scores (max |cos| = 0.0).
- No NaN/Inf (asserted in model.transform).
- Deterministic (repeat run reproduces identical transforms).
- MiniROCKET untouched (aeon canonical, verified earlier against all
  four reference datasets exactly).
- R2/R5 untouched (identity gate: [G||H] -> 0.5500 exact).
- Test evaluated once per model, after freezing.
- Macro-F1 cross-checked against stored canonical values.

## 10. Files

- Model package: `models/heramba_cca/{cca,model,diagnostics,complementarity}.py`
- Runner: `experiments/heramba_cca_haptics_seed42/runner.py`
- Tests: `tests/test_heramba_cca.py` (7 passed)
- Artifacts: this directory (`cca_summary.json`,
  `canonical_correlations.csv`, `diagnostics.json`, `effective_rank.csv`,
  `complementarity.json`, `complementarity_evidence.csv`,
  `final_comparison.csv`, `results.json`, `predictions_*.npy`,
  `figures/` — 4 figures, PDF+PNG).
- Reproduce: `python experiments/heramba_cca_haptics_seed42/runner.py`
  (banks are cached; ~5 min end-to-end).
