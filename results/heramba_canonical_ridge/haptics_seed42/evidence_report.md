# CCA-Adaptive Generalized Ridge ("Canonical Ridge") — Haptics, Seed 42 — Evidence Report

> **Status of the method**: the shrinkage rule and the generalized-ridge
> classification formulation implemented here are a **PROPOSED
> experimental model for this project**. They are NOT the classical
> canonical ridge of Vinod (1976) — that method regularizes CCA view
> projections with scalar penalties; no established literature formula
> with direction-specific penalties `alpha_k = alpha_base * (1 + gamma *
> rho_k^2)` was found or claimed. All artifacts label it "proposed".

## 1. Research question

Can **direction-specific shrinkage in the CCA basis** retain the useful
shared structure of HERAMBA (H) while still exploiting its genuinely
complementary structure — improving on both the hard discard of H_shared
and the blind raw `[G||H]` concatenation, without any mixing ratio and
without validation/test tuning of the decomposition?

## 2. Why hard CCA projection is insufficient

The rank-controlled experiment (`results/heramba_cca_ranked/`) showed:
"shared" (rho >= 0.5) does **not** mean useless — all 29 canonical
directions had rho 0.674–0.839 and ALL were labelled shared by the tau
rule, yet discarding them (`[G||H_unique]` = 0.5138) underperformed raw
concatenation (`[G||H]` = 0.5500) by 3.6 pp. A soft, monotone penalty on
near-duplicated directions is the principled middle ground.

## 3. Model definition

Representation (every fit on TRAIN = 132 rows only):

```
G_raw --trainPCA(95%)--> G_low (29)
H_raw --trainPCA(95%)--> H_low (88)
CCA(G_low, H_low)  ->  rho_1..rho_29, B (H-side canonical directions)
H_CCA  = whitened(H_low) @ B                (29 canonical coordinates)
H_perp = whitened(H_low) @ Q_perp           (59 orthogonal complement)
Z = [G_low || H_CCA || H_perp]              (117 features)
```

Classifier — generalized ridge via one-hot least squares, solved on the
dev split (train+val, canonical final-fit convention):

```
min_W ||Y - Z W||_F^2 + tr(W^T Lambda W),
Lambda = diag(alpha_base*1_29, alpha_1..alpha_29, alpha_base*1_59)
Y = one-hot indicator matrix (classes fixed from TRAIN)
W = (Z^T Z + Lambda)^{-1} Z^T Y   (solved by eigendecomposition of the
SPD pencil G v = theta Lambda v; NO explicit inverse is formed)
prediction = argmax_c (Z W)_c
```

## 4. Exact mathematical objective

`min_W ||Y - Z_c W||^2 + W^T (c * Lambda_0) W` with `Z_c` train-centered,
`Lambda_0 = diag(1,...,1, 1+rho_1^2, ..., 1+rho_29^2, 1,...,1)`, and the
global scale `c = alpha_base` the ONLY tuned scalar.

## 5. Exact alpha_k formula

```
alpha_k = alpha_base * (1 + gamma * rho_k^2),  gamma = 1.0 (FIXED, not tuned)
```

- rho low -> ~baseline penalty; rho -> 1 -> ~2x baseline (bounded, never
  discarded). alpha_perp = alpha_G = alpha_base.
- Verified exactly by unit test (`alpha_schedule`).

## 6. Why shared directions are retained, not discarded

rho_k^2 is the fraction of the canonical pair's variance that is
cross-view explained; a direction duplicated in G deserves *less trust
as independent H information* and hence *more shrinkage*, but its H-side
coordinates may still carry H-specific signal (rho < 1 leaves
orthogonal H variance in every direction). Hard thresholds destroy that
signal; the bounded multiplicative penalty only discounts it. The
basis construction is numerically exact: reconstruction error
`|H_CCA@B^T + H_perp@Q_perp^T - whitened H_low|` = **4.3e-15**; all
canonical directions remain available to the classifier.

## 7. Train/val/test protocol

- PCA (95% variance rule, predeclared), CCA, canonical basis: TRAIN only.
- alpha_base: **train-only GCV** (`GCV = n*RSS/(n-df)^2`,
  `df = sum theta_k/(theta_k+c)`), grid logspace(-4,4,81), on the dev
  fit. No validation or test information touches the selection.
- Final fit on dev (train+val, canonical convention); test evaluated
  ONCE per method. Validation Macro-F1 reported, never used to select.
- gamma, tau-free basis, ranks: all predeclared mathematical constants.

## 8. Leakage audit

- All decomposition fits: TRAIN rows only (asserted; unit tests 2-3).
- Labels: never enter PCA/CCA/basis (unit tests 3, 7).
- alpha_base: GCV on train rows only (dev rows = train+val for the
  final fit — the canonical convention used by every baseline here;
  no test labels anywhere before final scoring).
- Deterministic: identical repeat runs (unit tests 1, 8).
- Previous experiment artifacts byte-unchanged (unit-tested in the
  ranked suite; nothing in this run writes outside the new namespace).

## 9. Baseline results (same split, seed 42)

| Method | Test Macro-F1 | gate |
|---|---|---|
| A. MiniROCKET (G) | **0.5037** | exact match ✓ |
| B. Raw [G\|\|H] (canonical R2) | **0.5500** | exact match ✓ |
| C. Hard CCA [G\|\|H_unique] | **0.5138** | exact match ✓ |

## 10. Canonical Ridge results (proposed model)

| Variant | Val Macro-F1 | Test Macro-F1 | alpha_base (GCV) | df | dims |
|---|---|---|---|---|---|
| E. Uniform reduced gen-ridge | 0.7433 | **0.4680** | 630.96 | 16.3 | 29+29+59 |
| F. Adaptive canonical ridge (proposed) | 0.7433 | **0.4825** | 501.19 | 17.5 | 29+29+59 |

Deltas of the proposed model: **−2.12 pp vs MiniROCKET, −6.75 pp vs raw
[G||H], −3.13 pp vs hard CCA unique; +1.45 pp vs uniform gen-ridge.**

## 11. Uniform-vs-adaptive ablation

The adaptive schedule beats its own uniform control on test
(+1.45 pp) and is selected at a slightly smaller alpha_base with more
effective df — the direction-specific penalty does change behavior in
the intended direction. But **both reduced-representation variants fall
below every full-representation baseline**: compressing G to 29
dimensions costs more than any benefit from basis-adaptive shrinkage on
this tiny-n dataset.

## 12. Canonical rho and alpha table (excerpt; full CSV in artifacts)

| k | rho | alpha_k (adaptive) | null mean | p (S=500) |
|---|---|---|---|---|
| 1 | 0.8391 | 852.6 | 0.372 | 0.002 |
| 2 | 0.8267 | 840.0 | 0.372 | 0.002 |
| ... | ... | ... | ... | ... |
| 29 | 0.6740 | 728.5 | 0.372 | 0.002 |

All 29 directions are significant against the train-only permutation
null (28/29 at p<=0.05, most at the floor (1+0)/501 = 0.002); the CCA
alignment is real, not finite-sample inflation, though rho values
themselves remain upward-biased at n=132 (null mean ~0.37).

## 13. Numerical stability diagnostics

Both variants: rank(Z) = 117 (full), cond(Z) = 110.4,
cond(Λ-regularized system) ≈ 2.1–2.9, min eigenvalue of the regularized
system ≈ 501–631 (very well conditioned), Cholesky succeeded, no
fallback solver needed, reconstruction error 4.3e-15. Solver verified
exact against brute-force normal equations to 1e-8 (unit test 9).

## 14. Permutation-null CCA results

TRAIN-only diagnostic (S=500, seed 52042, model-independent): null mean
rho ≈ 0.372 ± small sd for every direction vs real 0.674–0.839 — the
canonical correlations are far above the finite-sample baseline (all
p = 0.002 additive), while confirming that raw rho values carry
inflation. Null results did not alter any model fit or selection.

## 15. Interpretation

The experiment answers the opening question **negatively on this
dataset/seed, with a mechanism-level positive on the ablation**:
direction-specific CCA shrinkage is implementable, numerically exact,
behaves exactly as declared (rho→1 gives up to 2x penalty, fig.
`canonical_rho_vs_alpha`), and its adaptive form beats the uniform
control — but the CCA-compressed representation itself loses more
predictive information (G: 29 dims) than the soft sharing recovers.
On Haptics seed 42, raw `[G||H]` with the canonical scalar-alpha Ridge
remains the strongest configuration; hard CCA unique is second; the
proposed Canonical Ridge is below both.

## 16. Limitations

- Single dataset, single seed, n=132: no variance estimate; deltas of
  1–2 pp are well within test-set noise (the bootstrap SD of M0 alone
  is ±0.028).
- GCV is a train-only criterion but selects for squared-error fit of
  one-hot targets, not Macro-F1; a classifier-calibrated train-only
  selector could differ.
- The rank-95% PCA of G (29 dims) is the binding constraint; a
  Canonical Ridge on the raw 9996-dim space is the natural follow-up.
- gamma = 1.0 fixed; no claim that it is a good constant, only that the
  mechanism was tested as declared.

## 17. Scope statement

This is a **SINGLE HAPTICS SEED-42 EXPERIMENT** of a **proposed**
CCA-adaptive generalized ridge.

## 18. Generalization statement

**No conclusion of general superiority is justified from one seed.**
The measured result on this dataset does not show an improvement over
the canonical baselines.

## Reproducibility

- Code: `models/canonical_ridge/model.py`,
  `experiments/heramba_canonical_ridge_haptics_seed42/runner.py`
- Tests: `tests/test_canonical_ridge.py` (10) + existing 13 = 23 passed
- Command: `python experiments/heramba_canonical_ridge_haptics_seed42/runner.py`
  (~2 s after the 500-permutation CCA null; banks cached)
- Artifacts: `results.json`, `final_comparison.csv`,
  `canonical_ridge_diagnostics.csv`, `canonical_correlations.csv`,
  `alpha_schedule.csv`, `config.json`, `predictions/` (5 npy),
  `figures/` (`canonical_rho_vs_alpha`, `cca_permutation_null`,
  `final_comparison`; PDF+PNG)
