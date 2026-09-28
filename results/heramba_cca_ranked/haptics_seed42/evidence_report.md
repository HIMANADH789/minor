# RankCCA-HERAMBA — Haptics, Seed 42 — Evidence Report

**Experiment namespace:** `results/heramba_cca_ranked/haptics_seed42/`
(the previous full-rank experiment in `results/heramba_cca/haptics_seed42/`
is preserved byte-identical as historical evidence; enforced by unit test).

---

# 1. Objective

Determine whether the HERAMBA/R2 representation H contains predictive
information **not already linearly represented** in canonical MiniROCKET G,
using a rank-controlled CCA shared/unique decomposition instead of the
degenerate full-rank CCA of the previous experiment. Final model:
`[G || H_unique]` — no mixing ratio rho anywhere.

# 2. Previous Full-Rank CCA Degeneracy

With n_train = 132 and numerical-rank reduction (131/132 components kept),
all 131 canonical correlations were 1.0: both banks span the entire sample
space, every direction of H is "shared", and H_unique was empty
(effective rank 1.0, unique variance fraction 0.0). CCA-HERAMBA therefore
degenerated to G-only (0.5037). That experiment stands as a demonstration
of the small-n/high-d degeneracy, not as evidence about H's information.

# 3. Rank-Controlled CCA Specification

- `G_train -> train z-score -> PCA(k_G) -> G_low`
- `H_train -> train z-score -> PCA(k_H) -> H_low`
- CCA(G_low, H_low) via whitened cross-covariance SVD (train only).
- shared := canonical components with rho >= 0.5 (predeclared tau).
- `H_unique = H_low - OLS(H_low ~ shared canonical scores)` (train-fit,
  frozen, applied unchanged to val/test).
- Final representation `[G (raw 4998) || H_unique (k_H)]`, canonical
  RidgeClassifierCV fit on train+val, ONE test evaluation.

# 4. Rank Selection Rule

```
rank_rule = "minimum_train_PCA_components_for_95pct_variance"
```

Fixed methodological constant (variance threshold 0.95), declared before
any predictive evaluation; never tuned on validation or test.

Result (of 132 maximum):

| bank | retained rank k | train variance kept |
|---|---|---|
| G (MiniROCKET) | **29** | 0.9514 |
| H (HERAMBA) | **88** | 0.9516 |

The asymmetry is itself informative: H's energy is spread over ~3x more
directions than G's (consistent with the earlier effective-rank finding:
G 68 vs H 105 at full rank).

# 5. CCA Spectrum

With rank control the degeneracy is broken:

- canonical dimensions = 29 (limited by k_G)
- rho max = **0.839**, rho mean = 0.780, rho min = 0.674
- all 29 components have rho >= 0.5 -> shared = 29 by the tau rule
- **No component reaches 1.0** — sanity requirement satisfied (the run
  would have STOPPED otherwise).

Interpretation: within the 95%-variance subspaces, every dominant
direction of G is strongly (rho 0.67-0.84) but not perfectly aligned with
H. The alignment is high — H shares most of its dominant structure with
G — but there is genuine residual geometry.

# 6. Shared/Unique Decomposition

- shared components = 29/29 (tau = 0.5, predeclared)
- variance partition of H_low (train): **shared 0.6253, unique 0.3747**
- H_unique dimension = 88 (of which 59 have nonzero variance on train)
- orthogonality on train: max |corr(H_unique, shared)| = **5.3e-15**
  (exact OLS construction; numerical tolerance ~1e-14), mean 2.7e-16,
  residual covariance norm reported in `cca_ranked_summary.json`.

# 7. Effective-Rank Analysis

| representation | numerical rank | effective rank |
|---|---|---|
| G_low | 29 | 19.05 |
| H_low | 88 | 71.16 |
| **H_unique** | 59 | **54.06** |

After removing the shared subspace, H retains a substantial,
non-degenerate component: 37.5% of H's (95%-energy) variance and
effective rank 54. HERAMBA is therefore **not** redundant with
MiniROCKET at the representation level once rank control is applied.

# 8. Label-Complementarity Test

Terminology note: this is a test of **incremental label-relevant
information** / conditional predictive complementarity under a specific
statistical model — not a mutual-information estimate and not a partial
information decomposition.

- Base `Y ~ G` vs augmented `Y ~ [G, H_unique]`, identical paired
  5-fold stratified CV on the dev set (train+val, n=155), canonical
  RidgeClassifierCV, no hyperparameter search.
- Observed CV increment: **+0.0271 Macro-F1** (0.4575 -> 0.4845),
  Cohen's d = 0.49 over folds.
- Null: H_unique rows permuted (destroys association with Y and G,
  preserves marginals); S = 1000 permutations (predeclared), seed 42042,
  additive p = (1 + #{null >= obs}) / (S+1).
- Null mean increment = -0.0371; **empirical p = 0.004**.

H_unique carries label-relevant information beyond G at the dev-CV level.

# 9. Final Classification Results

Single official test evaluation per model, canonical final fit on
train+val:

| Representation | Features | Macro-F1 | Accuracy | alpha |
|---|---|---|---|---|
| MiniROCKET (G only) | 4998 | 0.5037 | 0.5162 | 4.281 |
| R2/R5/HERAMBA `[G||H]` | 9996 | **0.5500** | 0.5649 | 4.281 |
| RankCCA-HERAMBA `[G||H_unique]` | 5086 | 0.5138 | 0.5315 | 1.624 |

# 10. Comparison With MiniROCKET and R2/R5

- RankCCA-HERAMBA beats MiniROCKET: +0.0101 Macro-F1 (0.5138 vs 0.5037).
- RankCCA-HERAMBA stays below full `[G||H]`: -0.0362 (0.5138 vs 0.5500).
- Per-class: RankCCA improves class 0 (F1 0.413 vs 0.370) and class 4
  (0.636 vs 0.589) over R2, but loses on classes 1-3; the full-H model
  remains the best overall.

# 11. Scientific Interpretation

Outcome classification: closest to **(B/C hybrid)** — H_unique is
substantial (effective rank 54, 37% of H variance) and shows significant
conditional predictive complementarity on the dev CV (+0.0271, p=0.004),
but the final test classifier exploits only part of it: RankCCA-HERAMBA
improves over MiniROCKET while remaining below raw concatenation.

The mechanism is consistent with the earlier findings: H's predictive
value on Haptics is real (R2 > M0 by +0.046 pp on test here) but not
confined to its CCA-shared dominant directions; rank-controlling the
decomposition removes 37% of H's variance as "shared" yet keeps most of
the predictive gain in the removed direction — i.e., part of what H adds
overlaps with dominant G structure when both are compressed to their
95% energy subspaces. The raw concatenation [G||H] remains the strongest
configuration on this dataset; the decomposition's value here is
diagnostic (it quantifies and localizes the complementarity), not a
performance improvement.

Seed uncertainty: NOT established — single seed 42, single split. The
dev-CV permutation p is an association statement under the chosen model,
not a guarantee of test-set behavior across seeds.

# 12. Leakage Audit

- PCA/scaler: fitted on TRAIN (132 rows) only; frozen before val/test.
- CCA: fitted on G_low/H_low from TRAIN only; label-free.
- Shared-subspace rule and tau: predeclared (0.5), never adjusted.
- Rank rule: predeclared 95% variance, TRAIN-only, never adjusted.
- H_unique residualization: fitted on TRAIN; identical frozen transform
  applied to val and test (unit-tested).
- Permutation test: dev-set CV only; test untouched until final evals.
- Test: one evaluation per model, after all freezing.
- Labels: never enter any decomposition fit (label-permutation
  invariance unit test passes).
- Previous full-rank artifacts: byte-unchanged (unit test asserts the
  stored degenerate values 131 shared / rho=1.0 / 0.5037 / 0.5500).
- MiniROCKET: canonical aeon implementation untouched (verified exactly
  against all four benchmark references earlier).
- R2/R5: untouched; identity gate `[G||H] -> 0.5500` passes in-run.

# 13. Reproducibility

- Model code: `models/heramba_cca/` (cca.py, complementarity.py,
  diagnostics.py, model.py) — unchanged except the added
  `plus_one_permutation_test` helper.
- Experiment: `experiments/heramba_cca_ranked_haptics_seed42/runner.py`.
- Tests: `tests/test_heramba_cca_ranked.py` (6 passed) +
  `tests/test_heramba_cca.py` (7 passed).
- Command:
  `python experiments/heramba_cca_ranked_haptics_seed42/runner.py`
  (~155 s; banks are cached frozen artifacts).
- Exact rules:
  - rank: `minimum_train_PCA_components_for_95pct_variance`
    (threshold 0.95, z-score with train statistics)
  - shared: canonical components with rho >= 0.5 (tau = 0.5)
  - permutation: S = 1000, row-permutation of H_unique, seed 42042,
    p = (1 + count(null >= observed)) / (S + 1)
- Artifacts in this directory: `cca_ranked_summary.json`,
  `canonical_correlations.csv`, `complementarity.json`,
  `effective_rank.csv`, `final_comparison.csv`, `results.json`,
  `predictions_m0.npy`, `predictions_rankcca.npy`, `figures/`
  (4 figures, PDF+PNG: `cca_ranked_spectrum`, `h_unique_spectrum`,
  `variance_partition`, `effective_rank_comparison`).
