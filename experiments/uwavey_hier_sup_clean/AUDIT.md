# AUDIT — HIER-SUP-CLEAN (code-derived; no dimensions assumed)

Audit performed before any implementation change. Every fact below was read
from the current repository code and stored result artifacts.

## 1. Dataset / split (LOCKED)
- Loader: `experiments.uwavey_nested_hierarchical.runner.load_data` —
  canonical UWaveGestureLibraryY, official UCR split 761/135/3,582, T=315,
  8 classes, univariate, per-sample z-normalization (`znorm`).
- Split identity is asserted in-run against the stored indices
  `results/uwavey_nested_hierarchical/seed42/{train,val,test}_indices.npy`
  (761/135/3,582) — identical artifact used by every prior UWaveY experiment.

## 2. Expanded MiniROCKET (LOCKED)
- `fit_minirocket_nk(Xtr_z, 19992)` — aeon `MiniRocket(n_kernels=19992,
  random_state=42)`, fit on TRAIN z-normed signals only.
- Verified stored extractor facts
  (`results/uwavey_capacity_scaled/seed42/diagnostics/dimensions.json`):
  **84 physical kernels x 238 quantiles = 19,992 root features.**
- Root matrix: `ppv_all(exHK, X)` -> `(N, 19,992)`, FULL width. The
  unified-pool path contains **no** 17,993 preselection anywhere (the
  incumbent continuous runner asserts `GHK_trva.shape == (N, ROOTS)` with
  `ROOTS = B_HIGH = 19992`).

## 3. Frozen hierarchy (LOCKED)
- `load_context_model` -> frozen seed-42 SSLTemporalEncoder;
  `_encoder_latents` computed on TRAIN z-normed signals ONLY;
  `build_latent_tree(lat.reshape(-1, 32), seed=42)` (recursive 2-means,
  n_clusters=2, n_init=10, random_state=42); `assign_levels` for val/test
  (transform-only).
- `nesting_errors(levels) == []` asserted; centroid structure [1,2,4,8,16].
- Level-16 assignments byte-identical to the stored
  `results/uwavey_nested_hierarchical/seed42/hierarchy_assignments.npy`.

## 4. Delta bank / telescoping residuals (LOCKED)
- `delta_banks(act[:, G_PART:], valid[G_PART:], reg16, KEPT_K,
  delta_cols=None)` -> chain layout: levels coarse->fine (K=2,4,8,16),
  child-major within a level, kernel-minor; 30 edges x 14,994 kernels =
  **449,820 candidates**.
- Parent PPV = occupancy-weighted child mean (`chain_from_counts_sums`:
  `pp_par = (pc0*ppv_c0 + pc1*ppv_c1)/(pc0+pc1)`); consequently the
  occupancy-weighted child residuals sum to zero per parent (chain
  `zero_sum_max` ~1e-16); zero-occupancy children get exactly 0 detail.
- `col_meta(c, F_H)` (imported from the hier_high_sup runner) maps a delta
  pool column -> (level K, child j, parent j>>1, global kernel id G_PART+k).

## 5. Selector / classifier (LOCKED)
- `top_f_select` = sklearn `f_classif` + `nan_to_num(0)` +
  `np.argsort(-f, kind="stable")` top-N — the SAME shared selector used by
  R5-HIGH, HIER-HIGH-SUP and HIER-CONTINUOUS-SUP (verified identical import).
- Final selection labels: train+validation (canonical R5 final-stage
  protocol). CV diagnostic: `r5_cv_fixed_rho(zeros(N,0), C, y, 0, B)` —
  n_g=0 exposes the full pool to fold-internal selection; diagnostic only.
- `ridge_eval` = RidgeClassifierCV(alphas=logspace(-4,4,20), 5-fold CV) fit
  on train+val, ONE official test evaluation.

## 6. Stored reference (HIER-CONTINUOUS-SUP, verified artifacts)
- `results/uwavey_hier_continuous_sup/seed42/results/hier_continuous_sup.json`:
  Macro-F1 **0.7599**, alpha 78.476, composition root 14,453 / K2 4,012 /
  K4 1,436 / K8 65 / K16 26, unified pool 469,812.
- `diagnostics/selected_feature_metadata.csv`: 19,992 rows, columns
  (selected_rank, candidate_index, source_type, level, parent, child,
  kernel_id, f_stat); candidate_index range 0..347,552 (< 469,812).
- `predictions/test_predictions.npy`: (3582,) int64.

## 7. Equivalence analysis (the scientific point of this experiment)
- With PPV^(0) = 0: Delta^(1) = PPV^(1) - 0 = PPV^(1) EXACTLY (same float64
  values — asserted in-run). The deeper residuals are the identical
  telescoping objects. The canonical column order
  [D1 | D2 | D4 | D8 | D16] equals the incumbent [roots | K2 | K4 | K8 |
  K16]. Selector, labels, classifier, seed, and pool are identical.
- Therefore HIER-SUP-CLEAN MUST reproduce the stored run: selected index
  set, per-candidate ANOVA-F, semantic metadata, composition, test
  predictions, and Macro-F1. The runner gates on this (verdict IDENTICAL,
  asserted in production); the smoke subset skips the gate.

## 8. Verdict
**No blocker.** The clean telescoping formulation is implementable without
changing any locked component; the expected outcome is an IDENTICAL
canonicalization of HIER-CONTINUOUS-SUP, and the run fails loudly if it is
not (implementation divergence, not a methodological difference).
