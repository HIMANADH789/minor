# AUDIT - HIER-SUP-MOD (code-derived; no dimensions assumed)

Audit performed before any implementation change. Every fact below was read
from the current repository code and stored result artifacts.

## 1. Dataset / split (LOCKED)
- Loader: `experiments.uwavey_nested_hierarchical.runner.load_data` -
  canonical UWaveGestureLibraryY, official UCR split 761/135/3,582, T=315,
  8 classes, univariate, per-sample z-normalization (`znorm`).
- Split identity asserted in-run against the stored
  `results/uwavey_nested_hierarchical/seed42/{train,val,test}_indices.npy`
  (761/135/3,582) - the same artifact used by every prior UWaveY experiment.

## 2. Expanded MiniROCKET (LOCKED)
- `fit_minirocket_nk(Xtr_z, 19992)` - aeon MiniRocket(n_kernels=19992,
  random_state=42), fit on TRAIN z-normed signals only.
- Stored extractor facts
  (`results/uwavey_capacity_scaled/seed42/diagnostics/dimensions.json`):
  **84 physical kernels x 238 quantiles = 19,992 root candidates.**
- Full root matrix via `ppv_all` -> (N, 19,992); NO truncation before
  unified selection.

## 3. Frozen hierarchy (LOCKED)
- `load_context_model` -> frozen seed-42 SSLTemporalEncoder;
  `_encoder_latents` on TRAIN z-normed signals ONLY;
  `build_latent_tree(lat.reshape(-1, 32), seed=42)` (recursive 2-means,
  n_clusters=2, n_init=10, random_state=42); `assign_levels` for val/test
  (transform-only); `nesting_errors == []` asserted; level-16 assignments
  byte-identical to the stored `hierarchy_assignments.npy`.

## 4. Telescoping candidate pool (LOCKED)
- PPV^(0) = 0 -> D1 = PPV^(1) (root increments, 19,992; bit-equal
  equivalence asserted as in HIER-SUP-CLEAN).
- Deeper residuals: `delta_banks(act[:, G_PART:], valid[G_PART:],
  reg16, KEPT_K, delta_cols=None)` - levels coarse->fine, child-major,
  kernel-minor; 30 edges x 14,994 kernels = **449,820**.
- Occupancy-weighted parent rule: `pp_par = (pc0*ppv_c0 + pc1*ppv_c1)/
  (pc0+pc1)`; per-parent zero-sum of occupancy-weighted residuals holds to
  ~1e-16 (chain `zero_sum_max`).
- Unified pool `C = [D1 | D2 | D4 | D8 | D16]`, **469,812** columns, one
  shared order for train/val/test (hard layout guard, per spec 27).

## 5. Raw ANOVA-F baseline (LOCKED reference)
- Incumbent selector: `top_f_select` = sklearn `f_classif` +
  `nan_to_num(0)` + `np.argsort(-f, kind="stable")` top-N, on train+val
  (canonical R5 final-stage protocol).
- Stored HIER-SUP-CLEAN run (`results/uwavey_hier_sup_clean/seed42`):
  Macro-F1 **0.7599**, alpha 78.476, composition D1 14,453 / K2 4,012 /
  K4 1,436 / K8 65 / K16 26; `selected_feature_metadata.csv` carries the
  per-candidate raw F for all 19,992 selected (source_type root_increment,
  level 1; hierarchy_delta, 2/4/8/16). Equivalence gate target.

## 6. ANOVA design (spec sections 8-9)
- C = 8 classes; d_between = C - 1 = 7; d_error = N - 8 (N = 896 dev rows
  in production -> d_error = 888). IDENTICAL for every candidate:
  occupancy is NOT used as degrees of freedom, NOT a ranking weight, NOT a
  quota. The hierarchy changes candidate VALUES (and hence their residual
  variances), never the ANOVA design.
- `f_classif` computes the same one-way model: SS_between via class means,
  SSE = SS_total - SS_between, MS_b = SS_b/7, F = MS_b/(SSE/d_error) -
  verified in the audit session against manual formulas on synthetic data.

## 7. EB variance moderation (spec sections 10-16) - the ONE change
- Per candidate: s_j^2 = SSE_j / d_error (candidate-specific residual
  variance). F_j = MS_between,j / s_j^2 (== f_classif's F).
- Prior: s_j^2 ~ s0^2 * F(d_e, d0) fitted ONCE over the full unified pool
  (borrowing strength across ALL levels - deliberate, per spec 11) by
  Smyth/limma log-moment matching:
  Var[log s^2] = trigamma(d_e/2) + trigamma(d0/2) -> solve d0 (Brent,
  bracketed); log s0^2 = mean(log s^2) - digamma(d_e/2) + digamma(d0/2)
  - log(d0/d_e).
- Posterior: s_t^2 = (d0*s0^2 + d_e*s^2) / (d0 + d_e); moderated F =
  MS_between / s_t^2; effective df = d_e + d0 (diagnostic p on
  F(7, d_e+d0)).
- Numerical edge cases (spec 13): non-finite/non-positive s^2 candidates
  are counted and reported; the log-moment fit uses a documented safe
  floor (smallest positive finite s^2) ONLY inside the log domain; ALL
  candidates stay in the moderation and ranking; degenerate fits are
  reported with condition + reason + fallback rule. No hand-picked
  hyperparameters; no validation tuning; no grid search.

## 8. Supervised boundary (LOCKED)
- CV diagnostic: 5-fold StratifiedKFold(shuffle=True, random_state=42);
  per fold the ANOVA statistics, EB prior (d0, s0^2), moderated ranking
  and top-B selection use FOLD-TRAIN rows/labels ONLY; fold-val is scored
  under the fold-frozen selection. Diagnostic only.
- Final production: statistics + prior + selection on train+val (896
  rows); test (3,582) transformed only; test labels touched only inside
  `ridge_eval` for scoring. d0/s0^2/selection frozen before test
  transform (spec 26).

## 9. Classifier (LOCKED)
- `ridge_eval` - RidgeClassifierCV(alphas=logspace(-4,4,20), 5-fold CV)
  fit on the selected 19,992-column representation over train+val, ONE
  official test evaluation. Identical to HIER-SUP-CLEAN.

## 10. Diagnostics implemented (spec 18-22, 35, 37, 38)
- Per-level median s^2 / s_t^2 / F / F_tilde and selection counts
  (diagnostic only; never fed back).
- Raw top-B vs moderated top-B overlap (intersection, raw-only,
  moderated-only, Jaccard; in/out counts).
- Fallback diagnostic: `select_and_mask` (min occupancy 1% of T) keeps
  regimes with count >= 4 timesteps, else falls back to the argmax-count
  regime; per-level per-sample fallback fractions reported. Fallback
  status is DIAGNOSTIC ONLY - no candidate is filtered.
- Occupancy-vs-variance/F Spearman correlations (min branch occupancy vs
  log s^2, F, log s_t^2 per level + full delta pool); occupancy-vs-class
  one-way ANOVA with BH adjustment per level. Neither feeds selection.

## 11. Equivalence gate (spec 28)
- Reconstructing the RAW ANOVA-F ranking from the same candidate matrix
  MUST reproduce HIER-SUP-CLEAN: selected index set + sequence, per-
  candidate raw F (vs the stored metadata), composition, predictions,
  alpha, Macro-F1 (0.7599). Production asserts PASS before accepting any
  HIER-SUP-MOD number; smoke skips (subset).

## 12. Verdict
**No blocker.** The moderation changes only the ranking statistic; every
other component is imported from the verified prior runners. Expected
candidate counts asserted: 19,992 / 449,820 / 469,812 / 19,992.
