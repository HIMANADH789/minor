# Selection protocol (audited from the existing R5-HIGH code)

Selector: `experiments/uwavey_capacity_scaled/runner.py::top_f_select` --
sklearn `f_classif` -> `np.nan_to_num(nan=0.0)` -> `np.argsort(-f_stat,
kind="stable")` -> top-N.  Deterministic; identical implementation, tie
handling, NaN/constant-feature handling and top-N semantics as R5-HIGH.

## CV diagnostic (reported, never used for selection)

`r5_cv_fixed_rho(G_full, H_full, y_dev, n_g=17,993, n_h=1,999)`:
5-fold `StratifiedKFold(shuffle=True, random_state=42)`; for EVERY fold:
1. fold-train rows/labels only -> `f_classif` -> top 1,999 hierarchical
   candidates (fold-internal selection),
2. held-out fold rows transformed with THOSE selected feature identities,
3. `RidgeClassifierCV(ALPHAS)` fit on the fold-train, macro-F1 on fold-val.

## Final stage (the canonical R5-HIGH protocol, replicated verbatim)

`top = top_f_select(H_dev, y_dev, 1,999)` on the FULL dev matrix
(train+val = 896 rows) -- i.e. the canonical protocol DOES use train+val
labels for the final selection, exactly as the existing R5-HIGH did
(cap-runner L388-396).  Then `ridge_eval([G_dev[:, :17993] | H_dev[:, top]],
y_dev, [G_te[:, :17993] | H_te[:, top]], yte)`: RidgeClassifierCV
(alphas=np.logspace(-4,4,20)) fit on train+val, ONE official test evaluation.

## Data/label flow

- hierarchy: TRAIN latents only; val/test transform-only.
- candidate pool: frozen transform, label-free.
- final selection: train+val labels (canonical R5 protocol).
- CV diagnostic: fold-train labels only.
- test labels: touched only inside `ridge_eval` for scoring.

## Pool notes (locked definition)

- 30 edges x 14,994 H-side kernels = 449,820 candidates; column order
  K=2,4,8,16 coarse->fine, children 0..K-1 within level, kernels within.
- At K=2 the two child deltas are exact anti-correlated rescalings (zero-sum
  with two children); their F-stats are equal up to float rounding, so both
  columns can enter the top-1,999.  The previous ENERGY allocator deduped
  K=2 child-1; that dedup belongs to the ablated allocator and is NOT used.
- No level quotas, no energy, no null, no largest-remainder budgeting:
  the selector ranks the entire pool and the observed per-level counts are
  diagnostic only.
