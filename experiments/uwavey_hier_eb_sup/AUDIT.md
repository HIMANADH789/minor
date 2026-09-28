# AUDIT — HIER-EB-SUP (written before any implementation change)

All facts below were derived from the current code (not documentation), from:

- `experiments/uwavey_hier_continuous_sup/runner.py` — the incumbent unified-pool experiment (the direct comparator)
- `experiments/uwavey_capacity_scaled/runner.py` — `top_f_select`, `r5_cv_fixed_rho`, `hier_banks_test`, `fit_minirocket_nk`, `extractor_facts`
- `experiments/uwavey_hier_high_sup/runner.py` — `col_meta`
- `experiments/uwavey_nested_hierarchical/runner.py` — `load_data`, `znorm`, `load_context_model`, `_encoder_latents`, `build_latent_tree`/`assign_levels` (via `models/nested_regimes/model.py`), `compute_activations`, `delta_banks`, `ppv_all`, `ridge_eval`

## 1. Expanded MiniROCKET (locked)

- `fit_minirocket_nk(Xtr_z, B_HIGH)` → aeon `MiniRocket(n_kernels=19_992, random_state=42)`, fit on TRAIN z-normed signals only.
- `extractor_facts` confirms: 84 physical kernels × 238 quantiles = **19,992** features (deterministic; asserted against `results/uwavey_capacity_scaled/seed42/diagnostics/dimensions.json` → `extractors.expanded`).

## 2. Root matrix / level 0 (locked)

- `GHK_trva = ppv_all(exHK, vstack([Xtr_z, Xva_z]))` → shape `(896, 19,992)` — the **full** expanded root matrix; the 17,993 cut in HIER-HIGH-SUP happens only at the final `hstack`, so no bypass is needed. `GHK_te = ppv_all(exHK, Xte_z)` → `(3,582, 19,992)`.

## 3. Hierarchy (locked)

- Frozen seed-42 SSL encoder (`load_context_model`) → TRAIN latents (761×315×32 = 239,715 rows) → `build_latent_tree(lat.reshape(-1,32), seed=42)` → recursive 2-means; levels `[1,2,4,8,16]`; `nesting_errors` == none.
- `reg16_tr` asserted **byte-identical** to `results/uwavey_nested_hierarchical/seed42/hierarchy_assignments.npy`.
- Val/test: `_encoder_latents` + `assign_levels(..., C)` — transform-only.

## 4. Delta bank + candidate ordering (locked)

- Dev pool: signals-chunked `compute_activations` (chunk 32) → `delta_banks(act[:, 4998:], valid[4998:], reg16, KEPT_K=[2,4,8,16], delta_cols=None, sample_chunk=96)` → `(896, 449,820)`.
- Layout (verified in `col_meta`): blocks coarse→fine `K=2 | K=4 | K=8 | K=16`; within a block child-major, kernel-minor. 30 edges × 14,994 kernels = **449,820**.
- Test pool: `hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)` with `sel_dict: K -> (n_sel, 2)` global `(kernel, child)` ids in the SAME row order as the selected columns.
- Unified pool: `C_dev = np.hstack([GHK_trva, Hcand_trva])` → `(896, 469,812)`; candidate `c < 19,992` is a root (level 0), else `col_meta(c - 19,992, 14,994)`.
- Level boundaries in the unified pool: root `[0, 19,992)`, K2 `[19,992, 49,980)`, K4 `[49,980, 109,956)`, K8 `[109,956, 229,908)`, K16 `[229,908, 469,812)`.

## 5. ANOVA-F statistic and p-values (incumbent internals)

- `top_f_select(H, y, n)`: `sklearn f_classif` → `F, p = f_classif(...)`; `F = nan_to_num(F, nan=0)`; `argsort(-F, kind="stable")[:n]`.
- **Key audit fact**: `f_classif` already returns the exact p-values with the same df (dfn = k−1, dfd = N−k) and class handling as the F statistics. HIER-EB-SUP therefore takes BOTH F and p from the identical `f_classif` call — the statistic is bit-identical to the incumbent selector's; only the ranking score changes (F → posterior signal probability q from the EB two-groups model).
- Edge cases (documented, no candidate discarded): non-finite F or p (constant features etc.) → F=0, p=1; p clipped to `[1e-300, 1−1e-16]` before `scipy.stats.norm.isf`; z finite by construction, extra `nan_to_num` belt.

## 6. Ridge evaluation (locked)

- `ridge_eval(Z_dev, y_dev, Z_te, yte)` → `RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))` fit on dev, one official test evaluation; returns macro-F1, accuracy, selected alpha, predictions.

## 7. Supervised boundary (locked; replicated exactly)

- CV diagnostic: 5-fold `StratifiedKFold(shuffle=True, random_state=42)` over train+val (896 rows); per fold, selection uses **fold-train rows/labels only**; held-out fold transformed with the selected identities; Ridge on fold-train; macro-F1 on the fold. Diagnostic only.
- Final production selection: on the full dev matrix (train+val, 896 rows) — exactly the canonical R5 final-stage boundary shared by R5-HIGH / HIER-HIGH-SUP / HIER-CONTINUOUS-SUP. Test labels touched only inside `ridge_eval` for scoring.

## 8. What changes for HIER-EB-SUP (the ONE change)

HIER-CONTINUOUS-SUP section 9 (`f_classif` → `nan_to_num` → `argsort(-F)[:B]`) is replaced by:

```
F, p = f_classif(C_dev, y_dev)            # SAME statistic (bit-identical)
z   = norm.isf(clip(p))                   # one-sided significance coordinate
q   = 1 - lfdr  per level (5 independent parametric two-groups EB fits)
sel = lexsort((cand_index, -q))[:19,992]  # ONE global ranking, ties by index
```

Everything else — pool construction, layout, metadata, shared-column-order assembly, `hier_banks_test` gather, `ridge_eval`, gates, artifact conventions — is imported verbatim from the verified helpers. No rho, no N_G/N_H, no per-level quota, no q multiplication of features, no moderated-F.

## 9. Artifact-safety constraints (for tests)

- New outputs only under `results/uwavey_hier_eb_sup/seed42/`; every `np.save`/`savefig` path must reference `OUT` inline (AST-enforced by the incumbent test pattern).
- Stored references used read-only: `results/uwavey_nested_hierarchical/seed42/` (split indices, hierarchy), `results/uwavey_capacity_scaled/seed42/diagnostics/dimensions.json`, stored reference scores {canonical_mr 0.7543, mr_high 0.7477, r5_high 0.7772, hier_high 0.7230, hier_high_sup 0.7823, hier_continuous_sup 0.7599}.
- MR-HIGH reproduction gate re-run in-process from the full root matrix (reference 0.7477, tol 0.002).
