# AUDIT — HIER-CONTINUOUS-SUP (code-derived, no documentation inferred)

Every fact below was read from the current code / verified against stored
artifacts on 2026-09-20. Files inspected:
`experiments/uwavey_hier_high_sup/runner.py`,
`experiments/uwavey_capacity_scaled/runner.py`,
`experiments/uwavey_nested_hierarchical/runner.py`,
`tests/test_uwavey_hier_high_sup.py`.

## 1. Expanded MiniROCKET (locked)

`experiments/uwavey_capacity_scaled/runner.py::fit_minirocket_nk(Xtr_z, n)`:
`MiniRocket(random_state=42, n_jobs=-1, n_kernels=n).fit(Xtr_z[:, None, :].astype(np.float32))`
— **train-only** fit. With `n_kernels=19,992` aeon allocates 84 physical
kernels × 238 quantiles → `total_features = 19,992`
(`extractor_facts()`: `physical_kernels=84, quantiles_per_kernel=238,
total_features=19992`). Verified equal to the stored
`results/uwavey_capacity_scaled/seed42/diagnostics/dimensions.json`
extractor facts in the HIER-HIGH-SUP run (assertion passed).

## 2. Root-level candidate matrix — FULL width is already materialized

In `experiments/uwavey_hier_high_sup/runner.py` (section "4. G banks"):

    GHK_trva = ppv_all(exHK, np.vstack([Xtr_z, Xva_z]))   # (896, 19,992)
    GHK_te   = ppv_all(exHK, Xte_z)                       # (3582, 19,992)

`ppv_all` returns the **FULL expanded width** — the runner asserts
`GHK_trva.shape == (896, 19,992)`. The protected `N_G = 17,993` cut happens
ONLY at the final assembly (`GHK_trva[:, :N_G]`). Therefore the full
19,992-root candidate matrix required by HIER-CONTINUOUS-SUP **already
exists in the pipeline and needs no bypass**; the new runner simply does not
apply the `[:, :N_G]` slice.

## 3. Hierarchical delta candidate pool (locked)

`experiments/uwavey_nested_hierarchical/runner.py::delta_banks(act_H,
valid_het, reg16, kept_K, delta_cols=None)` → with `delta_cols=None`
returns the FULL pool: column order level coarse→fine (K=2,4,8,16),
children 0..K−1 within level, kernels within a child (child-major,
kernel-minor). Column `c` of the pool ↔ `(K, child j, parent j>>1, global
kernel G_PART + k)` via `uwavey_hier_high_sup.runner.col_meta(c, F_H)` with
`G_PART=4998`, `F_H = 19,992 − 4,998 = 14,994`.

Pool size: `sum(KEPT_K) * F_H = 30 × 14,994 = 449,820` — asserted in the
HIER-HIGH-SUP run (`Hcand_trva.shape == (896, 449,820)`).

Telescoping/occupancy definition: produced by
`models/nested_regimes/model.py::hierarchy_chain(want_deltas=True)`;
parent PPV = occupancy-weighted mean of children; zero-sum within each
parent (Σ_c (n_c/n_p) Δ_{m,c} = 0). Unchanged.

Dev-side construction is signals-chunked (32-sample chunks; activations
never materialized at full split width). Test-side selected columns come
from `capacity_scaled.runner.hier_banks_test(extractor, Xte_z, reg16_te,
KEPT_K, sel_dict)` with `sel_dict: K -> (n_sel, 2)` arrays of
`(kernel_local, child)` in the SAME per-level row order as the selected dev
columns (`delta_banks(..., delta_cols=sel)` returns columns in `sel` order).

## 4. Supervised selector (reused object, not reimplemented)

`experiments/uwavey_capacity_scaled/runner.py::top_f_select(H, y, n)`:
sklearn `f_classif` → `np.nan_to_num(nan=0.0)` →
`np.argsort(-f_stat, kind="stable")` → top-N. Deterministic; NaN/constant
features get F=0 and sort last at equal keys.

## 5. Supervised selection protocol (canonical R5, replicated)

- **Final stage** (the canonical R5-HIGH protocol): selection on the FULL
  dev matrix with **train+val labels** (`y_dev`, 896 rows) — exactly what
  R5-HIGH does (`capacity_scaled/runner.py` final stage), documented in
  `uwavey_hier_high_sup/diagnostics/selection_protocol.md`. HIER-HIGH-SUP
  used `top_f_select(Hcand_trva, y_dev, 1,999)`.
- **CV diagnostic**: `r5_cv_fixed_rho` — 5-fold
  `StratifiedKFold(shuffle=True, random_state=42)`; for EVERY fold:
  fold-train rows/labels only → `top_f_select` (fold-internal selection) →
  held-out rows transformed with those identities →
  `RidgeClassifierCV(ALPHAS)` fit on fold-train → macro-F1 on fold-val.
  Diagnostic only, never used for selection.
- **Classifier**: `uwavey_nested_hierarchical.runner.ridge_eval` —
  `RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))` fit on train+val, ONE
  official test evaluation. Returns macro_f1, accuracy, selected_alpha.

## 6. Hierarchy (locked)

Rebuilt deterministically from TRAIN latents only:
`load_context_model` (frozen seed-42 SSLTemporalEncoder, d=32) →
`_encoder_latents(model, znorm(Xtr))` →
`models.nested_regimes.model.build_latent_tree(latents, seed=42)`
(recursive 2-means, n_init=10, random_state=42 per node) →
`assign_levels` for val/test (transform-only). HIER-HIGH-SUP asserted
`reg16_tr == stored results/uwavey_nested_hierarchical/seed42/
hierarchy_assignments.npy` byte-identically. 239,715 train latents
(761 × 315); all 16 leaf regimes occupied.

## 7. Data / split / seed (locked)

`load_data()` (official UCR UWaveGestureLibraryY loader): train 761,
validation 135, test 3,582, T=315, 8 classes; per-sample z-normalization
via `znorm`. Split identity asserted against the stored
`uwavey_nested_hierarchical/seed42/{train,val,test}_indices.npy` sizes and
(loaded in tests) exact values. `SEED = 42` everywhere; KMeans seeds fixed;
StratifiedKFold `random_state=42`.

## 8. Unified candidate count (derived, then asserted in code)

- root candidates: full expanded PPV width = `fHK["total_features"]` =
  238 × 84 = **19,992** (NOT the preselected 17,993 — §2).
- delta candidates: **449,820** (§3).
- unified: **19,992 + 449,820 = 469,812**. Asserted as
  `C_dev.shape == (896, 469,812)` before selection.

## 9. Where selection/allocation occurs in the incumbent design

`uwavey_hier_high_sup/runner.py`: (a) section 7 CV diagnostic
(fold-internal, H pool only, diagnostic); (b) section 8 final
`top_f_select(Hcand_trva, y_dev, N_H)`; (c) section 11 final assembly
`hstack([GHK_trva[:, :N_G], Hcand_trva[:, top]])` — **this `[:, :N_G]`
slice is the protected G/H boundary**, the single thing HIER-CONTINUOUS-SUP
removes. No other allocation exists (the energy allocator
`level_budgets/select_carriers` is absent from the SUP runner; it lives
only in the HIER-HIGH/capacity-scaled runner and is NOT reused here).

## 10. Label-flow boundaries (unchanged)

- MiniROCKET fit: train signals only.
- Hierarchy: train latents only; val/test transform-only.
- Delta pool / root matrix: frozen transform, label-free.
- Final selection: train+val labels (canonical R5 protocol — §5).
- CV diagnostic: fold-train labels only.
- Test labels: touched only inside `ridge_eval` for scoring.

## 11. Feasibility / fairness statement

All three components of the unified pool are produced by the same frozen
train-fitted transform on the same rows, so unified competition is
well-posed: `C = [G_full ‖ Delta_full] ∈ R^(896 × 469,812)` ≈ 3.4 GB
float64 — materializable (the 449,820-wide pool alone was already
materialized by HIER-HIGH-SUP for `f_classif`). Final selection
`top_f_select(C, y_dev, 19,992)`; no rho, no N_G/N_H, no per-level quota
anywhere in the new runner. Nothing in the existing implementation blocks
the equal-budget comparison; **no deviation from the incumbent protocol is
required.**
