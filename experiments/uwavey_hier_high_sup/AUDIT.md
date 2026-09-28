# AUDIT — HIER-HIGH-SUP (code-derived, no documentation inference)

All facts below were read from the current code and stored artifacts on 2026-09-20.
Sources: `experiments/uwavey_capacity_scaled/runner.py` (cap-runner),
`experiments/uwavey_nested_hierarchical/runner.py` (nested-runner),
`models/nested_regimes/model.py`, `models/hierarchical_budget/model.py`,
`results/uwavey_capacity_scaled/seed42/`, `results/uwavey_nested_hierarchical/seed42/`.

## 1. Data / split / preprocessing (LOCKED)

- Loader: `load_data()` (nested-runner L86) → official UCR split for
  `UWaveGestureLibraryY`: **train 761 / validation 135 / test 3,582**, T=315,
  8 classes. Normalization: per-sample z-norm `znorm(X)` (nested-runner L102).
- Stored, verified split indices: `results/uwavey_nested_hierarchical/seed42/
  {train,val,test}_indices.npy` (regression target for index identity).
- The cap-runner asserted loader ≡ stored R2 indices (sorted, disjoint, sizes).

## 2. Expanded MiniROCKET (LOCKED)

- `fit_minirocket_nk(Xtr_z, 19992)` (cap-runner L104): aeon `MiniRocket(
  random_state=42, n_jobs=-1, n_kernels=19992)`, fit **train-only** on
  z-normed train signals. `n_kernels` in aeon sets the TOTAL feature target:
  84 physical kernels × 238 quantiles = **19,992** features (verified
  empirically; audit run measured `quantiles_per_kernel=238`).
- Stored extractor facts (`results/uwavey_capacity_scaled/seed42/diagnostics/
  dimensions.json` → `extractors.expanded`): physical_kernels 84,
  quantiles_per_kernel 238, n_unique_dilations 21, total_features 19992,
  n_biases 19992. Expansion factor exactly 2.0 vs canonical 84×119.
- No stored G/H arrays exist from the cap run (only scores + predictions), so
  bit-identity of the rebuilt bank is verified by (a) identical code path +
  seed + train-only fit, (b) `extractor_facts` equality, and (c) an in-run
  **MR-HIGH reproduction gate** (`ridge_eval(G_full, y_dev, G_te, yte)` must
  reproduce the stored 0.7477 within ±0.002). Documented as the regression
  assertion for `new_G_high == previous_G_high` at score level.

## 3. G_high convention (LOCKED, identical to R5-HIGH)

- `ppv_all(extractor, X_z)` (nested-runner L155) returns the FULL 19,992-wide
  PPV bank, chunked (chunk=64), float64.
- R5-HIGH uses `G_full[:, :N_G_R5_HIGH]` with `N_G_R5_HIGH = 17,993`
  (cap-runner L54–55: `N_H = round(0.1*19992) = 1,999`,
  `N_G = 19992-1999 = 17,993`). HIER-HIGH-SUP reuses exactly this.

## 4. Hierarchy (LOCKED, reused byte-identically)

- Frozen context model: `load_context_model(device)` (seed-42 SSL checkpoint,
  same file the R2/Haptics line uses). Latents `_encoder_latents(model, X_z,
  device)` → (N, 315, 32).
- Tree: `build_latent_tree(lat.reshape(-1, d), seed=42)` (models/
  nested_regimes/model.py L87) — recursive binary KMeans
  (`n_clusters=2, n_init=10, random_state=42` per node) over **train** latents
  only (239,715 vectors for full train), levels K=1→2→4→8→16.
  Val/test: `assign_levels` transform-only (cap-runner L320/L326).
- Reusable stored artifact: `results/uwavey_nested_hierarchical/seed42/
  hierarchy_assignments.npy` = **train level-16 assignments (761, 315) int64**.
  The runner rebuilds the tree deterministically and asserts
  `np.array_equal(reg16_tr, saved)` — the hierarchy-equality regression.
- Nesting invariants via `nesting_errors(levels)` (must be empty).

## 5. Hierarchical candidate pool (LOCKED definition)

- Activation matrix: `compute_activations(extractor, X_z)` →
  (N, 19,992, T) bool + validity; H-side slice `[:, G_PART:]` =
  kernels 4,998..19,991 → **F_H = 14,994 kernels**.
- Frozen-transform chain: `hierarchy_chain(act_H, valid_H, reg16,
  want_deltas=True)` (models/nested_regimes/model.py L253) computes
  occupancy-weighted PPVs and per-child deltas
  `Delta_{m,c} = PPV_{m,c} − PPV_{m,p}`, with
  `PPV_{m,p} = Σ_c (n_c/n_p) PPV_{m,c}` ⇒ `Σ_c (n_c/n_p) Δ_{m,c} = 0`.
  Zero-occupancy nodes are excluded (`min_occupancy`), never fabricated.
- Full pool: `delta_banks(act_H, valid_H, reg16, [2,4,8,16], delta_cols=None)`
  (nested-runner L202) — column order **level coarse→fine (K=2,4,8,16),
  children 0..K−1 within level, kernels 0..F_H−1 within child**
  (`np.hstack([ch["deltas"][K][:, j, :] for j in range(K)]) for K in kept_K`).
  Total = (2+4+8+16) × 14,994 = **449,820** columns (matches the stored
  `hier_high.candidate_features = 449820`).
- Known structural property (documented, not altered): at K=2 the two child
  deltas are exact anti-correlated rescalings of each other (zero-sum with 2
  children), so their ANOVA-F stats are equal up to float rounding. The pool
  definition is locked at 30 edges/kernel; the supervised selector sees both
  columns. (The previous ENERGY allocator explicitly de-duplicated K=2 child-1;
  that dedup belongs to the ablated allocator and is NOT used here.)

## 6. The ablated allocator (what HIER-HIGH did — now forbidden)

- HIER-HIGH (cap-runner L366–407): in-chain per-edge energies on TRAIN →
  500-permutation shuffled-regime null → `stopping_rule` (L*=16, all levels
  retained) → `level_budgets(E, kept_K, budget=14,994)` (largest-remainder
  energy fractions → {K2:4437, K4:3767, K8:3128, K16:3662}) →
  `select_carriers(edge_e, budgets, kept_K)` (top-energy per level).
  HIER-HIGH-SUP uses **none** of these (no energy, no null, no quotas).

## 7. The retained supervised selector (identical to R5-HIGH)

- `top_f_select(H_dev, y_dev, n_h)` (cap-runner L146): sklearn
  `f_classif` → `np.nan_to_num(nan→0)` → `np.argsort(-f_stat, kind="stable")`
  → top-n_h. Deterministic; NaN/constant handling exactly as R5-HIGH.
- Existing R5-HIGH protocol (cap-runner L388–396), replicated verbatim:
  1. **CV diagnostic** `r5_cv_fixed_rho(G_full, H_full, y_dev, n_g, n_h)`
     (cap-runner L154): 5-fold StratifiedKFold(shuffle=True, random_state=42)
     — for each fold: `top_f_select(H_full[tr], y_dev[tr], n_h)` (**fold-
     internal selection on train-fold labels only**), RidgeClassifierCV(ALPHAS)
     on the fold-train, macro-F1 on the fold-val. Reported, never used for
     selection.
  2. **Final**: `top = top_f_select(H_dev, y_dev, n_h)` on the FULL dev matrix
     (train+val, 896 rows) → `ridge_eval([G_dev[:, :17993] | H_dev[:, top]],
     y_dev, [G_te[:, :17993] | H_te[:, top]], yte)` (nested-runner L167:
     RidgeClassifierCV(alphas=np.logspace(-4,4,20)) fit on train+val, ONE
     official test evaluation).
  So the canonical protocol **does** use train+VAL labels for the final
  feature selection (and train+val for the final Ridge fit) — exactly as the
  existing R5-HIGH did. Test labels are touched only inside `ridge_eval` for
  scoring. This is documented in `diagnostics/selection_protocol.md`.
- ALPHAS = np.logspace(-4, 4, 20) (cap-runner L63) — unchanged.

## 8. Exact train/validation/test boundaries for every decision

| Decision | Data used | Labels used |
|---|---|---|
| z-normalization | per sample (each split separately) | none |
| MiniROCKET fit | train only | none |
| SSL encoder | frozen checkpoint (train-fit historically) | none |
| Tree (hierarchy) | train latents only | none |
| val/test regimes | frozen-tree transform | none |
| candidate pool | frozen transform over train/val/test | none |
| ANOVA-F selection (final) | train+val rows (y_dev) | train+val labels (canonical R5 protocol) |
| ANOVA-F selection (CV diag) | each fold's train rows only | fold-train labels |
| RidgeClassifierCV final fit | train+val rows | train+val labels |
| test evaluation | once, inside ridge_eval | test labels (scoring only) |

## 9. Dimensions asserted end-to-end

- G_high candidate bank: (896, 19,992); G used: first 17,993.
- H candidate pool: (896, 449,820) train+val; selected H: exactly 1,999.
- Test: G_te (3,582, 19,992) → first 17,993; H_te selected (3,582, 1,999).
- Final Z: (896, 19,992) and (3,582, 19,992). **N_G + N_H = 17,993 + 1,999
  = 19,992 asserted.**

## 10. Blockers

None. Every component needed for the one-change ablation exists and is
reused via imports (no reimplementation). The equal-budget comparison is
feasible: all three high-capacity methods used B_high = 19,992 and
HIER-HIGH-SUP matches R5-HIGH's G exactly, changing only the source of the
1,999 H features (hierarchical 449,820 pool + supervised ranking instead of
flat 14,994 pool + supervised ranking) and, vs HIER-HIGH, only the allocator
(energy→supervised) plus the spec-mandated G/H split (§10).
