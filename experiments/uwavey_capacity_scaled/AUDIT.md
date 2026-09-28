# AUDIT — existing implementation facts (derived from code, not prior notes)

Experiment: `uwavey_capacity_scaled` (capacity-scaling controlled experiment).
Every statement below was verified against the current repository code.

## 1. How canonical MiniROCKET generates the G feature pool

- Fit: `aeon.transformations.collection.convolution_based.MiniRocket(
  random_state=42, n_jobs=-1)` (see `experiments/uwavey_nested_hierarchical/
  runner.py::fit_minirocket`; identical convention in the R2/R5 runners via
  aeon's direct `transform`).
- Input: per-sample z-normalized TRAIN signals only (`znorm` with
  `std + 1e-8`, float32) — the project-wide convention.
- aeon internals (`aeon/.../convolution_based/_minirocket.py`):
  - exactly **84 physical kernels** (`n_kernels = 84` inside `_static_fit`);
  - the constructor parameter `n_kernels` is actually the **TOTAL FEATURE
    COUNT**: `n_features_per_kernel = n_features // 84`, and the output
    width = 84 × (quantiles per kernel);
  - default `n_kernels=10_000` → 10_000 // 84 = **119 quantiles per kernel**
    → total width 84 × 119 = **9,996** at T=315 (21 unique dilations,
    `n_features_per_dilation` summing to 119, biases = 9,996 golden-ratio
    quantiles; verified on the actual UWaveY train fit:
    `biases.shape=(9996,)`).
- The repository additionally uses a numba per-timestep path
  (`experiments/drtn_conditioned_minirocket_transfer_seed42/core.py::
  compute_raw_activations` + `ppv_from_activations`), identity-verified
  against aeon's transform (drift ≤ 1.9e-3 from an aeon-version quantile
  artifact; the per-run recompute is the repository convention).

## 2. Why the current G dimensionality is 4,998

Canonical convention (DRTN/RCMKN line): the 9,996 MiniRocket features are
split in half by feature index — **G bank = features [0:4998]** (the
"kernel half"), **H-bank kernels = features [4998:9996]** (the heterogeneity
half). The fixed total budget P = 9,996 with rho = 0.5 gives
N_G = 4,998 + N_H = 4,998 (the R2/HERAMBA budget). So 4,998 is a
**budget-split convention**, not a property of the extractor.

## 3. How the canonical ~9,996 total output is represented

- M0 (MiniROCKET-only): all 9,996 features.
- R2/HERAMBA flat: `[G(4,998) || H_flat(4,998)]` = 9,996.
- Split indices: official UCR train 896 → stratified 15% val (seed 42) →
  train 761 / val 135 / test 3,582 (`experiments/rcmkn_r2_uwave_seed42/
  data.py::load_and_split`, asserted 761/135/3582, T=315, 8 classes).

## 4. How R5/HERAMBA constructs G and H; the rho/allocation rule

`experiments/rcmkn_r5_uwave_hbudget_seed42/` (config.py + runner.py):

- **P = TOTAL_BUDGET = 9,996** (fixed predeclared budget).
- `rho` = **H-fraction**: `N_H = int(round(rho * P))`, `N_G = P - N_H`
  (`budget_split`; sum exactly P for every rho; rho=0.5 → 4,998/4,998).
- **G selection: FIRST N_G canonical MiniROCKET features** (fixed ordering,
  never label-ranked).
- **H selection: top-N_H by ANOVA F-statistic (sklearn `f_classif`),
  recomputed INSIDE every CV training fold** (train-fold labels only).
- **rho selection: 5-fold stratified CV on train+val (dev) set**, mean CV
  Macro-F1, tie tolerance 0.001 favors smaller rho; final fit on dev,
  official test evaluated exactly once.
- Stored UWaveY result (`C:/temp/results/r5_uwave_hbudget_seed42/
  UWaveGestureLibraryY/result.json`): cv_curve means 0.7722 / 0.7788 /
  0.7668 / 0.7636 / 0.7688 / 0.7754 for rho 0.0..0.5 →
  **selected_rho = 0.1** → N_G = 8,996, N_H = 1,000; r5.test_macro_f1 =
  0.7720, selected_alpha ≈ 11.2884; controls m0_test = 0.7539,
  r2_test = 0.7551.
- H_flat construction (unchanged): `compute_regime_heterogeneity(
  act[:, 4998:], valid[4998:], regimes)` — flat K=8 hard-VQ regimes from
  the frozen seed-42 SSL context checkpoint; H_m = Σ_k q_k (PPV_{m,k} −
  PPV_m)² with valid-region masks, min-occupancy 0.01, renormalized q_k.

**rho/allocation rule for R5-HIGH: EXISTS and is reused** — rho\*=0.1 is the
documented, already-validation-selected allocation on this exact
dataset/split. Per the task ("use that exact rule"), R5-HIGH keeps
rho = 0.1 at B_high: N_G = 17,993 first-ordered G_high features, N_H = 1,999
top-f_classif H_high features. NO new validation search over rho is run.

## 5. How the fixed total feature budget is enforced

R5 `assemble()`: `np.hstack([G_full[:, :n_g], H_full[:, h_idx[:n_h]]])` with
`assert X.shape[1] == TOTAL_BUDGET` — hard dimensionality assertion.

## 6. How hierarchical H (K=1→2→4→8→16) is constructed

`models/nested_regimes/model.py` + `experiments/uwavey_nested_hierarchical/`:

- What is partitioned: frozen `SSLTemporalEncoder` (d=32) temporal latents
  of the stored seed-42 UWaveY context checkpoint
  (`results/r2_uwave_seed42/UWaveGestureLibraryY/checkpoints/
  context_model_seed42.pt`), pooled over TRAIN samples (761×315 = 239,715
  vectors).
- Recursive 2-means: `KMeans(n_clusters=2, n_init=10, random_state=42)` per
  node on its own member latents; binary path ids 2p/2p+1 (bit-0 = lower
  first-centroid coordinate); 0 singleton fallbacks; frozen tree applied
  transform-only to val/test.
- Per-timestep activations for H-bank kernels + valid-region masks; padding
  groups by valid slice; min-occupancy 0.01 at level 16 (argmax-count
  fallback); levels restricted to the occupied-finest support (nested, so
  coarse exclusions are vacuous).
- Occupancy-weighted nesting: PPV_parent = Σ_c n_c PPV_c / Σ_c n_c over
  occupied children; Δ_c = PPV_c − PPV_parent → Σ_c n_c Δ_c = 0 exactly.
- Chain invariants (measured, seed 42 run): nested ANOVA identity
  Var16 = E2+E4+E8+E16 to 5.8e-16 relative; zero-sum 2.3e-16; max
  cross-level inner product 1.9e-13; no Gram-Schmidt/CCA (AST-enforced).
- Label-free null: S=500 permutations of the level-16 regime sequence
  (occupancy multiset preserved), seed 52042, plus-one p, BH-FDR; stopping
  rule "real E_l > null p95 AND q<0.05, coarse→fine" → L\*=16 on UWaveY
  (all four levels 7.7–29.5× above the null p95).

## 7. Which MiniROCKET kernel responses are reused across H variants

ALL H variants (flat and hierarchical) consume the SAME per-timestep
activation indicators `act` (n, F_het, T) + `valid` (F_het, T) from
`compute_raw_activations` — flat H uses the 3seed
`compute_regime_heterogeneity` on `act[:, N_GLOBAL:]`; hierarchical H uses
the nested chain on the same slice. With the expanded extractor both
operate on features [4998:19992] of the expanded bank (9,994 kernels-side
features → H-side = features 4,998..19,991, i.e. 14,994... **exact split:
expanded total 19,992; G-side = [0:4998] canonical convention extended to
[0:17,993] by first-N ordering; H-side kernels = [4,998:19,992] (14,994
kernel units)** — see §9.

## 8. How features are selected/truncated when a budget is imposed

- Fixed-ordering truncation: `G_full[:, :N_G]` (first-N convention).
- Label-ranked: ANOVA-F top-N_H, fold-internal (R5 H selection).
- Energy-ranked (hierarchical line): per-edge structural energies
  e_{m,c,l} = mean_i π_c Δ² computed exactly inside the chain; top-e per
  level under a largest-remainder level allocation
  (`models/hierarchical_budget/model.py`; sums exactly to the budget).

## 9. Capacity expansion mechanism (this experiment)

aeon MiniRocket parameterization: `MiniRocket(n_kernels=B)` yields output
width = 84 × (B // 84) when B//84 ≤ 32 dilation slots... precisely:
`n_features_per_kernel = B // 84`; `true_max_dilations = min(nfpk, 32)`;
per-dilation counts sum to nfpk; **width = 84 × nfpk**. Empirically verified
on UWaveY train (T=315): n_kernels=42→84, 84→84, 168→168, 252→252, 336→336,
504→504; default 10,000 → 9,996.

- **B_base = 9,996** (canonical total budget, from the implementation).
- **B_high = 2 × B_base = 19,992** = 84 × 238 → `MiniRocket(n_kernels=19_992,
  random_state=42)` gives width exactly 19,992 (nfpk 119 → 238 = exactly 2×
  quantiles per kernel; same 84 physical kernels and the same 21-dilation
  grid by seed; quantiles are a different slice of the golden-ratio
  sequence → genuinely additional, non-duplicated features).
- Kernel capacity factor = 2.000× (84 kernels both; per-kernel quantiles
  119 → 238). Feature split on the expanded bank (index convention
  preserved): H-side kernels = features [4,998 : 19,992] (14,994 units).
- Determinism: same `random_state=42`; train-only fitting; deterministic
  feature ordering (index order = aeon emission order).

## 10. Fairness check: equal final budgets are possible

All three high-capacity models can be built at exactly B_high = 19,992:
MR-HIGH (first 19,992 expanded MiniRocket features = the whole expanded
bank), R5-HIGH (rho=0.1 rule: 17,993 G + 1,999 H_f-classif), HIER-HIGH
(label-free energy allocation of 19,992 = G_high 4,998-first-N + 9,999
budgeted hierarchical details... allocation fixed in CONFIG.yaml: G part =
4,998 (first-N canonical convention), H part = 9,994 energy-allocated
hierarchical details). No blocker; the experiment proceeds.

## 11. G/H decomposition convention for the HIGH models

To keep a single interpretable accounting across methods:

- The **G part** of every HIGH model = FIRST 4,998 features of the expanded
  bank (the canonical G-bank convention, unchanged).
- MR-HIGH additionally takes the remaining 14,994 expanded features as
  global MiniRocket features (first-N ordering) → 19,992 total.
- R5-HIGH: G part 4,998 + next 12,995 first-ordered expanded global
  features (= 17,993 total G) + 1,999 top-f_classif H features = 19,992.
- HIER-HIGH: G part 4,998 + 9,994 energy-allocated hierarchical detail
  features from the H-side kernel slice [4,998:19,992] = 19,992.

(The exact per-part numbers are re-asserted in code and recorded in
`diagnostics/dimensions.json`.)
