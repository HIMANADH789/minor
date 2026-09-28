# HIER-SUP-MOD - EB variance-moderated F on the unified pool (UWaveY, seed 42)

The next controlled ablation after HIER-HIGH-SUP / HIER-CONTINUOUS-SUP /
HIER-SUP-CLEAN. **Only the supervised ranking statistic changes.**

| | HIER-SUP-CLEAN | HIER-SUP-MOD |
|---|---|---|
| candidate statistic | raw ANOVA-F | ANOVA pieces (same SS/MS) |
| residual variance | s^2 = SSE/(N-8) (per candidate) | same, then **EB-moderated** |
| prior | none | s^2 ~ s0^2 * F(d_e, d0), Smyth/limma log-moment fit |
| posterior variance | - | s_t^2 = (d0*s0^2 + d_e*s^2)/(d0 + d_e) |
| ranking | F descending | **F_t = MS_between / s_t^2** descending |

Everything else is byte-locked to HIER-SUP-CLEAN: telescoping pool
[D1 | D2 | D4 | D8 | D16] of 469,812 candidates (PPV^(0)=0 makes the root
block D1), canonical split 761/135/3,582, expanded MiniROCKET 84x238,
frozen seed-42 train-only hierarchy, occupancy-weighted deltas, one global
top-19,992 selection (no rho / G-H / quotas), train+val selection labels,
canonical RidgeClassifierCV.

Design invariants (spec 9): d_between = 7 and d_error = N - 8 for EVERY
candidate; occupancy is never used as df, weight, or quota; the fallback
rule is diagnostic only; no candidate is filtered. No pi0/lfdr/two-groups
(that was HIER-EB-SUP) and no feature-value shrinkage.

## Safety gates

- **Raw-F equivalence gate (production)**: the raw ranking reconstructed
  from the same matrix must reproduce the stored HIER-SUP-CLEAN run
  (selected set+sequence, per-candidate F, composition, predictions, alpha,
  Macro-F1 0.7599) before any HIER-SUP-MOD number is accepted.
- Candidate-layout guard: identical semantic column order
  [D1 | D2 | D4 | D8 | D16] on train/val/test (guards the historical
  rank-order vs level-grouped test-layout failure mode).
- Fold-internal CV diagnostic: prior fit + moderation + selection on
  fold-train labels only.

## Usage

```bash
python -m pytest tests/test_uwavey_hier_sup_mod.py -q            # focused
python -m pytest tests/test_uwavey_hier_sup_mod.py -k smoke -q   # full path
python experiments/uwavey_hier_sup_mod/runner.py                 # production
```

## Artifacts (`results/uwavey_hier_sup_mod/seed42/`)

- `results/` - hier_sup_mod.json, comparison.csv (8 methods), summary.md
- `diagnostics/` - dimensions.json, selection_protocol.md,
  variance_prior.json (d0, s0^2, fit diagnostics), variance_statistics.csv
  (per-level medians), occupancy_variance.csv (Spearman),
  occupancy_class_association.csv (per-regime ANOVA + BH),
  fallback_statistics.csv, raw_vs_moderated_selection.csv (overlap /
  Jaccard / in-out), leakage_report.json, equivalence_gate.json
- `figures/` - comparison_macro_f1, raw_vs_moderated_F,
  raw_vs_moderated_variance, occupancy_vs_residual_variance,
  occupancy_vs_raw_F, occupancy_vs_moderated_F,
  selected_features_by_level, fallback_fraction_by_level (8 PNGs)
- `predictions/` - test_predictions.npy

All locked components are imported from the verified prior runners; prior
experiment artifacts are never modified (test-enforced).
