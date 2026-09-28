# HIER-SUP-CLEAN — clean unified telescoping hierarchy (UWaveY, seed 42)

The complete model is ONE telescoping multiresolution hierarchy of increments
with `PPV^(0) = 0`:

```
Delta^(1)_m     = PPV_m^(1) - PPV_m^(0) = PPV_m^(1)              (root increment)
Delta_{m,c}^(K) = PPV_{m,c}^(K) - PPV_{m,parent(c)}^(K/2)        K in {2,4,8,16}
```

Candidate pool `C = [D1 | D2 | D4 | D8 | D16]` in `R^(N x 469,812)`
(19,992 root increments + 449,820 deeper deltas). ONE supervised ANOVA-F
ranking (`top_f_select`, train+val labels) picks exactly 19,992 features;
the composition across levels fully emerges. No rho, no G/H split, no
protected budgets, no per-level quota, no energy allocation, no empirical
Bayes, no moderated-F, no new classifier.

## Scientific status: canonicalization, not a new method

HIER-CONTINUOUS-SUP already used this exact pool: its "root / level-0"
block IS `Delta^(1)` because `PPV^(0) = 0`. This experiment is therefore the
clean mathematical reformulation of the same model. The runner contains an
**equivalence gate** against the stored HIER-CONTINUOUS-SUP run
(`results/uwavey_hier_continuous_sup/seed42`): selected index set,
per-candidate ANOVA-F, semantic metadata, composition, test predictions and
Macro-F1 must all match, verdict `IDENTICAL` (asserted in production;
skipped on the smoke subset). A different result would indicate an
implementation divergence — not a methodological difference — and fails the
run before any scientific interpretation.

## Usage

```bash
# focused tests (fast)
python -m pytest tests/test_uwavey_hier_sup_clean.py -q
# full-path smoke (small subsets; asserts 469,812 -> 19,992)
python -m pytest tests/test_uwavey_hier_sup_clean.py -k smoke -q
# production
python experiments/uwavey_hier_sup_clean/runner.py
```

## Artifacts (`results/uwavey_hier_sup_clean/seed42/`)

- `results/` — hier_sup_clean.json, comparison.csv (7 methods), summary.md
- `diagnostics/` — dimensions.json, selection_protocol.md,
  selected_counts_by_level.csv, selected_feature_metadata.csv,
  leakage_report.json, equivalence_check.json
- `figures/` — comparison_macro_f1.png, selected_features_by_level.png,
  candidate_pool_composition.png
- `predictions/` — test_predictions.npy

All locked components are imported from the verified prior runners
(`uwavey_nested_hierarchical`, `uwavey_capacity_scaled`,
`uwavey_hier_high_sup`); nothing is reimplemented. Prior experiment
artifacts are never modified (test-enforced).
