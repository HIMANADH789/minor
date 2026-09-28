# HIER-HIGH-SUP — one-change controlled ablation

**Question:** was HIER-HIGH's poor result (0.7230) caused by the label-free
energy-based feature allocation, rather than by the hierarchical
representation itself?

**Only change:** hierarchical candidate pool (449,820) → **supervised
ANOVA-F ranking, top 1,999** (the exact selector used inside R5-HIGH),
replacing the label-free energy allocator
(`level_budgets`/`select_carriers`, K2=4437/K4=3767/K8=3128/K16=3662 — those
quotas are precisely what is being tested and are NOT reused).

Locked and asserted identical to the capacity-scaled experiment:
expanded MiniROCKET (aeon, `n_kernels=19992`, random_state=42, train-only),
the frozen K=1→2→4→8→16 recursive-2-means hierarchy (train latents only,
byte-identical to the stored `hierarchy_assignments.npy`), the
occupancy-weighted parent/child delta definition (30 edges × 14,994 kernels),
G_high = first 17,993 (R5-HIGH convention, rho=0.1), and the canonical
RidgeClassifierCV protocol. Final budget exactly **19,992**.

## Run

```bash
python experiments/uwavey_hier_high_sup/runner.py --smoke   # ~5-10 min
python experiments/uwavey_hier_high_sup/runner.py           # full run
pytest -q tests/test_uwavey_hier_high_sup.py                # focused tests
```

## Outputs

`results/uwavey_hier_high_sup/seed42/`:
- `results/` — hier_high_sup.json, comparison.csv, summary.md
- `diagnostics/` — dimensions.json, leakage_report.json,
  selection_protocol.md, selected_feature_counts_by_level.csv,
  selected_feature_metadata.csv
- `figures/` — comparison_macro_f1.png, selected_features_by_level.png
- `predictions/` — test_predictions.npy

Selection protocol details: `diagnostics/selection_protocol.md` (written by
the runner from the audited R5-HIGH path). Design audit:
`experiments/uwavey_hier_high_sup/AUDIT.md`.
