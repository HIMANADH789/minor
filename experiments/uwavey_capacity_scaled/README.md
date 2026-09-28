# uwavey_capacity_scaled

Capacity-scaling controlled experiment on UWaveGestureLibraryY (seed 42).

**Question:** does increasing the total representation/classifier capacity
allow the continuous hierarchical regime-conditioned representation to
exploit the multi-resolution structure previously detected on UWaveY?

- `AUDIT.md` — every reused fact derived from the current code (MiniRocket
  `n_kernels` = total-feature-count mechanism, 84 kernels × 119 → 9,996;
  expanded 84 × 238 → 19,992; the R5 rho rule with the stored
  validation-selected rho = 0.1; the canonical split/budget conventions).
- `CONFIG.yaml` — all predeclared constants.
- `runner.py` — the experiment: canonical baselines (MR / R2-flat / R5),
  then MR-HIGH, R5-HIGH, HIER-HIGH, all at exactly B_high = 19,992 final
  features.
- Budget fairness: the three high-capacity models have identical final
  dimensionality entering the same canonical RidgeClassifierCV protocol.

Run: `python experiments/uwavey_capacity_scaled/runner.py [--smoke]`

Results: `results/uwavey_capacity_scaled/seed42/` (results/, figures/,
diagnostics/, predictions/). Tests: `tests/test_uwavey_capacity_scaled.py`
(set RUN_SMOKE=1 to include the end-to-end smoke run).

No existing experiment artifacts are modified.
