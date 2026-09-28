# EpilepticSeizures HERAMBA R5 — AUDIT
Date: 2026-09-23. Verdict: 3/3 VALID. CWRU PAUSED (zero CWRU processes
launched; all CWRU artifacts preserved: results/ccwru_final_3seed/ complete
with 4/4 gates PASS from before the switch).

## Phase 1 — CWRU pause
- Verified 0 python processes running before starting ES work.
- No new CWRU experiment launched; no CWRU artifact modified.
- (CWRU was already complete when the pause instruction arrived: all 12
  arms + gates + reports finished; state preserved as-is.)

## Phase 2 — discovery
- Repo-wide search for EpilepticSeizures found exactly one result source:
  results/rcmkn_ssl_context_transfer_seed42/EpilepticSeizures/
  (result.json with R0/R2/C1/C2, context_model_seed42.pt, predictions,
  diagnostics) plus the per_dataset_results.json entry.
- Canonical seed-42 R2 = 0.9457 test Macro-F1 (val 1.0, alpha 1.6238,
  feature_dim 9996). This is the HERAMBA-family R5 object per the
  user-established naming (R5 family; R2 fixed-rho subset).
- No other ES artifacts (drtn line has no ES row; baseline_bench has no ES
  file). Seeds 43/44 missing -> status matrix REUSE(42)/RUN(43)/RUN(44).

## Canonical identification and freeze
- R5 = rcmkn R2 composition [G 4998 || H 4998] with per-seed learned
  context (RCMKNContextModel SSL+VQ K=8+joint), identical frozen config to
  the ES seed-42 producer (verified byte-equal configs across the rcmkn
  lines), occupancy-weighted H (min_occupancy 0.01, valid-region masks),
  RidgeClassifierCV logspace(-4,4,20), train-only -> val diagnostic /
  train+val refit -> one test eval. Architecture untouched.
- Split: PROVIDED canonical val (80/20/11420), T=178, 2 classes; loaded
  once, shared across seeds; split seed never conflated with outer seed.
  Asserted in the runner.

## Seed control
- Outer seed propagated via the audited core set_seed (Python/NumPy/
  torch/CUDA) before model init, SSL/joint training (seeded dataloader
  generator), and regime extraction; per-seed regime determinism asserted
  by re-extraction; regime hash per seed recorded.
- MiniRocket carriers fixed at random_state=42 (canonical convention:
  the extractor bank is part of the frozen protocol, not the outer seed).

## Gate
Fresh seed-42 run (new per-seed context): 0.9461 vs canonical stored R2
0.9457, tol 0.011 (repo R2 gate scale) -> PASS. The canonical seed-42
artifact remains the reference and was not modified.

## Verification checklist
[x] CWRU paused / preserved          [x] R5 implementation identified
[x] R5 frozen (no changes)           [x] canonical seeds {42,43,44}
[x] existing seed audited (42)       [x] only missing seeds executed
[x] 3/3 valid R5 seeds               [x] no test leakage
[x] no test-driven tuning            [x] predictions valid (11,420 each)
[x] validation metrics recorded      [x] test metrics recorded
[x] mean ± std calculated            [x] report + audit generated

Runtime: full 3-seed production ~3.5 min total (small train split);
mechanical smoke passed before launch (subset through every stage).
