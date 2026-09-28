# HAPTICS FINAL 5-SEED VALIDATION AND BASELINE AUDIT — FINAL REPORT
Date: 2026-09-23. Dataset: UCR Haptics (132 train / 23 val / 308 test,
T=1092, 5 classes, univariate, per-sample z-normalization). Seeds: 42-46.

## 1. Executive summary

Every required cell is now VALID (status_matrix.csv: all six methods
COMPLETE). MiniRocket is deterministic (one gated run reused across all five
seeds); HERAMBA reuses the audited 3-seed run (42/43/44) plus two new runs
(45/46) through the identical audited per-seed path; all four neural
baselines were retrained under a corrected, validation-only-tuned protocol
after the original runs were found to be invalid due to a documented
training-collapse pathology.

## 2. Final benchmark (test Macro-F1)

| Method        | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean +- Std   |
|---------------|--------:|--------:|--------:|--------:|--------:|---------------|
| MiniRocket    |  0.4974 |  0.4974 |  0.4974 |  0.4974 |  0.4974 | 0.4974 +- 0.0000 |
| HERAMBA       |  0.5470 |  0.5213 |  0.5387 |  0.5323 |  0.5182 | 0.5315 +- 0.0120 |
| InceptionTime |  0.4230 |  0.5154 |  0.3657 |  0.4595 |  0.4384 | 0.4404 +- 0.0545 |
| FCN           |  0.3638 |  0.3886 |  0.3959 |  0.3808 |  0.4147 | 0.3888 +- 0.0188 |
| ResNet1D      |  0.4027 |  0.4530 |  0.3697 |  0.4304 |  0.3373 | 0.3986 +- 0.0463 |
| PatchTST      |  0.3940 |  0.2266 |  0.3579 |  0.3828 |  0.3044 | 0.3331 +- 0.0688 |

Validation Macro-F1 is reported separately in FINAL_TABLE.md (never mixed
with test). Std is the sample standard deviation (ddof=1) across the five
per-seed test Macro-F1 values, per the frozen convention.

## 3. Paired MR vs HERAMBA (per-seed differences d_s = HERAMBA_s - MR_s)

    seed 42: +0.0496    seed 43: +0.0239    seed 44: +0.0413
    seed 45: +0.0349    seed 46: +0.0208
    mean +0.0341 | median +0.0349 | std 0.0120

HERAMBA exceeds MiniRocket on all five seeds (5/5 positive). With n=5 this
is a descriptive stability statement only; no significance claim is made.

## 4. What was reused vs newly run

- MiniRocket: REUSED for all five seeds. The canonical M0 convention is
  deterministic (aeon MiniROCKET random_state=42 fitted on train only;
  RidgeClassifierCV LOO deterministic), so one run legitimately serves all
  seeds. A reproduction gate re-ran the pipeline: observed 0.4974 ==
  canonical 0.4974 (tol 0.011) -> PASS (mr_seed42_gate.json).
- HERAMBA (R5 family; R2 fixed-rho subset per user clarification):
  REUSED seeds 42/43/44 from the audited results/rcmkn_r2_haptics_3seed run
  (0.5470 / 0.5213 / 0.5387, with embedded AUDIT 1-17: dataset identity,
  split identity, z-norm, MiniROCKET seed fixed at 42, G/H budgets, VQ K=8,
  H-formula recompute, Ridge protocol, no-test-in-selection).
  NEW seeds 45/46 run through the SAME run_one_seed entry point with the
  SAME frozen config: 0.5323 and 0.5182; both predict all 5 classes.
- Baselines: NO existing baseline seed was reused as benchmark evidence.
  All 12 original runs (seeds 42-44, four models) were audited and found
  INVALID for benchmark use (see bug report below); all 20 cells
  (4 models x 5 seeds) come from the corrected retrain.

## 5. Baseline bug report (documented, not hidden)

Baseline:         FCN, InceptionTime, ResNet1D, PatchTST (neural baselines)
Original result:  test Macro-F1 0.063-0.248; e.g. FCN seed42 = 0.0634,
                  ResNet1D seed43 = 0.0715.
Observed anomaly: 5/12 original runs predict essentially ONE class on the
                  308-sample test set (FCN-42: all 308 -> class 1; ResNet1D-43:
                  all 308 -> class 4); 8/12 select their "best" checkpoint at
                  epoch 1-2; collapsed runs' val macro-F1 = 0.0714 (the
                  constant-predictor value on the 23-sample val split).
Root cause:       canonical training protocol (BS=64, max 15 epochs,
                  patience 6) yields 2-3 optimizer steps/epoch and <=45 total
                  steps on 132 training samples -> training collapse. Not an
                  architecture error.
Evidence:         per-run predictions.npy class distributions
                  (BASELINE_AUDIT.csv), training histories, and the repo's
                  own pre-existing diagnosis in
                  experiments/baseline_audit_retrain/runner.py.
Correction 1      (objective implementation bug, fixed): the retrain runner
                  fed (N, T) tensors to conv1d models that require
                  (N, 1, T) — it crashed immediately
                  ("expected input[1, 16, 1093] to have 1 channels, but got
                  16 channels instead"). Fixed by adding the channel
                  dimension at all three data-feeding sites, matching the
                  original baselines runner's X[:, None, :] convention.
Correction 2      (protocol, pre-registered in the runner): batch 16, max 100
                  epochs, patience 25, OneCycleLR, grad clip 1.0, CE loss;
                  LR/WD grid {3e-4, 1e-3} x {1e-2, 1e-4} selected on
                  VALIDATION ONLY at seed 42, then frozen for seeds 43-46.
                  All selected configs: lr=1e-3, wd=1e-4.
Affected seeds:   all 12 original baseline runs (42-44) + 8 missing (45/46).
Rerun protocol:   experiments/baseline_audit_retrain/runner.py (32 runs:
                  16 selection + 16 frozen-seed runs), outputs under
                  results/baseline_audit/retrained_haptics/.
Final result:     the section-2 table. SANITY_CHECKS.csv: 22 retrained/new
                  runs checked, 0 single-class, max class fraction 0.5097 —
                  no residual collapse. Original collapsed artifacts are
                  preserved as the audit trail, untouched.

No test labels were used for any selection, tuning, or checkpoint decision.
No baseline was tuned to beat HERAMBA; the corrections above are the
runner's pre-registered protocol plus one objectively identified shape bug.

## 6. Fairness notes

- Identical Haptics split, labels, and official test set for every method
  (single canonical loaders; asserted 132/23/308 in both runners).
- Per-sample z-normalization: identical canonical formula
  ((X - mean)/(std + 1e-8)) used by MR, HERAMBA, and the baselines.
- MiniROCKET random_state is fixed at 42 for all outer seeds by the frozen
  repo convention (documented in the audited R2 config: "never varied by the
  outer seed"); HERAMBA varies only its learned context per outer seed.
- Where representations differ intrinsically (MiniROCKET transform vs raw
  wavefronts for neural nets), the difference is documented in
  CONFIG_AUDIT.csv rather than forced identical.

## 7. Audit checklist

[x] MR has 5 valid seeds (deterministic, gated)     [x] all seeds are 42-46
[x] HERAMBA has 5 valid seeds                       [x] valid runs reused
[x] every baseline has 5 valid seeds                [x] missing runs completed
[x] suspicious baselines audited                    [x] invalid runs rerun
[x] no test leakage                                 [x] no test-based tuning
[x] correct Haptics split asserted                  [x] labels correct
[x] predictions valid (no unexplained collapse)     [x] configs recorded
[x] per-seed metrics saved                          [x] mean +- std consistent
[x] final tables generated                          [x] audit report generated

## 8. Artifacts

results/haptics_final_5seed/
    AUDIT.md, FINAL_RESULTS.csv, AGGREGATED_RESULTS.csv, PER_SEED_RESULTS.csv,
    BASELINE_AUDIT.csv, SANITY_CHECKS.csv, CONFIG_AUDIT.csv, status_matrix.csv,
    FINAL_TABLE.md, PAIRED_MR_VS_HERAMBA.json, mr_seed42_gate.json,
    phase2_mr.json, phase3_heramba_new.json, heramba_seed{45,46}/,
    mr_test_predictions.npy
experiments/haptics_final_5seed/
    phase2_3_mr_heramba.py, aggregate.py
results/baseline_audit/retrained_haptics/
    all_runs.json, selected_configs.json, per-model selection + seed runs

## 9. Scope statement (stop condition honored)

Only Haptics was touched; no other dataset, seed, preprocessing, or model was
modified; HERAMBA and MiniRocket implementations are untouched (seeds 45/46
ran through their unmodified audited code path); HIER-HIGH-SUP and LARP were
not started.
