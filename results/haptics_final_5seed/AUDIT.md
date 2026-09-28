# HAPTICS FINAL 5-SEED — PHASE 1 AUDIT (2026-09-23)

Scope: UCR Haptics (132 train / 23 val / 308 test, T=1092, 5 classes, univariate,
per-sample z-norm before all branches). Goal: 5 seeds (42-46) for
MiniRocket (MR/M0), HERAMBA (= R5 family; R2 fixed-rho subset counts as the
same HERAMBA family per user clarification), and all external neural baselines,
reusing every valid existing artifact.

## A. Naming resolution (user-confirmed)

- "HERAMBA" == the R5/R2 (rcmkn) HERAMBA family. R2 is the fixed-rho=0.5
  subset of R5; both are the same family and count as HERAMBA evidence.
- The separate `heramba_canonical_ridge` experiment (Haptics seed 42 only) is a
  different CCA-based variant, NOT the HERAMBA 5-seed object of this task.

## B. Existing artifact inventory (verified on disk)

### MiniRocket (M0)
- `results/rcmkn_haptics_seed42/` — canonical deterministic run.
  REPORT.md records M0 = 0.4974 test Macro-F1.
  Convention (confirmed in `experiments/rcmkn_r2_haptics_3seed/config.py` and
  `experiments/rcmkn_final_validation/multiseed.py` docstring): MiniROCKET is
  FIXED at random_state=42 and RidgeClassifierCV(LOO) is deterministic ->
  **M0 is a single deterministic run reused for all seeds** (42-46).
- Seeds 43/44 (and 45/46) therefore do not need fresh MR runs; they reuse the
  deterministic value. A reproduction gate recomputes it to confirm.

### HERAMBA (R2 family, 3-seed audited run)
- `results/rcmkn_r2_haptics_3seed/seed{42,43,44}/result.json`:
    seed 42: val 0.9014  test 0.5470  alpha 4.2813  (canonical ref 0.5500,
             stored val/test slightly differ; gate tolerance 0.011)
    seed 43: val 0.9014  test 0.5213  alpha 4.2813
    seed 44: val 0.8618  test 0.5387  alpha 4.2813
- Protocol audits embedded in the run (AUDIT 1-17): dataset identity, split
  identity, z-norm invariant, MiniROCKET fixed seed 42 (never varied by outer
  seed), G budget 4998, SSL arch equal to canonical R2 config, H formula
  recompute, K=8 VQ, no gating/modulation/hydra, final dim 9996, Ridge alphas
  logspace(-4,4,20) fit train+val, no test information in selection.
- Verdict recorded: ROBUST ACROSS TESTED SEEDS (3/3 > M0).
- `experiments/rcmkn_final_validation/multiseed.py` (M0/R2/R5 wrapper) was
  written but NEVER executed (no multiseed_haptics.json exists).

### External baselines (SUSPICIOUS — Phase 4 target)
- `results/haptics_baselines_3seed/{fcn,inceptiontime,patchtst,resnet1d}/seed{42,43,44}`
- All 12 runs audited via predictions:
    5/12 predict essentially ONE class on test (FCN-42 all 308 -> class 1;
    ResNet1D-43 all 308 -> class 4; FCN-44 306/308 class 1; ITime-42 307/308
    class 1; PatchTST-44 307/308 class 3).
    8/12 select "best" checkpoint at epoch 1-2 (patience-6 collapse region).
- Root cause (matches `experiments/baseline_audit_retrain/runner.py` docstring):
  canonical protocol LR 3e-4 / WD 1e-2 / BS 64 / max 15 epochs / patience 6
  gives 132/64 = 2-3 optimizer steps per epoch and <=45 total steps -> training
  collapse. Models themselves are correct (architectures imported verbatim).
- A legitimate retrain runner exists (`experiments/baseline_audit_retrain/`)
  with validation-only config selection (seed 42) and frozen config for the
  remaining seeds, full training history, single test evaluation — but it was
  NEVER executed (no `results/baseline_audit/` outputs) and covers only seeds
  42-44. This task extends it to seeds 45/46 without touching its frozen
  selection logic.

### Other Haptics dirs (NOT the 5-seed objects; untouched)
- drtn_conditioned_minirocket_* , drtn_haptics_* , rpms_ , haptics_ensemble ,
  haptics_nested_regimes , haptics_intrinsic_heterogeneity ,
  heramba_canonical_ridge{,_full} , heramba_cca{,_ranked} : separate lines.
- HIER-HIGH-SUP etc. (UWaveY line): untouched per task section 15.

## C. Audit table (Method x Seed)

| Method  | Seed | Result exists | Config valid | Output valid | Suspicious | Action |
|---------|-----:|---------------|--------------|--------------|------------|--------|
| MR      |   42 | yes (0.4974)  | yes          | yes          | no         | REUSED (deterministic; reproduction gate) |
| MR      |   43 | (=42, det.)   | yes          | yes          | no         | REUSED via deterministic convention |
| MR      |   44 | (=42, det.)   | yes          | yes          | no         | REUSED via deterministic convention |
| MR      |   45 | (=42, det.)   | yes          | yes          | no         | REUSED via deterministic convention |
| MR      |   46 | (=42, det.)   | yes          | yes          | no         | REUSED via deterministic convention |
| HERAMBA |   42 | yes (0.5470)  | yes (audits) | yes          | no         | REUSED |
| HERAMBA |   43 | yes (0.5213)  | yes (audits) | yes          | no         | REUSED |
| HERAMBA |   44 | yes (0.5387)  | yes (audits) | yes          | no         | REUSED |
| HERAMBA |   45 | no            | -            | -            | -          | NEW RUN (audited run_one_seed path) |
| HERAMBA |   46 | no            | -            | -            | -          | NEW RUN (audited run_one_seed path) |
| FCN     | 42-44 | yes          | yes          | suspect      | YES (collapse) | RETRAIN under corrected protocol |
| InceptionTime | 42-44 | yes     | yes          | suspect      | YES (collapse) | RETRAIN under corrected protocol |
| PatchTST | 42-44 | yes          | yes          | suspect      | YES (collapse) | RETRAIN under corrected protocol |
| ResNet1D | 42-44 | yes          | yes          | suspect      | YES (collapse) | RETRAIN under corrected protocol |
| FCN/ITime/PatchTST/ResNet1D | 45,46 | no | -        | -            | -          | NEW RUNS (corrected protocol) |

## D. Frozen protocol (unchanged, from audited sources)

- Data/split: canonical UCR Haptics 132/23/308, T=1092, 5 classes; single load,
  shared arrays; labels verified by dataset manifest.
- Preprocessing: per-sample z-norm before all branches (audit3 pass).
- MR: aeon MiniROCKET n_features=9996, random_state=42 fixed for all outer
  seeds; classifier RidgeClassifierCV(alphas=logspace(-4,4,20)) fit on
  train+val, internal LOO selects alpha; ONE official test evaluation.
- HERAMBA/R2: X_R2 = [G || H] with G=4998 (MiniROCKET global) and H=4998
  (occupancy-weighted regime heterogeneity from frozen per-seed learned
  context: SSL encoder + VQ K=8, trained per outer seed, seed-varying ONLY in
  the learned context; VQ code identities not comparable across seeds).
- Metric: test Macro-F1 (zero_division=0); val Macro-F1 for checkpoint/config
  selection only. Seeds: outer seeds 42-46 (MR deterministic single extractor).

## E. Baseline bug report (Phase 4, pre-registered)

Baseline:      FCN / InceptionTime / PatchTST / ResNet1D (neural baselines)
Original:      test Macro-F1 0.063-0.248 across seeds 42-44
Anomaly:       single-class or near-single-class test predictions in 5/12
               runs; 8/12 best checkpoints at epoch 1-2; val macro-F1 stuck
               at 0.0714 (= majority-class constant predictor on 23 val
               samples) in collapsed runs.
Root cause:    canonical training protocol under-optimizes on 132 samples:
               BS=64 -> 2-3 steps/epoch, <=45 total steps with patience 6;
               not an architecture error.
Evidence:      per-run predictions.npy class distributions (listed above);
               training histories; matches repo's own diagnosis in
               experiments/baseline_audit_retrain/runner.py docstring.
Correction:    protocol-only retrain (batch 16, max 100 epochs, patience 25,
               OneCycleLR, grad clip 1.0, CE loss) with LR/WD selected on
               VALIDATION ONLY at seed 42 then frozen for all other seeds.
               No architecture changes; test set untouched by any decision.
Affected:      all 12 existing baseline runs (seeds 42-44) + 8 missing (45/46).
Rerun:         experiments/baseline_audit_retrain protocol, extended to seeds
               42-46; existing collapsed runs preserved for the audit trail.

## F. Conclusion of Phase 1

MR: all 5 seeds REUSED (deterministic single run; gate re-verifies 0.4974).
HERAMBA: seeds 42-44 REUSED; 45-46 to run now via the audited path.
Baselines: all 12 existing runs INVALID for benchmark use (documented collapse,
not accepted as legitimate low scores); retrain under corrected protocol for
seeds 42-46. No test-based tuning anywhere.
