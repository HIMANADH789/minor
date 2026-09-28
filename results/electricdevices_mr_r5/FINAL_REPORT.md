# ElectricDevices — FINAL REPORT (MiniRocket vs HERAMBA R5, NO FALLBACK)
Date: 2026-09-23 · Seeds {42, 43, 44} · **3/3 + 3/3 cells VALID**

## 1. Objective
Install the ElectricDevices dataset and run the established 3-seed MR vs
HERAMBA R5 benchmark. **NO FALLBACK** (task directive): no validation-stage
model switching — both arms are final; deltas reported as measured.

## 2. Dataset (installed & verified)
- Source: repo's verified UCRArchive_2018 copy
  (`data/kaggle/_ucrarchive_2018/.../ElectricDevices/`), sha256-16:
  TRAIN bf9b3e4ebb4bcecc, TEST 64ea1c73040b8c60.
- Canonical UCR shapes: TRAIN 8926 × 97 (T=96), TEST 7711 × 97.
- 7 classes; TRAIN 727/2231/851/1474/2406/509/728; TEST
  667/1956/755/1165/1869/743/556 (imbalanced → Macro-F1, zero_division=0).
- Split: canonical train/test never re-split + stratified 15% of train
  as val (random_state=42) → **7587/1339/7711**, asserted in the runner.
- No prior ElectricDevices artifacts in the R2/rcmkn line (repo-wide
  grep: zero hits) → seed-42 values are the FIRST audited references
  (no gate possible; recorded in gates.json).

## 3. Protocol (frozen; identical to the audited rcmkn lines except
the task-directed no-fallback)
- **MR**: aeon MiniRocket(random_state=42, fit TRAIN only) → 9996
  features; RidgeClassifierCV(logspace −4..4, 20 pts) train-only fit →
  val diagnostic; train+val refit → ONE test evaluation. Deterministic
  across seeds (canonical convention; one run shown in every seed cell).
- **HERAMBA R5**: per-seed RCMKNContextModel (SSL + VQ K=8 + joint;
  frozen config byte-identical to the stored canonical configs), H =
  occupancy-weighted regime heterogeneity (min_occ 0.01), X_R5 =
  [G 4998 ‖ H 4998], same Ridge/α grid, train-only fit → val diagnostic;
  train+val refit → ONE test evaluation.
- Seed propagates through Python/NumPy/PyTorch/CUDA via the audited
  core `set_seed`; MiniRocket carriers fixed at 42. No test-driven
  tuning; R5 frozen; MR frozen; test touched exactly once per arm.

## 4. Per-seed results (test Macro-F1)

| Model      | Seed 42 | Seed 43 | Seed 44 | Mean ± Std     |
|------------|--------:|--------:|--------:|----------------|
| MiniRocket |  0.6537 |  0.6537 |  0.6537 | 0.6537 ± 0.0000 |
| HERAMBA R5 |  0.6669 |  0.6688 |  0.6669 | **0.6675 ± 0.0011** |

Validation Macro-F1: MR 0.8874 (α 4.2813); R5 0.8797 / 0.8814 / 0.8918
(α 1.6238 all seeds). Accuracy: MR 0.7400; R5 0.7478 / 0.7523 / 0.7497.

## 5. Matched-seed Δ (HERAMBA − MR, final — no fallback applied)
| Seed | MR | HERAMBA R5 | Δ |
|---:|---:|---:|---:|
| 42 | 0.6537 | 0.6669 | **+0.0132** |
| 43 | 0.6537 | 0.6688 | **+0.0151** |
| 44 | 0.6537 | 0.6669 | **+0.0132** |

Mean Δ **+0.0138** · median +0.0132 · std 0.0011 · HERAMBA wins **3/3**
seeds. No significance claims (n = 3). Note: validation ranks MR first
(0.8874 vs 0.8797–0.8918) while test ranks HERAMBA first on all seeds —
under a fallback protocol the final cells would have been the MR values;
under this task's no-fallback directive the raw HERAMBA results ARE the
final results and are reported as measured.

## 6. Per-cell validation
- Split identity asserted (7587/1339/7711, T=96, 7 classes); val_source
  recorded (stratified_15pct_of_train_seed42).
- Feature budget 9996 (4998 G + 4998 H) asserted; no NaN/Inf at features
  or X_R5; extractor identity 0.0; regimes deterministic per seed
  (re-extraction equality, 3/3 true).
- Predictions: 6/6 files — length 7711, finite, all 7 classes predicted
  in every cell (no collapse); class balance recorded in result.json.
- One-shot test evaluation per arm; α recorded per cell; context_train
  keys identical across seeds (seed-dependent training curves only) —
  no configuration drift.

## 7. Computational cost
MR arm 80 s (shared across seeds, deterministic). HERAMBA R5 per seed:
720 s / 1086 s / 1315 s (~12–22 min; the 3-seed context training
dominates). Seed 42 was completed in the first launch and RESUMED from
disk (never re-executed); seeds 43/44 executed fresh.

## 8. Status
ElectricDevices MR = 3/3 VALID (deterministic control, first references
recorded) · HERAMBA R5 = 3/3 VALID (per-seed contexts, all audits
passed) · NO FALLBACK applied · deltas honest and unmodified.

## 9. Artifacts
`results/electricdevices_mr_r5/` — AUDIT.md, FINAL_REPORT.md,
per_run_results.json, gates.json, seed{42,43,44}/
{mr_result.json, heramba_r5_result.json, mr_predictions.npy,
heramba_r5_predictions.npy} · `experiments/electricdevices_mr_r5/`
runner.py · logs electricdevices_mr_r5.log, electricdevices_mr_r5_resume.log.
