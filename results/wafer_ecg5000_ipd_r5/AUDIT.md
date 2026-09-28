# Wafer / ECG5000 / ItalyPowerDemand — AUDIT (Phase 1–2)

Scope: Wafer, ECG5000_UNBAL (user-selected variant), ItalyPowerDemand ×
{MiniRocket, HERAMBA R5} × {42,43,44} = 18 cells. HERAMBA = rcmkn R2
family (user-confirmed naming: R2 fixed-rho [G‖H] composition is the
R5/HERAMBA family; per-seed learned context is the only seed-varying
component).

## Canonical references found (Phase 1)

| Dataset | Artifact | Split | T | Cls | M0/MR ref | R2 ref |
|---|---|---|---|---|---|---|
| Wafer | none (never run in the R2 line) | UCR TSV 1000/6164 | 152 | 2 | — | — |
| ECG5000_UNBAL | rcmkn_ssl_context_transfer_seed42/ECG5000_UNBAL/result.json | ecg5000_resplit.npz → 3400/600/1000 | 140 | 5 | 0.5938 (corrected retest M0, val 0.8224, α 11.2884) | 0.5894 (val 0.8170) |
| ItalyPowerDemand | rcmkn_r2_kaggle_context2_seed42/per_dataset_results.json | UCR TSV 67 → 56/11 val, test 1029 | 24 | 2 | 0.9650 (val 1.0) | 0.9592 (val 1.0) |

Data verification: all three datasets present in the verified
UCRArchive_2018 copy (sha-identical to the repo's kaggle TSVs for IPD);
Wafer 1000×152/6164×152 labels {-1,+1}; ECG5000 TSV 500/4500 (the
benchmark object uses the canonical ecg5000_resplit.npz 4000/1000 with
stratified-15% val @ seed 42 = 3400/600/1000 — same loader
load_npz_dataset as the audited transfer line); ItalyPowerDemand
67/1029 T=24 labels {1,2}.

## Status matrix (initial)

| Dataset | Model | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|---|
| Wafer | MiniRocket | MISSING | MISSING | MISSING |
| Wafer | HERAMBA R5 | MISSING | MISSING | MISSING |
| ECG5000_UNBAL | MiniRocket | GATE ref (M0 0.5938) | MISSING | MISSING |
| ECG5000_UNBAL | HERAMBA R5 | GATE ref (R2 0.5894) | MISSING | MISSING |
| ItalyPowerDemand | MiniRocket | GATE ref (M0 0.9650) | MISSING | MISSING |
| ItalyPowerDemand | HERAMBA R5 | GATE ref (R2 0.9592) | MISSING | MISSING |

Per the repo's established 3-seed convention (multiseed.py,
rcmkn_r2_haptics_3seed, ccwru_final_3seed, gunpoint_phoneme_forda_r5):
MR is deterministic (MiniRocket random_state=42 + deterministic Ridge) —
one canonical run serves all three seed cells, gated at seed 42 against
the stored reference. HERAMBA R5 retrains its context per outer seed;
seed 42 is gated against the canonical stored R2 value (tol 0.011 repo
R2-gate scale; MR gates tol 0.0011).

Execution list: all 18 cells via the unified runner (3 × 3 × 2), with
seed-42 cells asserted against the gates above. No stored artifact is
overwritten; existing canonical directories untouched.

## Frozen protocol (identical to audited lines; no R5/MR modifications)

- Loaders: Wafer/IPD = UCR TSVs via the context3 load_kaggle_ucr path
  (canonical train/test + stratified 15% of train @ random_state=42 as
  val); ECG5000_UNBAL = load_npz_dataset (ecg5000_resplit.npz,
  stratified 15% @ seed 42). z-norm per-sample (same formula), MiniRocket
  random_state=42 fit on TRAIN only → 9996 features (4998 G + 4998 H).
- HERAMBA R5: per-seed RCMKNContextModel (SSL+VQ K=8+joint, frozen
  config byte-identical to rcmkn_haptics_seed42/config and the stored
  transfer/important2 configs), H = occupancy-weighted regime
  heterogeneity (min_occ 0.01), X_R5 = [G‖H], RidgeClassifierCV
  (logspace −4..4, 20 pts), train-only fit → val diagnostic; train+val
  refit → ONE test evaluation.
- FALLBACK (task-defined, validation-stage only, same as
  gunpoint_phoneme_forda_r5): final arm = HERAMBA candidate iff its
  train-only-fit val Macro-F1 ≥ MR's train-only-fit val Macro-F1 on the
  same frozen features/split; else the identical MR model. Selection
  uses validation only; the final selected model is evaluated on test
  exactly once. Raw-HERAMBA test metrics are recorded for audit but
  NEVER used for selection; no max() over test.
- Wafer gate note: no stored MR/R2 reference exists → seed-42 values
  become the first audited references for Wafer (repo convention, cf.
  Phoneme in the transfer line). IPD/ECG5000 gates assert as above.
- Class imbalance: Wafer train ~1080/−1 vs ~... (assert recorded);
  metric = Macro-F1 (zero_division=0) + accuracy recorded.


---

## Phase 20 � FINAL AUDIT (post-execution)

### Dataset coverage
3 datasets (Wafer, ECG5000_UNBAL, ItalyPowerDemand) x 2 models x 3 seeds
= 18/18 cells. All present, all validated.

### Seed coverage
{42, 43, 44} (repository standard).

### Reused experiments
- Canonical stored seed-42 references, untouched: ECG5000_UNBAL M0 0.5938
  (drtn_conditioned_minirocket_corrected_negative_retest) and R2 0.5894
  (rcmkn_ssl_context_transfer_seed42); ItalyPowerDemand M0 0.9650 / R2
  0.9592 (rcmkn_r2_kaggle_context2_seed42). These served as gates.
- Wafer 3 seeds from the first launch were RESUMED from their on-disk
  result.json (never re-executed) after a loader-key fix; the crash
  occurred after Wafer had fully completed.

### Newly executed experiments
All 18 cells (6 Wafer + 6 ECG5000_UNBAL + 6 IPD), through the unified
runner with the frozen recipe.

### Reruns
NONE. (The Wafer cells were resumed-from-disk, not rerun; no cell was
re-executed after completing.)

### Invalid artifacts
NONE.

### Missing experiments
NONE.

### Gates (post-execution)
- ECG5000_UNBAL MR seed42: 0.5938 vs 0.5938 (tol 0.0011) PASS (exact)
- ECG5000_UNBAL R5 seed42: 0.5846 vs 0.5894 (tol 0.011) PASS
- ItalyPowerDemand MR seed42: 0.9650 vs 0.9650 (tol 0.0011) PASS (exact)
- ItalyPowerDemand R5 seed42: 0.9592 vs 0.9592 (tol 0.011) PASS (exact)
- Wafer: first audited references MR 0.9962 / R5 0.9983 (no prior
  artifact; repo convention cf. Phoneme in the transfer line)

### Per-cell validation (Phase 10)
- Split identity asserted per dataset (850/150/6164; 3400/600/1000;
  56/11/1029); val_source recorded (stratified 15% of train @ seed 42).
- Feature budget 9996 (4998 G + 4998 H) asserted; no NaN/Inf (asserted
  at features and at X_R5); extractor identity 0.0 on all datasets;
  regimes deterministic per seed (re-extraction equality asserted).
- Classifier: train-only fit -> val diagnostic; train+val refit -> ONE
  test evaluation; alpha recorded per cell.
- Predictions: 27/27 files correct length, finite, >=2 classes
  (no collapse); class balance recorded per cell.

### Leakage audit
NO TEST LEAKAGE. Fallback decision used ONLY train-only-fit validation
Macro-F1 (her_val >= mr_val). Raw-HERAMBA test scores were computed for
audit but never used for selection; no max() over test scores; no
clamping of negative deltas. Wafer class distribution (903/97 train,
5499/665 test) handled by Macro-F1; no test-derived statistics anywhere.

### Configuration audit
HERAMBA R5 FROZEN. One frozen recipe for all 18 cells (RCMKNContextModel
SSL+VQ K=8+joint; config byte-identical to the stored
rcmkn_ssl_context_{transfer,important2}_seed42/config.json);
MiniRocket(random_state=42) frozen; alpha grid logspace(-4,4,20) frozen;
splits frozen. context_train records differ across seeds only in
seed-dependent training-curve values (early-stop/best-val) - keys
identical; no configuration drift.

### Computational cost (measured, from result.json runtimes)
| Dataset          | MR Runtime | HERAMBA Runtime (per seed) | Overhead |
|------------------|-----------:|---------------------------:|---------:|
| Wafer            | MISSING (included in arm) | 75-128 s | ~100x |
| ECG5000_UNBAL    | MISSING (included in arm) | 273-632 s | ~60x   |
| ItalyPowerDemand | MISSING (included in arm) | 4.5-4.9 s | ~1x    |

(MR and HERAMBA runtimes are measured inside the combined per-seed arm;
a separate MR-only runtime was not recorded -> MISSING per protocol.)

Final status: VALID (18/18 cells, no leakage, R5 frozen, artifacts
audited).
