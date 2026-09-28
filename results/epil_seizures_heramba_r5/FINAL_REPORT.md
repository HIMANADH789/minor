# EpilepticSeizures — HERAMBA R5 — FINAL REPORT
Date: 2026-09-23. CWRU PAUSED and preserved (all prior artifacts intact).

## 1. Objective
Complete the existing HERAMBA R5 (rcmkn R2 family) experiment for
EpilepticSeizures to exactly 3 valid seeds ({42,43,44}), reusing valid
existing seeds and running only what is missing. No redesign.

## 2. Existing experiments discovered
- `results/rcmkn_ssl_context_transfer_seed42/EpilepticSeizures/result.json`:
  canonical seed-42 run — **R2 = 0.9457** test Macro-F1 (val 1.0, accuracy
  0.9650, alpha 1.6238, feature_dim 9996), plus R0 0.9399 / C1 0.9414 /
  C2 0.9409 references, context checkpoint `context_model_seed42.pt`,
  predictions. Split 80/20/11420 (PROVIDED canonical val.ts, never
  resampled), T=178, 2 classes. Context config byte-identical to the
  frozen rcmkn recipe.
- No EpilepticSeizures artifacts anywhere else (no drtn-line ES row; no
  baseline_bench ES file). Seeds 43/44 did not exist for any model.
- Canonical seed set: {42, 43, 44} (repository standard).

## 3. Status matrix (Phase 2)
| Seed | Existing? | Valid? | Action |
|---:|---|---|---|
| 42 | yes (canonical R2 0.9457) | yes | REUSE as gate reference; fresh gated run |
| 43 | no | — | RUN |
| 44 | no | — | RUN |

## 4. Seeds reused / newly executed / reruns
- Reused: seed-42 canonical R2 = 0.9457 as the gate reference (artifact
  untouched).
- Newly executed: seeds 42 (fresh per-seed context, gate-compared),
  43, 44 — each trains the RCMKN context per outer seed (the only
  seed-varying component); carriers fixed at random_state=42.
- Reruns: none. Nothing was invalid.

## 5. Frozen R5 protocol (actual implementation)
- Carriers: aeon MiniRocket(random_state=42) fit on z-normed TRAIN only ->
  9996 features (G 4998 global + H 4998 het); raw-extractor identity audit
  max|diff| = 0.0.
- Context: RCMKNContextModel (SSL encoder + VQ K=8 + joint head),
  n_classes=2, trained per outer seed via the audited
  rcmkn_r2_haptics_3seed core machinery (train_context_model /
  extract_context_regimes); config byte-identical to the seed-42 producer.
  Regime determinism asserted per seed (re-extraction equality).
- H: occupancy-weighted regime heterogeneity
  (H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2, min_occupancy 0.01,
  valid-region masks); no NaN (asserted).
- Classifier: RidgeClassifierCV(alphas=logspace(-4,4,20)); train-only fit
  -> val diagnostic; train+val refit -> ONE test evaluation.
- Leakage: no test labels used in any training/selection stage; test
  touched exactly once per seed.

## 6. Per-seed results (test Macro-F1)
| Model | Seed | Validation | Test | Alpha | Runtime | Status |
|---|---:|---:|---:|---:|---:|---|
| HERAMBA R5 | 42 | 1.0000 | 0.9461 | 1.6238 | 60.9 s | NEW_RUN_GATE_PASS |
| HERAMBA R5 | 43 | 1.0000 | 0.9404 | 0.6158 | 50.2 s | NEW_RUN |
| HERAMBA R5 | 44 | 1.0000 | 0.9444 | 0.6158 | 55.1 s | NEW_RUN |

Gate: 0.9461 vs canonical 0.9457 (tol 0.011) -> PASS.

## 7. Aggregate
Mean 0.9436 +- 0.0029 (sample std, ddof=1); median 0.9444; min 0.9404;
max 0.9461. Descriptive only (n=3); no significance claim.

Prediction sanity: 11,420 predictions per seed, all finite, both classes
predicted; pred class balance (0: ~2,340 / 1: ~9,080) tracks the true
distribution (2,260 / 9,160) — no collapse.

## 8. Final status
HERAMBA R5 = 3/3 VALID (EpilepticSeizures).

## 9. Artifacts
results/epil_seizures_heramba_r5/
  per_run_results.json, seed{42,43,44}/heramba_r5_result.json +
  heramba_r5_predictions.npy, PREDICTION_CHECKS.csv, PER_SEED_RESULTS.csv,
  AGGREGATED_RESULTS.csv, STATUS_MATRIX.csv, CONFIG_AUDIT.csv,
  FINAL_TABLE.md, FINAL_REPORT.md, AUDIT.md
experiments/epil_seizures_heramba_r5/ (runner.py, aggregate.py)
Prior artifacts (rcmkn_ssl_context_transfer_seed42/EpilepticSeizures/)
preserved untouched.
