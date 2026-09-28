# CCWRU FINAL 3-SEED VALIDATION — FINAL REPORT

Date: 2026-09-23. Scope: CWRU_UNBAL and CWRU_BAL, MiniRocket (MR) vs
HERAMBA R5 (rcmkn R2/R5 family — naming confirmed by user). Canonical seed
set {42, 43, 44} (repo standard; cf. drtn_conditioned_minirocket_ecg5000_bal_3seed
and rcmkn_r2_haptics_3seed). No positive result is forced (spec section 22).

## 1. Objective

Final three-seed CCWRU validation of MiniRocket vs HERAMBA R5 under the
frozen canonical protocol, exactly 3 valid seeds per model per dataset.

## 2. Existing experiments discovered (Phase 1 audit)

All pre-existing CWRU artifacts were seed-42 only:

- `results/drtn_conditioned_minirocket_transfer_seed42/CWRU_UNBAL`:
  M0 MiniRocket 0.9917; M1 regime-conditioned 0.9792; explicit
  `"rung": "R5"` DRTN checkpoint (a DIFFERENT R5-flavored line).
- `results/drtn_conditioned_minirocket_transfer_seed42/CWRU_BAL`:
  M0 0.9947; M1 0.9930.
- `results/rcmkn_ssl_context_transfer_seed42/CWRU_UNBAL`:
  R0 0.9792; **R2 0.9917** (canonical rcmkn R2 reference).
- `results/rcmkn_ssl_context_important2_seed42/CWRU_BAL`:
  R0 0.9930; **R2 0.9982** (canonical rcmkn R2 reference).
- No 43/44 seeds existed for any model on either dataset.
- The literal name "CCWRU" does not occur in the repository; the CWRU
  benchmark objects are CWRU_UNBAL / CWRU_BAL. Both were included per user
  selection. HERAMBA R5 = rcmkn R2/R5 line per user selection (the
  drtn-transfer M1 line is a different implementation).

## 3. Seeds reused

- MR seed 42 (both datasets): REUSED via the deterministic canonical
  convention — the runner re-executes the deterministic pipeline and the
  seed-42 gate must reproduce the stored canonical value exactly before any
  other seed is accepted:
  CWRU_UNBAL 0.9917 == 0.9917 (PASS), CWRU_BAL 0.9947 == 0.9947 (PASS).
- HERAMBA R5 seed 42 (both datasets): REUSED as gate reference; the fresh
  seed-42 run (new per-seed context training) is gated against the canonical
  stored R2 within the repo R2 tolerance 0.011:
  CWRU_UNBAL 0.9833 vs 0.9917 (PASS), CWRU_BAL 0.9965 vs 0.9982 (PASS).
  Original seed-42 artifacts untouched.

## 4. Seeds newly executed

- MiniRocket 43/44 (both datasets): deterministic pipeline (identical
  values to seed 42 — expected and verified; MiniRocket extractor fixed at
  random_state=42 and Ridge is deterministic by repo convention).
- HERAMBA R5 43/44 (both datasets): per-seed learned context trained fresh
  per outer seed (the only seed-varying component), all other stages frozen.

## 5. Seeds rerun

None. No existing artifact was invalidated; no rerun was required.

## 6. MiniRocket protocol (actual implementation)

aeon MiniRocket(random_state=42, n_jobs=-1) fitted on z-normed TRAIN only ->
9996 features; RidgeClassifierCV(alphas=logspace(-4,4,20)) fitted on TRAIN
only -> validation Macro-F1 diagnostic; refit on TRAIN+VAL -> ONE official
test evaluation. Deterministic across outer seeds by canonical convention.

## 7. HERAMBA R5 protocol (actual implementation)

Frozen R2 composition X_R2 = [G || H], G = MiniRocket global 4998, H =
occupancy-weighted regime heterogeneity 4998
(H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2, min_occupancy 0.01, valid-region
masking). Learned context = RCMKNContextModel (SSL encoder + VQ K=8 + joint
head), trained PER OUTER SEED via the audited rcmkn_r2_haptics_3seed core
machinery (train_context_model / extract_context_regimes, n_classes=4);
context config byte-identical to the CWRU seed-42 producers
(rcmkn_ssl_context_{transfer,important2}_seed42/config.json) and to the
audited 3-seed core. Regime extraction: argmax VQ assignment, transform
only. Regime determinism within a seed is asserted (AUDIT-8-style
re-extract). Ridge protocol identical to MR arm (same alpha grid, same
train-only -> val / train+val refit -> one test evaluation).

Pipeline: CWRU raw -> per-sample z-norm -> MiniRocket carriers (fixed
rs=42) -> per-seed SSL+VQ context -> hard regime assignment -> global PPV G
-> regime-conditioned heterogeneity H -> [G||H] -> Ridge (LOO alpha on
train-only for val diagnostic; train+val refit) -> single test evaluation.

## 8. Per-seed results (test Macro-F1)

### CWRU_UNBAL

| Model | Seed 42 | Seed 43 | Seed 44 | Mean +- Std | Median | Min | Max |
|---|---|---|---|---|---|---|---|
| MiniRocket | 0.9917 | 0.9917 | 0.9917 | 0.9917 +- 0.0000 | 0.9917 | 0.9917 | 0.9917 |
| HERAMBA R5 | 0.9833 | 0.9792 | 0.9750 | 0.9792 +- 0.0042 | 0.9792 | 0.9750 | 0.9833 |

### CWRU_BAL

| Model | Seed 42 | Seed 43 | Seed 44 | Mean +- Std | Median | Min | Max |
|---|---|---|---|---|---|---|---|
| MiniRocket | 0.9947 | 0.9947 | 0.9947 | 0.9947 +- 0.0000 | 0.9947 | 0.9947 | 0.9947 |
| HERAMBA R5 | 0.9965 | 0.9912 | 0.9947 | 0.9941 +- 0.0027 | 0.9947 | 0.9912 | 0.9965 |

Validation Macro-F1 per seed is recorded in per_run_results.json /
PREDICTION_CHECKS.csv (MR UNBAL 0.9804; R5 UNBAL 0.9902/0.9853/1.0000;
MR BAL 0.9896; R5 BAL 0.9834/0.9834/0.9959). Validation and test are never
mixed in any table.

## 9. Aggregate results

- CWRU_UNBAL: MiniRocket 0.9917 +- 0.0000 (median 0.9917, min=max=0.9917);
  HERAMBA R5 0.9792 +- 0.0042 (median 0.9792, min 0.9750, max 0.9833).
- CWRU_BAL: MiniRocket 0.9947 +- 0.0000; HERAMBA R5 0.9941 +- 0.0027
  (median 0.9947, min 0.9912, max 0.9965).

## 10. Matched-seed comparison (HERAMBA R5 - MiniRocket)

- CWRU_UNBAL: seed42 -0.0084, seed43 -0.0125, seed44 -0.0167;
  mean -0.0125, median -0.0125, std 0.0042; HERAMBA better on 0/3,
  tied 0, MR higher 3/3.
- CWRU_BAL: seed42 +0.0018, seed43 -0.0035, seed44 +0.0000 (exact tie);
  mean -0.0006, median +0.0000, std 0.0027; HERAMBA better 1/3, tied 1/3,
  MR higher 1/3.

No significance claims from three seeds (descriptive only).

## 11. Stability

MR has zero seed variance (deterministic). HERAMBA R5 seed-to-seed std:
0.0042 (UNBAL), 0.0027 (BAL) — small but nonzero, driven solely by the
per-seed learned context. Seed-to-seed spread is comparable to the
repo-established R2 gate tolerance (0.011).

## 12. Computational cost

Per-seed R5 context training + extraction + Ridge: 214-403 s (UNBAL),
300-835 s (BAL) on the RTX 4050 laptop GPU; MR arm seconds per seed.
Total run 11:37:32 -> ~12:20 (about 43 minutes for all 12 arms).

## 13. Audit

- Reused: MR seed-42 canonical values (both datasets; gate-verified);
  R5 seed-42 canonical values as gate references.
- Newly run: MR 43/44 and HERAMBA R5 42*/43/44 per dataset
  (*fresh per-seed-context seed-42 runs, gated; original artifacts
  preserved untouched).
- Rerun: none. Invalid: none. Missing: none at completion.

## 14. Final status

MiniRocket = 3/3 VALID (both datasets)
HERAMBA R5 = 3/3 VALID (both datasets)

All four seed-42 gates PASS; all predictions present, correct length
(240 / 567), finite; split identity asserted (1156/204/240, 2727/482/567);
per-sample z-norm identical; no test-driven tuning; test evaluated exactly
once per (model, seed).

## 15. Honest scientific reading (no forcing)

On CWRU_UNBAL the learned-context HERAMBA R5 consistently trails plain
MiniRocket by 0.8-1.7 pp across all three seeds. On CWRU_BAL the two are
statistically indistinguishable at this seed count (delta mean -0.06 pp,
one win each plus one exact tie). This is the clean record: on CWRU, where
MiniRocket is near ceiling (>= 0.99), the regime-conditioned composition
adds no measurable benefit and can cost up to ~1.7 pp. This contrasts with
Haptics, where HERAMBA exceeded MR on 5/5 seeds — the family's benefit is
dataset-dependent, not universal.
