# Wafer / ECG5000_UNBAL / ItalyPowerDemand — FINAL REPORT
(MiniRocket vs HERAMBA R5, 3 seeds) · Date: 2026-09-23 · **18/18 cells VALID**

## 1. Objective & scope
Final three-seed MR vs HERAMBA R5 benchmark on Wafer, ECG5000
(user-selected canonical variant **ECG5000_UNBAL**), and
ItalyPowerDemand. Canonical seed set **{42, 43, 44}** (repository
standard). HERAMBA R5 frozen; MiniRocket frozen; no test-driven tuning;
existing canonical directories untouched.

## 2. Existing experiments discovered (audit-before-execution)
- **Wafer**: no MR or R2/R5 artifact anywhere in the R2/rcmkn line —
  all 6 cells genuinely MISSING. Canonical UCR TSVs verified in the
  repo's UCRArchive_2018 copy (1000/6164, T=152, classes {−1,+1}).
- **ECG5000**: two benchmark objects exist; the user selected
  **ECG5000_UNBAL** (rcmkn transfer line, split 3400/600/1000 from
  `ecg5000_resplit.npz`, T=140, 5 classes). Canonical references: M0 =
  0.5938 (corrected_negative_retest), R2 = 0.5894
  (rcmkn_ssl_context_transfer_seed42).
- **ItalyPowerDemand**: canonical R2-line artifact exists
  (rcmkn_r2_kaggle_context2_seed42): M0 = 0.9650, R2 = 0.9592, split
  56/11/1029 (stratified 15% of train @ seed 42), T=24, 2 classes.
  Repo TSVs sha-identical to the archive copy.

## 3–7. Seeds reused / newly executed / reruns
- **MR reused**: none as stored per-seed artifacts — MR is
  deterministic in this repo (MiniRocket random_state=42 +
  deterministic Ridge); the canonical stored M0 values serve as gates
  and are reproduced **exactly** (ECG5000_UNBAL 0.5938 = reference;
  IPD 0.9650 = reference).
- **HERAMBA reused**: canonical stored seed-42 R2 artifacts (ECG5000_UNBAL
  0.5894, IPD 0.9592) kept untouched as gate references; fresh runs
  reproduce them within the repo's 0.011 R2-gate tolerance (IPD exact;
  ECG5000_UNBAL 0.5846, within tolerance).
- **Newly executed**: all 18 cells via the unified runner (3 datasets ×
  3 seeds × 2 arms) — Wafer had no prior artifacts; ECG5000/IPD per-seed
  HERAMBA contexts did not exist for any seed.
- **Reruns**: none — no stored artifact was invalid. (Wafer's 3 seeds
  from the first launch were RESUMED from disk, not re-executed, after a
  loader-key fix; the interruption happened after Wafer had fully
  completed.)

## 8. Final 18-cell status matrix

| Dataset          | Model      | 42 | 43 | 44 | Final |
|------------------|------------|----|----|----|-------|
| Wafer            | MiniRocket | NEW_RUN | NEW_RUN | NEW_RUN | 3/3 VALID |
| Wafer            | HERAMBA R5 | NEW_RUN | NEW_RUN | NEW_RUN | 3/3 VALID |
| ECG5000_UNBAL    | MiniRocket | NEW_RUN_GATE_PASS | GATE_PASS | GATE_PASS | 3/3 VALID |
| ECG5000_UNBAL    | HERAMBA R5 | NEW_RUN_GATE_PASS | NEW_RUN | NEW_RUN | 3/3 VALID |
| ItalyPowerDemand | MiniRocket | NEW_RUN_GATE_PASS | GATE_PASS | GATE_PASS | 3/3 VALID |
| ItalyPowerDemand | HERAMBA R5 | NEW_RUN_GATE_PASS | NEW_RUN | NEW_RUN | 3/3 VALID |

(GATE_PASS = deterministic MR cell equals the canonical gate value.)

## 9. Per-seed MR vs HERAMBA table (test Macro-F1)

| Dataset | Seed | MR Val | HER Raw Val | Selected | Fallback | MR Test | Raw HER Test | Final HER Test | Δ |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| Wafer | 42 | 1.0000 | 1.0000 | HERAMBA | NO | 0.9962 | 0.9983 | 0.9983 | +0.0021 |
| Wafer | 43 | 1.0000 | 1.0000 | HERAMBA | NO | 0.9962 | 0.9958 | 0.9958 | −0.0004 |
| Wafer | 44 | 1.0000 | 1.0000 | HERAMBA | NO | 0.9962 | 0.9979 | 0.9979 | +0.0017 |
| ECG5000_UNBAL | 42 | 0.6405 | 0.6420 | HERAMBA | NO | 0.5938 | 0.5846 | 0.5846 | −0.0092 |
| ECG5000_UNBAL | 43 | 0.6405 | 0.6427 | HERAMBA | NO | 0.5938 | 0.5828 | 0.5828 | −0.0110 |
| ECG5000_UNBAL | 44 | 0.6405 | 0.6359 | MiniRocket | YES | 0.5938 | 0.5921 | 0.5938 | +0.0000 |
| ItalyPowerDemand | 42 | 1.0000 | 1.0000 | HERAMBA | NO | 0.9650 | 0.9592 | 0.9592 | −0.0058 |
| ItalyPowerDemand | 43 | 1.0000 | 1.0000 | HERAMBA | NO | 0.9650 | 0.9572 | 0.9572 | −0.0078 |
| ItalyPowerDemand | 44 | 1.0000 | 1.0000 | HERAMBA | NO | 0.9650 | 0.9611 | 0.9611 | −0.0039 |

Full tables (incl. validation-only table, MR values table, α, accuracy):
FINAL_TABLE.md.

## 10–12. Wafer
MR **0.9962 ± 0.0000** · raw HERAMBA **0.9973 ± 0.0013** ·
final fallback-protected **0.9973 ± 0.0013** (fallback never fired).

## 13–15. ECG5000_UNBAL
MR **0.5938 ± 0.0000** · raw HERAMBA **0.5865 ± 0.0049** ·
final fallback-protected **0.5871 ± 0.0059**.

## 16–18. ItalyPowerDemand
MR **0.9650 ± 0.0000** · raw HERAMBA **0.9592 ± 0.0020** ·
final fallback-protected **0.9592 ± 0.0020** (fallback never fired).

## 19–20. Per-seed and per-dataset Δ
| Dataset | d42 | d43 | d44 | Mean Δ | Median | Std |
|---|---:|---:|---:|---:|---:|---:|
| Wafer | +0.0021 | −0.0004 | +0.0017 | **+0.0011** | +0.0017 | 0.0013 |
| ECG5000_UNBAL | −0.0092 | −0.0110 | +0.0000 | **−0.0067** | −0.0092 | 0.0060 |
| ItalyPowerDemand | −0.0058 | −0.0078 | −0.0039 | **−0.0058** | −0.0058 | 0.0020 |

## 21–22. Selection counts
HERAMBA-selected **8/9** · fallback **1/9** (ECG5000_UNBAL seed 44,
where HERAMBA's train-only validation 0.6359 fell below MR's 0.6405 →
the identical MR model was frozen before its single test evaluation).
HERAMBA wins 2/9 cells; deltas reported as-is, never clamped.

## 23. Stability
| Dataset | MR std | Raw-HER std | Final-HER std | Δstd (HER−MR) |
|---|---:|---:|---:|---:|
| Wafer | 0.0000 | 0.0013 | 0.0013 | +0.0013 |
| ECG5000_UNBAL | 0.0000 | 0.0049 | 0.0059 | +0.0059 |
| ItalyPowerDemand | 0.0000 | 0.0020 | 0.0020 | +0.0020 |

MR's zero std is a protocol property (deterministic control), not a
stability claim. HERAMBA's seed noise is modest; the largest is
ECG5000_UNBAL (std 0.0049 raw), consistent with its 5-class
minority-class difficulty.

## 24. Wafer audit (Phase 11a)
Class distribution: TRAIN 903/97, TEST 5499/665 (≈10:1 imbalance) —
handled by Macro-F1 (zero_division=0); no resampling invented. Split =
canonical TSVs + stratified 15% of train (850/150/6164), asserted.
Near-ceiling behaviour (0.996–0.998) — differences not overinterpreted;
no protocol change made to manufacture a gain. First audited references
recorded: MR 0.9962, HERAMBA 0.9983.

## 25. ECG5000_UNBAL audit (Phase 11b)
Seed-to-seed raw HERAMBA 0.5846/0.5828/0.5921 (std 0.0049);
validation 0.6420/0.6427/0.6359; fallback decision at seed 44 was the
protocol's correct validation-stage behavior, not a test-based choice.
Selected α: MR 11.2884 (canonical), HERAMBA 4.2813 on all seeds
(11.2884 for the fallback cell since it IS the MR model). Regime
assignments deterministic per seed (re-extraction identity asserted).
This analysis involves HERAMBA R5 only — no HIER-SUP-MOD / HIER-2
conclusions are transferred. Minority-class limitation (class-5 F1 = 0
in the canonical record) is a data property, not a protocol artifact.

## 26. ItalyPowerDemand audit (Phase 11c)
Short-T=24 dataset: 9996 MiniRocket features preserved (T=24 → truncated
kernel bank still yields the canonical 9996-dim budget; extractor
identity 0.0 asserted). HERAMBA raw 0.9592/0.9572/0.9611 — very stable
(std 0.0020). Validation is saturated (1.0) at all seeds, so the
validation tie selects HERAMBA by the pre-registered ≥ rule; the
resulting small negative deltas (−0.0058/−0.0078/−0.0039) are reported
honestly and match the canonical seed-42 record's own R2−M0 = −0.0058.
α identical across seeds (4.2813 HER / 11.2884 MR).

## 27. Missing/invalid artifacts
NONE missing; NONE invalid. 27/27 prediction files validated
(correct length 6164/1000/1029, finite, ≥2 classes predicted in every
cell — no collapse; per-cell class balance recorded in result.json).

## 28. Artifact paths
`results/wafer_ecg5000_ipd_r5/` — FINAL_HERAMBA_R5_RESULTS.csv,
HERAMBA_R5_PER_SEED.csv, HERAMBA_R5_AGGREGATED.csv, MR_VS_HERAMBA.csv,
FALLBACK_AUDIT.csv, STATUS_MATRIX.csv, CONFIG_AUDIT.csv,
PREDICTION_CHECKS.csv, FINAL_TABLE.md, FINAL_REPORT.md, AUDIT.md,
per-dataset seed42/43/44 result.json + 3 prediction files each ·
`experiments/wafer_ecg5000_ipd_r5/` — runner.py, aggregate.py ·
logs: wafer_ecg5000_ipd_r5.log, wafer_ecg5000_ipd_r5_resume.log.

## 29. Final audit status
Gates 4/4 PASS (MR exact on both gated datasets; R5 within repo
tolerance; Wafer = first references). Configuration audit: single
frozen R2 recipe; `context_train` keys identical across seeds with only
seed-dependent training-curve values — no drift. Leakage: none (fallback
uses train-only-fit validation only; raw-HERAMBA test scores recorded
but never used for selection; no max() over test). Existing artifacts
preserved.

**VALID** — 18/18 cells valid, no test leakage, R5 frozen, all artifacts
audited. No significance claims from n = 3.
