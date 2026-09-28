# WAFER / ECG5000_UNBAL / ITALYPOWERDEMAND — FINAL 3-SEED TABLES

Canonical seeds: **{42, 43, 44}** · Primary metric: **test Macro-F1** ·
MR is the deterministic canonical control (MiniRocket random_state=42 +
deterministic Ridge) shown in every seed cell.

## Test Macro-F1 (primary)

| Dataset          | Seed | MR Test | HERAMBA Raw Val | Selected  | Fallback | Raw HERAMBA Test | Final HERAMBA Test | Δ vs MR |
|------------------|-----:|--------:|----------------:|-----------|----------|-----------------:|-------------------:|--------:|
| Wafer            |   42 |  0.9962 |          1.0000 | HERAMBA   | NO       |           0.9983 |             0.9983 | +0.0021 |
| Wafer            |   43 |  0.9962 |          1.0000 | HERAMBA   | NO       |           0.9958 |             0.9958 | −0.0004 |
| Wafer            |   44 |  0.9962 |          1.0000 | HERAMBA   | NO       |           0.9979 |             0.9979 | +0.0017 |
| ECG5000_UNBAL    |   42 |  0.5938 |          0.6420 | HERAMBA   | NO       |           0.5846 |             0.5846 | −0.0092 |
| ECG5000_UNBAL    |   43 |  0.5938 |          0.6427 | HERAMBA   | NO       |           0.5828 |             0.5828 | −0.0110 |
| ECG5000_UNBAL    |   44 |  0.5938 |          0.6359 | MiniRocket| YES      |           0.5921 |             0.5938 | +0.0000 |
| ItalyPowerDemand |   42 |  0.9650 |          1.0000 | HERAMBA   | NO       |           0.9592 |             0.9592 | −0.0058 |
| ItalyPowerDemand |   43 |  0.9650 |          1.0000 | HERAMBA   | NO       |           0.9572 |             0.9572 | −0.0078 |
| ItalyPowerDemand |   44 |  0.9650 |          1.0000 | HERAMBA   | NO       |           0.9611 |             0.9611 | −0.0039 |

## MR values table (per-seed cells)

| Dataset          | Model      | Seed 42 | Seed 43 | Seed 44 | Mean ± Std     |
|------------------|------------|--------:|--------:|--------:|----------------|
| Wafer            | MiniRocket |  0.9962 |  0.9962 |  0.9962 | 0.9962 ± 0.0000 |
| Wafer            | HERAMBA R5 |  0.9983 |  0.9958 |  0.9979 | 0.9973 ± 0.0013 |
| ECG5000_UNBAL    | MiniRocket |  0.5938 |  0.5938 |  0.5938 | 0.5938 ± 0.0000 |
| ECG5000_UNBAL    | HERAMBA R5 |  0.5846 |  0.5828 |  0.5938 | 0.5871 ± 0.0059 |
| ItalyPowerDemand | MiniRocket |  0.9650 |  0.9650 |  0.9650 | 0.9650 ± 0.0000 |
| ItalyPowerDemand | HERAMBA R5 |  0.9592 |  0.9572 |  0.9611 | 0.9592 ± 0.0020 |

Raw-HERAMBA means (without the one validation-fallback cell):
Wafer 0.9973 ± 0.0013 · ECG5000_UNBAL 0.5865 ± 0.0049 ·
ItalyPowerDemand 0.9592 ± 0.0020.

## Validation Macro-F1

| Dataset          | Model      | Seed 42 | Seed 43 | Seed 44 |
|------------------|------------|--------:|--------:|--------:|
| Wafer            | MiniRocket |  1.0000 |  1.0000 |  1.0000 |
| Wafer            | HERAMBA R5 |  1.0000 |  1.0000 |  1.0000 |
| ECG5000_UNBAL    | MiniRocket |  0.6405 |  0.6405 |  0.6405 |
| ECG5000_UNBAL    | HERAMBA R5 |  0.6420 |  0.6427 |  0.6359 |
| ItalyPowerDemand | MiniRocket |  1.0000 |  1.0000 |  1.0000 |
| ItalyPowerDemand | HERAMBA R5 |  1.0000 |  1.0000 |  1.0000 |

(Validation and test are kept in separate tables; never mixed.)

## Matched-seed Δ (Final HERAMBA − MR)

| Dataset          | d42     | d43     | d44     | Mean Δ   | Median Δ | Std Δ  |
|------------------|--------:|--------:|--------:|---------:|---------:|-------:|
| Wafer            | +0.0021 | −0.0004 | +0.0017 | +0.0011  | +0.0017  | 0.0013 |
| ECG5000_UNBAL    | −0.0092 | −0.0110 | +0.0000 | −0.0067  | −0.0092  | 0.0060 |
| ItalyPowerDemand | −0.0058 | −0.0078 | −0.0039 | −0.0058  | −0.0058  | 0.0020 |

HERAMBA-selected 8/9 · fallback 1/9 (ECG5000_UNBAL seed 44) · HERAMBA
beats MR 2/9 cells. No significance claims (n = 3).

## Gates

| Gate | Observed | Reference | Tol | Verdict |
|---|---|---|---|---|
| ECG5000_UNBAL MR seed42 | 0.5938 | 0.5938 | 0.0011 | PASS (exact) |
| ECG5000_UNBAL raw-HERAMBA seed42 | 0.5846 | 0.5894 | 0.011 | PASS |
| ItalyPowerDemand MR seed42 | 0.9650 | 0.9650 | 0.0011 | PASS (exact) |
| ItalyPowerDemand raw-HERAMBA seed42 | 0.9592 | 0.9592 | 0.011 | PASS (exact) |
| Wafer | first audited references | — | — | MR 0.9962 / R5 0.9983 (no prior artifact) |
