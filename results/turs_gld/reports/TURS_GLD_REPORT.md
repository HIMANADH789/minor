# TURS-GLD: Global Bank + Local Lag-Drift — Full Report

*Runtime 4.5 min · seed 42 · M_global=2048 · M_local=128*

## 1. Executive Summary

- **ECG5000_UNBAL**: A0=0.6944 → A3(+local)=0.6599 → A7(+localG4)=0.6726
  - Locality: B1(whole)=0.6726 → B2(local)=0.6726 (Δ=+0.0000)
- **ECG5000_BAL**: A0=0.6502 → A3(+local)=0.6551 → A7(+localG4)=0.6573
  - Locality: B1(whole)=0.6551 → B2(local)=0.6573 (Δ=+0.0022)
- **CWRU_UNBAL**: A0=0.9458 → A3(+local)=0.9458 → A7(+localG4)=0.9458
  - Locality: B1(whole)=0.9458 → B2(local)=0.9458 (Δ=+0.0000)
- **CWRU_BAL**: A0=0.9824 → A3(+local)=0.9824 → A7(+localG4)=0.9824
  - Locality: B1(whole)=0.9824 → B2(local)=0.9824 (Δ=+0.0000)

## 6. B1 vs B2 Locality Experiment

| Dataset | B1 (whole G4) | B2 (local G4) | Δ(B2−B1) |
|---|---|---|---|
| ECG5000_UNBAL | 0.6726 | 0.6726 | +0.0000 |
| ECG5000_BAL | 0.6551 | 0.6573 | +0.0022 |
| CWRU_UNBAL | 0.9458 | 0.9458 | +0.0000 |
| CWRU_BAL | 0.9824 | 0.9824 | +0.0000 |

## 8. A0-A7 Ablation Results (Test Macro-F1)

| Dataset | A0 | A1 | A2 | A3 | A4 | A5 | A6 | A7 |
|---|---|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.6944 | 0.5101 | 0.3896 | 0.6599 | 0.6599 | 0.5244 | 0.6726 | 0.6726 |
| ECG5000_BAL | 0.6502 | 0.4747 | 0.4043 | 0.6551 | 0.6520 | 0.4928 | 0.6551 | 0.6573 |
| CWRU_UNBAL | 0.9458 | 0.9029 | 0.5767 | 0.9458 | 0.9458 | 0.8986 | 0.9458 | 0.9458 |
| CWRU_BAL | 0.9824 | 0.9023 | 0.6582 | 0.9824 | 0.9824 | 0.9077 | 0.9824 | 0.9824 |

## 12. Global vs Local Analysis

| Dataset | Global | Global+Local | Global+Local+WholeG4 | Global+Local+LocalG4 |
|---|---|---|---|---|
| ECG5000_UNBAL | 0.6944 | 0.6599 | 0.6726 | 0.6726 |
| ECG5000_BAL | 0.6502 | 0.6551 | 0.6551 | 0.6573 |
| CWRU_UNBAL | 0.9458 | 0.9458 | 0.9458 | 0.9458 |
| CWRU_BAL | 0.9824 | 0.9824 | 0.9824 | 0.9824 |

## 14. Limitations

- Single seed (42); multi-seed robustness not assessed.
- Local G4 lag features are aggregated over time; finer temporal diagnostics possible.
- CWRU may be inherently less sensitive to drift-based transport features.

## 15. Final Scientific Conclusion

See Section 22 (Final Decision Criteria) in the full implementation spec.

