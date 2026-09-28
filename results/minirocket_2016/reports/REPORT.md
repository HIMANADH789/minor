# MiniROCKET(2016) Baseline Report

*Runtime 0.1 min · seed 42 · n_kernels=2016*

## 1. Implementation

- aeon `MiniRocket(n_kernels=2016, random_state=42)`
- Feature dim: 2,016 (PPV only)
- Same data splits as GLD (experiments.turs_rrmt.data.load_split)
- Per-block TRAIN-fitted standardization (z-score)
- DualRidge with lambda grid [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
- Lambda selected on validation (Macro-F1 primary, NLL tie-break)
- Final fit on TRAIN with frozen lambda

## 2. Four-Dataset Results

| Dataset | MF1 | Acc | WF1 | Lambda | Features | Runtime |
|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.6944 | 0.96 | 0.9575 | 1000.0 | 2016 | 1.03s |
| ECG5000_BAL | 0.6502 | 0.945 | 0.9466 | 100.0 | 2016 | 0.71s |
| CWRU_UNBAL | 0.9458 | 0.9458 | 0.9458 | 1000.0 | 2016 | 0.25s |
| CWRU_BAL | 0.9824 | 0.9824 | 0.9824 | 100.0 | 2016 | 0.49s |

## 3. Comparison with GLD-A3 (MiniROCKET_2016 + TURS Local)

| Dataset | MR_2016 | GLD-A3 (+Local) | Δ | Relative |
|---|---|---|---|---|
| ECG5000_UNBAL | 0.6944 | 0.6599 | -0.0345 | -4.97% |
| ECG5000_BAL | 0.6502 | 0.6551 | 0.0049 | +0.76% |
| CWRU_UNBAL | 0.9458 | 0.9458 | -0.0 | -0.00% |
| CWRU_BAL | 0.9824 | 0.9824 | -0.0 | -0.00% |

**Mean Δ: -0.0074**

## 4. 10K Matched Comparison

GLD-A3 uses M=2016, not M=10K. No M=10K+TURS Local result exists.

| Dataset | MR_10K | MR_10K+Local | Note |
|---|---|---|---|
| ECG5000_UNBAL | 0.5938 | N/A | GLD-A3 uses M=2016 |
| ECG5000_BAL | 0.6553 | N/A | GLD-A3 uses M=2016 |
| CWRU_UNBAL | 0.9917 | N/A | GLD-A3 uses M=2016 |
| CWRU_BAL | 0.9947 | N/A | GLD-A3 uses M=2016 |

## 5. Scientific Interpretation

**Mean Δ = -0.0074: TURS Local slightly degrades MiniROCKET(2016) performance**

### Per-dataset interpretation

- **ECG5000_UNBAL**: Δ=-0.0345 → TURS Local hurts
- **ECG5000_BAL**: Δ=+0.0049 → No meaningful difference
- **CWRU_UNBAL**: Δ=-0.0000 → No meaningful difference
- **CWRU_BAL**: Δ=-0.0000 → No meaningful difference

## 6. Final Conclusion

This baseline answers: does TURS Local add information beyond MiniROCKET(2016)?

- Positive Δ on 0/4 datasets
- Negative Δ on 1/4 datasets
- Neutral on 3/4 datasets
- Mean Δ: -0.0074
