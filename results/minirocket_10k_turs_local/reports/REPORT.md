# MiniROCKET(10K) + TURS Local — Experiment Report

*Runtime 1.3 min · seed 42 · n_kernels=10000*

## 1. Objective

Determine whether TURS Local adds predictive value beyond full-capacity MiniROCKET(10K).

## 2. Implementation

- aeon `MiniRocket(random_state=42)` — default 10,000 kernels, 9,996 features
- TURS Local: FixedPatternBank M=128 + LocalActivity, 1,024 features
- Combined: 9,996 + 1,024 = 11,020 features
- RidgeClassifierCV with `alphas=np.logspace(-4, 4, 20)`
- Train+val combined for final classifier fit (canonical benchmark protocol)
- Per-sample z-normalization

## 3. Four-Dataset Results

| Dataset | MR_10K | MR_10K+Local | Δ | Δ% |
|---|---|---|---|---|
| ECG5000_UNBAL | 0.619 | 0.619 | 0.0 | 0.0% |
| ECG5000_BAL | 0.6613 | 0.6653 | 0.004 | 0.6% |
| CWRU_UNBAL | 0.975 | 0.975 | 0.0 | 0.0% |
| CWRU_BAL | 0.9947 | 0.993 | -0.0017 | -0.17% |
| **MEAN** | **0.8125** | **0.8131** | **0.0006** | **0.2%** |

## 4. Capacity Comparison (Δ at 2K vs 10K)

| Dataset | Δ at 2016 kernels | Δ at 10K kernels |
|---|---|---|
| ECG5000_UNBAL | -0.0345 | 0.0 |
| ECG5000_BAL | 0.0049 | 0.004 |
| CWRU_UNBAL | 0.0 | 0.0 |
| CWRU_BAL | 0.0 | -0.0017 |

## 5. Scientific Interpretation

**Mean Δ = +0.0006: TURS Local provides no measurable gain beyond MiniROCKET(10K)**

- Positive Δ on 0/4 datasets
- Negative Δ on 0/4 datasets
- Neutral on 4/4 datasets

### Per-dataset

- **ECG5000_UNBAL**: Δ=+0.0000 → No meaningful difference
- **ECG5000_BAL**: Δ=+0.0040 → No meaningful difference
- **CWRU_UNBAL**: Δ=+0.0000 → No meaningful difference
- **CWRU_BAL**: Δ=-0.0017 → No meaningful difference

## 6. Final Conclusion

MiniROCKET(10K) + TURS Local is **EQUIVALENT** than MiniROCKET(10K).

Mean Δ = +0.0006

**Does TURS Local retain measurable predictive value when the global representation is full-capacity MiniROCKET?**

**NO.** TURS Local does not provide measurable predictive value beyond full-capacity MiniROCKET(10K). The local pattern bank adds 1,024 features that do not improve held-out Macro-F1 under an otherwise identical protocol.

**Should TURS Local be considered experimentally redundant?**

Based on these four datasets with matched capacity and protocol, **yes** — TURS Local is experimentally redundant when the global representation has full MiniROCKET capacity.
