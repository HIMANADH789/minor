# R2 on GunPoint (seed 42)

First run of the R2 architecture (SSL-learned causal temporal encoder + HardVQ regime-conditioned MiniROCKET heterogeneity) on GunPoint, the ceiling-saturated dataset from the context-dependence screen. No architecture or hyperparameter changes; every component imported unchanged from the audited implementations.

## 1. Objective

Establish R2 reference values on GunPoint and test whether the SSL temporal-context mechanism behaves as on Haptics. Prior context3 screen: MiniRocket M0 0.9933 and all conditioned variants 1.0000 (ceiling-saturated, verdict NEUTRAL).

## 2. Configuration (unchanged from Haptics R2)

- MiniRocket: `aeon MiniRocket(random_state=42)`, 9,996 features, fit on z-normed TRAIN rows.
- SSL encoder: 4 causal dilated Conv1d blocks (k=3/5/7/9, d=1/2/4/8, ch=32/32/64/64), d=32 projection; masked-span SSL (10%, span 16); epochs<=120, patience 20.
- HardVQ: K=8, EMA 0.99, beta=0.25, lam_div=0.01, revival patience 100; joint fine-tune epochs<=60, lambda_cls=0.10.
- Heterogeneity: audited valid-region H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2.
- Ridge: `RidgeClassifierCV(alphas=logspace(-4,4,20))`, fit on train+val; test touched exactly once per variant.

## 3. Data

- Source: `data\kaggle\GunPoint` (verified Kaggle UCRArchive_2018 copy, bit-identical to canonical aeon data).
- Split: train 42 / val 8 / test 150, T=150, 2 classes (stratified_15pct_of_train_seed42) -- the exact split the context3 M0 reference used.
- Label map: {'1': 0, '2': 1}.
- File SHA-256 recorded in result.json provenance.

## 4. Audit results

| Audit | Result |
|---|---|
| 1. config | train=42 val=8 test=150 PASS |
| 2. extractor identity | max|diff| = 0.00e+00 |
| 3. budget | 9996 = 4998 + 4998 PASS |
| 4. causality | max|diff| = 0.0e+00 |
| 5. regime determinism | PASS |
| 6. VQ diagnostics | active=4/8, norm-entropy=0.571 |
| 7/8. controls | occupancy failures = 0, distinct arrays PASS |
| 9. independent H recompute | max diff = 1.05e-08 |
| 10. valid region | 177312 out-of-mask flips -> H unchanged PASS |
| M0 gate | 0.9933 vs 0.9933 (tol 0.0011) PASS |

## 5. Test results (Macro-F1, seed 42, one pass)

| Variant | Val Macro-F1 | Test Macro-F1 | Alpha |
|---|---|---|---|
| M0 | 1.0000 | 0.9933 | 0.0001 |
| R2 | 1.0000 | 1.0000 | 0.0001 |
| C1 | 1.0000 | 1.0000 | 0.0001 |
| C2 | 1.0000 | 1.0000 | 0.0001 |

## 6. Deltas

- R2-M0: +0.0067
- R2-C1: +0.0000
- R2-C2: +0.0000
- C1-C2: +0.0000

## 7. Mechanistic heterogeneity stats (train+val)

| Variant | mean H | median H | max H | frac nonzero |
|---|---|---|---|---|
| R2 | 0.03556 | 0.02315 | 0.24982 | 0.814 |
| C1 | 0.00426 | 0.00251 | 0.27778 | 0.853 |
| C2 | 0.00445 | 0.00266 | 0.25000 | 0.853 |

## 8. Context

- Haptics R2 reference: 0.55 (seed 42, canonical).
- Context3 GunPoint: M0 0.9933, M1/M2/M3/A_SOFT 1.0000 (ceiling-saturated).

## 9. Verdict

**UNINFORMATIVE (ceiling-saturated): M0 and R2 both solve the task; no headroom to assess the heterogeneity effect**

## 10. Limitations

- Single seed (42), single split; GunPoint is small (42/8 train+val rows) and ceiling-saturated, so differences between variants are not meaningful for the mechanism.
- The M0 gate ties this run to the context3 canonical reference; the R2 values are the first stored references for this dataset.

## 11. Reproducibility

```bash
cd ECG_Benchmark
python -m experiments.rcmkn_r2_gunpoint_seed42.runner [--smoke]
python -m pytest tests/test_rcmkn_r2_gunpoint_seed42.py -q
```

