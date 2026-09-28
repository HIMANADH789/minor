# MiniROCKET Baseline Provenance Report

## 1. Source of 0.5938 / 0.6553 / 0.9917 / 0.9947

**Source file:** `results/baseline_bench/{dataset}.json`
**Producer script:** `experiments/benchmark_baselines.py`

## 2. Exact Configuration (FOUND)

| Item | Value | Status |
|---|---|---|
| Library | aeon `MiniRocket` | FOUND |
| Kernel count | default (10,000) | FOUND |
| Feature count | 9,996 | FOUND |
| Fixed/semi-learned | Semi-learned (fits biases/dilations from training data) | FOUND |
| Bias fitting | Yes (84 biases per kernel) | FOUND |
| Dilation fitting | Yes (from training data) | FOUND |
| Random seed | `random_state=42` | FOUND |
| Feature extraction | PPV only (proportion of positive values) | FOUND |
| Input shape | `[N, 1, T]` (1-channel) | FOUND |
| Input normalization | Per-sample z-normalization | FOUND |
| Dataset split | sklearn `train_test_split(test_size=0.15, stratify=ya, random_state=42)` | FOUND |
| Train/val split | 85% train, 15% val from combined train+val set | FOUND |
| Ridge classifier | `RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))` | FOUND |
| Ridge alpha selection | Internal CV within RidgeClassifierCV | FOUND |
| Final training | Train+val combined (`np.concatenate([Xtr_t, Xva_t])`) | FOUND |
| Seed | 42 | FOUND |
| Multi-seed | No (single seed) | FOUND |

## 3. Reproduction

EXACT REPRODUCTION ACHIEVED:

```
ECG5000_UNBAL: MF1=0.5938 Acc=0.9540 alpha=11.29 feat=9996
ECG5000_BAL:   MF1=0.6553 Acc=0.9520 alpha=1.62  feat=9996
CWRU_UNBAL:    MF1=0.9917 Acc=0.9917 alpha=0.23  feat=9996
CWRU_BAL:      MF1=0.9947 Acc=0.9947 alpha=0.62  feat=9996
```

## 4. Why GLD A0 Differs from Historical Baseline

The GLD A0 used a DIFFERENT configuration:

| Aspect | Historical MiniROCKET | GLD A0 (corrected) |
|---|---|---|
| Kernel count | 10,000 (default) | 2,016 (via MiniRocketGlobal) |
| Feature count | 9,996 | 2,016 |
| Ridge | RidgeClassifierCV (internal CV) | DualRidge (manual lambda grid) |
| Train data for Ridge | Train + Val combined | Train only |
| Alpha search | Internal CV | Validation set selection |

**The GLD A0 used 2,016 kernels instead of 10,000** because `MiniRocketGlobal(n_kernels=2016)` was a deliberate downscaling for the GLR A8 variant.

The kernel count explains MOST of the CWRU discrepancy (0.946 vs 0.992).

## 5. Two Separate Benchmarks

This project has TWO different MiniROCKET baselines:

1. **Historical MiniROCKET** (0.594/0.655/0.992/0.995): aeon default (10K kernels) + RidgeClassifierCV + train+val combined
2. **GLR A8 MiniRocket** (described but not fully validated): aeon 2016 kernels + different Ridge

These are NOT the same benchmark and must not be conflated.
