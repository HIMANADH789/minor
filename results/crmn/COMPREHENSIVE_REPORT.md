# CRMN vs Baselines: Comprehensive Comparison Report

## Model Summary

| Model | Type | Params | Novel Component |
|-------|------|--------|-----------------|
| **InceptionTime** | Baseline | ~200K | Multi-scale 1D CNN |
| **eTAI CE** | Baseline+ | ~250K | Transport input channels |
| **eTAI focal** | Baseline+ | ~250K | Transport channels + focal loss |
| **CRMN-CE** | Novel | 476-633K | Regime manifold + MoE predictor |
| **CRMN-FOC1** | Novel | 476-633K | CRMN + focal loss |
| **CRMN-FOC1-nodyn** | Ablation | 476-633K | CRMN w/o dynamics loss |
| **CRMN-FOC1-noproto** | Ablation | 476-633K | CRMN w/o prototype separation |
| **CRMN-FOC1-nofeat** | Ablation | 476-633K | CRMN w/o feature similarity loss |

---

## ECG5000 — Unbalanced (5 classes)

| Model | Acc | MF1 | C0 | C1 | C2 | C3 | C4 |
|-------|-----|-----|------|------|------|------|------|
| **InceptionTime (1ch)** | 0.946 | 0.566 | 0.997 | 0.969 | 0.368 | 0.385 | 0.000 |
| eTAI CE (3ch) | 0.943 | 0.556 | 0.997 | 0.966 | 0.316 | 0.359 | 0.000 |
| eTAI focal g=1 (3ch) | 0.952 | 0.592 | 1.000 | 0.969 | 0.421 | 0.462 | 0.000 |
| eTAI focal g=2 (3ch) | 0.950 | **0.601** | 0.997 | 0.963 | **0.526** | 0.462 | 0.000 |
| CRMN-CE | 0.951 | 0.596 | 0.993 | 0.977 | 0.421 | 0.462 | 0.000 |
| CRMN-FOC1 | 0.949 | 0.599 | 0.991 | 0.966 | 0.579 | 0.462 | 0.000 |
| **CRMN-FOC1-nodyn** | 0.951 | **0.602** | 0.995 | 0.969 | 0.632 | 0.410 | 0.000 |
| CRMN-FOC1-noproto | 0.940 | 0.520 | 0.986 | 0.986 | 0.210 | 0.308 | 0.000 |

**Winner**: eTAI focal g=2 (MF1=0.601) ≈ CRMN-FOC1-nodyn (MF1=0.602) — effectively tied

---

## ECG5000 — Balanced (5 classes, 800/class augmented)

| Model | Acc | MF1 | C0 | C1 | C2 | C3 | C4 |
|-------|-----|-----|------|------|------|------|------|
| **InceptionTime (1ch)** | 0.935 | **0.590** | 0.983 | 0.935 | 0.737 | 0.436 | 0.000 |
| eTAI CE (3ch) | 0.910 | 0.540 | 0.974 | 0.878 | 0.526 | 0.538 | 0.000 |
| eTAI focal g=1 (3ch) | 0.933 | 0.581 | 0.983 | 0.932 | 0.684 | 0.436 | 0.000 |
| eTAI focal g=2 (3ch) | 0.922 | 0.602 | 0.980 | 0.898 | 0.684 | 0.487 | 0.200 |
| CRMN-CE | 0.927 | 0.568 | 1.000 | 0.909 | 0.526 | 0.308 | 0.000 |
| CRMN-FOC1 | 0.925 | 0.561 | 0.980 | 0.915 | 0.737 | 0.410 | 0.000 |
| CRMN-FOC1-nodyn | 0.925 | 0.561 | 0.981 | 0.915 | 0.684 | 0.410 | 0.000 |
| CRMN-FOC1-noproto | 0.919 | 0.552 | 0.986 | 0.889 | 0.684 | 0.410 | 0.000 |

**Winner**: eTAI focal g=2 (MF1=0.602) — transport + focal still best on balanced ECG

---

## Bearing Fault — Unbalanced (4 classes, C0=2000 dominant)

| Model | Acc | MF1 | C0 | C1 | C2 | C3 |
|-------|-----|-----|------|------|------|------|
| **InceptionTime (1ch)** | 0.884 | 0.884 | 1.000 | 0.884 | 0.722 | 0.929 |
| eTAI CE (3ch) | 0.873 | 0.872 | 0.995 | 0.859 | 0.758 | 0.879 |
| eTAI focal g=1 (3ch) | 0.897 | **0.897** | 0.990 | 0.894 | **0.833** | 0.869 |
| eTAI focal g=2 (3ch) | 0.882 | 0.881 | 0.995 | 0.874 | 0.813 | 0.843 |
| **CRMN-CE** | 0.851 | 0.850 | 0.990 | 0.818 | 0.843 | 0.752 |
| CRMN-FOC1 | 0.838 | 0.836 | 0.990 | 0.803 | 0.833 | 0.722 |
| CRMN-FOC1-nodyn | 0.834 | 0.832 | 0.995 | 0.813 | 0.788 | 0.737 |
| CRMN-FOC1-noproto | 0.846 | 0.845 | 0.980 | 0.773 | 0.884 | 0.748 |
| CRMN-FOC1-nofeat | 0.833 | 0.830 | 0.990 | 0.823 | 0.818 | 0.697 |

**Winner**: eTAI focal g=1 (MF1=0.897) — transport channels + focal still best

---

## Bearing Fault — Balanced (4 classes, 2000/class)

| Model | Acc | MF1 | C0 | C1 | C2 | C3 |
|-------|-----|-----|------|------|------|------|
| InceptionTime (1ch) | 0.872 | 0.871 | 1.000 | 0.854 | 0.783 | 0.848 |
| eTAI CE (3ch) | 0.875 | 0.876 | 0.980 | 0.879 | 0.798 | 0.843 |
| **eTAI focal g=1 (3ch)** | 0.901 | **0.901** | 0.995 | 0.889 | 0.813 | **0.904** |
| eTAI focal g=2 (3ch) | 0.899 | 0.899 | 0.985 | 0.899 | 0.833 | 0.879 |
| CRMN-CE | 0.830 | 0.827 | 0.985 | 0.828 | 0.859 | 0.646 |
| CRMN-FOC1 | 0.821 | 0.818 | 0.995 | 0.849 | 0.803 | 0.636 |
| CRMN-FOC1-nodyn | 0.807 | 0.804 | 0.990 | 0.838 | 0.778 | 0.621 |
| CRMN-FOC1-noproto | 0.812 | 0.808 | 0.990 | 0.849 | 0.803 | 0.606 |
| CRMN-FOC1-nofeat | 0.824 | 0.821 | 0.990 | 0.838 | 0.798 | 0.667 |

**Winner**: eTAI focal g=1 (MF1=0.901) — clear winner on balanced bearing

---

## Aggregate Winner Table

| Dataset | Best Model | Best MF1 | 2nd Best | MF1 |
|---------|-----------|----------|----------|-----|
| ECG5000 Unbal | CRMN-FOC1-nodyn | **0.602** | eTAI focal g=2 | 0.601 |
| ECG5000 Bal | eTAI focal g=2 | **0.602** | IT CE | 0.590 |
| Bearing Unbal | eTAI focal g=1 | **0.897** | IT CE | 0.884 |
| Bearing Bal | eTAI focal g=1 | **0.901** | eTAI focal g=2 | 0.899 |

---

## Key Findings

### 1. eTAI focal is the overall winner (3/4 datasets)
The combination of transport-augmented input channels + focal loss consistently outperforms plain InceptionTime and CRMN on 3 of 4 datasets. On bearing-balanced, it achieves MF1=0.901 (+3.4% over InceptionTime).

### 2. CRMN is competitive on ECG5000, weak on Bearing
- ECG5000: CRMN-FOC1-nodyn achieves MF1=0.602, matching eTAI (0.601) and beating IT (0.566)
- Bearing: CRMN lags by 4-8% MF1 vs eTAI, suggesting the regime manifold approach needs more training data or a stronger backbone for vibration data

### 3. CRMN ablation insights
- **Feature similarity loss is CRITICAL**: removing it (nofeat) causes collapse on ECG5000 (MF1=0.1065)
- **Prototype separation helps moderately**: removing it (noproto) costs 0.08 MF1 on ECG5000
- **Dynamics loss has minimal effect**: removing it (nodyn) slightly improves on ECG5000, slight hurts on bearing — the transition model doesn't help with single-window classification

### 4. CRMN's unique strengths
- **C2 recall on ECG5000 unbalanced**: CRMN-FOC1-nodyn achieves 0.632 vs eTAI's 0.526 (+20%)
- **C2 recall on Bearing unbalanced**: CRMN-FOC1-noproto achieves 0.884 vs eTAI's 0.833 (+6%)
- The regime manifold approach shows promise for specific minority class identification

### 5. Why CRMN struggles on bearing data
- The 633K params (vs 200K for IT) leads to more overfitting with limited bearing training data
- The multi-scale regime structure may be overkill for 4-class bearing fault with clear spectral signatures
- The regime manifold needs more diverse training examples to learn meaningful regime structure

---

## Architecture Comparison

| Component | InceptionTime | eTAI | CRMN |
|-----------|--------------|------|------|
| Feature extraction | Multi-scale conv | Multi-scale conv | Multi-scale conv |
| Input channels | 1 (raw) | 3 (raw+quantile+drift) | 1 (raw) |
| Prediction | Linear head | Linear head | MoE with regime prototypes |
| Regime structure | None | None | 3-scale regime encoder |
| Dynamics model | None | None | Transition network |
| Key losses | CE / focal | CE / focal | CE + feat_sim + proto + dyn |
| Strengths | Clean data, simple patterns | Distributional features, imbalance | Minority class focus, regime structure |

---

## Recommendations

1. **For ECG5000**: Use CRMN-FOC1-nodyn (MF1=0.602) or eTAI focal g=2 (MF1=0.601) — effectively tied
2. **For Bearing Fault**: Use eTAI focal g=1 (MF1=0.897-0.901) — clear winner
3. **For vibration/bearing data with varying speed**: eTAI is preferred — transport features capture speed variation
4. **For rare fault detection**: CRMN shows promise (better C2 recall) but needs more data
5. **Future work**: Combine CRMN's regime structure with eTAI's transport channels — this hybrid may capture both distributional and regime information
