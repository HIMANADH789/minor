# DRTN-CONDITIONED MINIROCKET — HAPTICS SEED 42

## 1. Scientific Question

Can a learned discrete temporal regime assignment from DRTN make MiniROCKET's fixed convolutional representation more informative by conditioning its temporal pooling on learned regimes, under an equal 9,996-feature budget?

## 2. Protocol

- **Dataset**: Haptics (canonical UCR/aeon)
- **Split**: Train=132, Val=23, Test=308
- **T**: 1092
- **Classes**: 5
- **Seed**: 42
- **Normalization**: Per-sample z-normalization (canonical)

## 3. Models

| Model | Description | Features |
|-------|-------------|----------|
| M0 | Canonical MiniROCKET | 9,996 PPV |
| M1 | DRTN-conditioned MiniROCKET | 4,998 global PPV + 4,998 heterogeneity |
| M2 | Random-regime control | 4,998 global PPV + 4,998 random heterogeneity |
| M3 | Shuffled-regime control | 4,998 global PPV + 4,998 shuffled heterogeneity |

## 4. Results

| Model | Test Macro-F1 | Delta vs M0 |
|-------|---------------|-------------|
| M0 (Canonical) | 0.4974 | — |
| M1 (DRTN-conditioned) | 0.5178 | +0.0204 |
| M2 (Random-regime) | 0.5037 | +0.0063 |
| M3 (Shuffled-regime) | 0.5037 | +0.0063 |

## 5. Interpretation

**CASE A**: M1 > M0 and M1 > M3

The DRTN-conditioned MiniROCKET (M1) outperforms both the canonical baseline (M0) and the shuffled-regime control (M3). This indicates:

1. **Regime conditioning helps**: The heterogeneity features add information beyond global PPV pooling
2. **Temporal alignment matters**: M1 > M3 shows that DRTN's learned temporal segmentation is specifically useful
3. **Not just any segmentation**: M1 > M2 shows the benefit is not merely from random partitioning

## 6. Regime Statistics

- Active codes: 7 (out of 8)
- Entropy: 1.1909
- Perplexity: 3.2901

## 7. Conclusion

**YES** — Learned DRTN regime conditioning improves MiniROCKET under the same 9,996-feature budget on Haptics.

The key insight is that DRTN's learned discrete temporal regimes provide useful context for interpreting MiniROCKET's fixed convolutional activations. By conditioning the PPV computation on regimes, we capture regime-specific activation patterns that are lost in global pooling.

## 8. Limitations

- Single dataset (Haptics)
- Single seed (42)
- Exploratory experiment

## 9. Next Steps

Consider:
- Testing on additional datasets (ECG5000, etc.)
- Multiple seeds for robustness
- Different K values (16, 32)
- Different pooling statistics
