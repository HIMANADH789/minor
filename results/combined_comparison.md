# Cross-Model Comparison: TURS-RRMT vs TURS-Stack vs Baselines

## Test Macro-F1 (head_width=192)

| Dataset | TURS-RRMT A4 (neural) | TURS-RRMT A4+Ridge | TURS-RRMT A0 Ridge | TURS-Stack softvote | TURS-Stack best |
|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.417 | **0.526** | 0.376 | 0.615 | 0.629 |
| ECG5000_BAL | 0.550 | **0.555** | 0.496 | 0.663 | 0.673 |
| CWRU_UNBAL | 0.695 | **0.900** | 0.886 | 0.954 | 0.958 |
| CWRU_BAL | 0.832 | **0.929** | 0.898 | 0.988 | 0.988 |

## A4 vs A3 Significance (paired, head_width=192)

| Dataset | MF1 Delta | McNemar p | Verdict |
|---|---|---|---|
| ECG5000_UNBAL | +0.011 | 0.118 | n.s. |
| ECG5000_BAL | +0.012 | **0.021** | SIGNIFICANT |
| CWRU_UNBAL | +0.097 | **0.016** | SIGNIFICANT |
| CWRU_BAL | +0.052 | **0.006** | HIGHLY SIGNIFICANT |

## Ablation Ladder (Test Macro-F1)

| Dataset | A0 Ridge | A1 Pattern | A2 1-flavor | A3 Uniform | A4 Routed | A5 NoTopK |
|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.376 | 0.387 | 0.379 | 0.407 | 0.417 | 0.499 |
| ECG5000_BAL | 0.496 | 0.462 | 0.536 | 0.538 | 0.550 | 0.538 |
| CWRU_UNBAL | 0.886 | 0.703 | 0.629 | 0.597 | 0.695 | 0.676 |
| CWRU_BAL | 0.898 | 0.795 | 0.784 | 0.780 | 0.832 | 0.803 |

## Key Findings

1. **The classifier head was the bottleneck**: With head_width=192 (up from 64), routing becomes statistically significant on 3/4 datasets (ECG5000_BAL p=0.021, CWRU_UNBAL p=0.016, CWRU_BAL p=0.006).

2. **Ridge on A4 features is the overall winner** on 3/4 datasets, dramatically outperforming the neural head (CWRU_UNBAL: 0.900 vs 0.695). The representation is strong; the neural pipeline doesn't fully exploit it.

3. **TURS-Stack still wins on all 4 datasets**. The Stack's 4-branch fusion with learned combiners remains the strongest architecture.

4. **A6/A7 interventions confirm routing is functional**: Shuffling routing destroys performance (0.21-0.52 vs 0.42-0.83).

5. **The routing is meaningful**: D2 shows all 4 flavors correlate with kurtosis and lag_disagreement (q < 0.05). D4 shows targeted perturbation of high-routing regions causes larger confidence drops.

## Files
- Full comparison: `results/combined_comparison.md`
- RRMT report: `results/turs_rrmt/TURS_RRMT_REPORT.md`
- RRMT results: `results/turs_rrmt/<DATASET>/full_results.json`
- Stack results: `results/turs_stack/<DATASET>/full_results.json`
- Handoff: `TURS_RRMT_HANDOFF.md`
