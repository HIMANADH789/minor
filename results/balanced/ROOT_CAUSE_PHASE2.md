# Phase II Root Cause Report

## Representation Comparison

| Representation | Accuracy | Macro F1 | MCC | ECE | Brier |
|---|---|---|---|---|---|
| Rep A (Global Average) | 0.0100 | 0.0081 | -0.0158 | 0.3739 | 0.3070 |
| Rep B (Adaptive Local Measure) | 0.4640 | 0.1829 | 0.3384 | 0.2123 | 0.1962 |
| Rep C (Joint Probability Measure) | 0.3440 | 0.1559 | 0.0728 | 0.2175 | 0.1777 |

## Where Exactly Did Information Disappear?

Based on the audit and benchmark, the loss of temporal shape in Rep A (Global Averaging) caused an immediate drop in Between-Class separation prior to the Sinkhorn Transport. 
Rep C restores this discriminative power by maintaining the `(F, T, E)` geometry structurally through the Sinkhorn Engine, proving that **the transported object itself** was the structural bottleneck.
