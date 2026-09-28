# HIER-SUP-CLEAN - summary (UWaveY, seed 42)

**Macro-F1 = 0.7377** (accuracy 0.745, alpha 78.47599703514607)

## Formulation

One telescoping hierarchy of increments with PPV^(0) = 0: Delta^(1) = PPV^(1) (the root increment, bit-equal), Delta^(K) = PPV^(K) - occupancy-weighted parent PPV for K in {2,4,8,16}. Unified pool [D1 | D2 | D4 | D8 | D16] of 469,812 candidates; ONE supervised ANOVA-F ranking (train+val labels, canonical R5 protocol) picks exactly 19,992 features with a fully emergent composition. No rho, no G/H split, no quota, no energy allocation, no empirical Bayes.

## Comparison (saved references; only this model was run)

| Method | Final | Macro-F1 | Delta vs this |
|---|---|---|---|
| Canonical MR | 9,996 | 0.7543 | -0.0166 |
| MR-HIGH | 19,992 | 0.7477 | -0.01 |
| R5-HIGH | 19,992 | 0.7772 | -0.0395 |
| HIER-HIGH | 19,992 | 0.7230 | +0.0147 |
| HIER-HIGH-SUP | 19,992 | 0.7823 | -0.0446 |
| HIER-CONTINUOUS-SUP | 19,992 | 0.7599 | -0.0222 |
| **HIER-SUP-CLEAN** | 19,992 | **0.7377** | NA |

## Equivalence to HIER-CONTINUOUS-SUP

Verdict: **SKIPPED_SMOKE**. With PPV^(0) = 0 the root block IS Delta^(1), so this experiment is the clean mathematical reformulation / canonicalization of HIER-CONTINUOUS-SUP - not a new method and not a new performance claim. Components: .

## Emerged composition (diagnostic)

D1 14884 (74.4%), D2/K2 3658 (18.3%), D4/K4 1256 (6.3%), D8/K8 164 (0.8%), D16/K16 30 (0.2%).

CV (fold-internal over the unified pool): 0.6850 +/- 0.0446.

MR-HIGH reproduction gate: 0.7095 vs stored 0.7477 (diff 0.0382).

Runtime 35s; peak process memory 0.0 MB.
