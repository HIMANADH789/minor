# HIER-CONTINUOUS-SUP - summary (UWaveY, seed 42)

**Macro-F1 = 0.7377** (accuracy 0.745, alpha 78.47599703514607)

## The one change

HIER-HIGH-SUP protects G at 17,993 (rho=0.1) and supervises only the 449,820-delta pool for 1,999 H slots. HIER-CONTINUOUS-SUP removes that boundary: the full 19,992 root matrix (level 0 of the same hierarchy) joins the 449,820 delta candidates in ONE pool of 469,812, and a single ANOVA-F ranking (train+val labels, canonical R5 protocol) picks exactly 19,992 features.

## Comparison (saved references; only this model was run)

| Method | Final | Macro-F1 | Delta vs this |
|---|---|---|---|
| Canonical MR | 9,996 | 0.7543 | -0.0166 |
| MR-HIGH | 19,992 | 0.7477 | -0.01 |
| R5-HIGH | 19,992 | 0.7772 | -0.0395 |
| HIER-HIGH | 19,992 | 0.7230 | +0.0147 |
| HIER-HIGH-SUP | 19,992 | 0.7823 | -0.0446 |
| **HIER-CONTINUOUS-SUP** | 19,992 | **0.7377** | NA |

## Emerged composition (diagnostic)

root 14884 (74.4%), K2 3658 (18.3%), K4 1256 (6.3%), K8 164 (0.8%), K16 30 (0.2%).

CV (fold-internal over the unified pool): 0.6850 +/- 0.0446.

MR-HIGH reproduction gate: 0.7095 vs stored 0.7477 (diff 0.0382).

Runtime 35s; peak process memory 0.0 MB.
