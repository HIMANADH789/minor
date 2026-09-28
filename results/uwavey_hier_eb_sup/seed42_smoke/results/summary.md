# HIER-EB-SUP - summary (UWaveY, seed 42)

**Macro-F1 = 0.7021** (accuracy 0.71, alpha 29.763514416313132)

## The one change

HIER-CONTINUOUS-SUP ranks the unified 469,812-candidate pool by raw ANOVA-F. HIER-EB-SUP keeps the identical statistic but converts it to exact p-values, then z = Phi^-1(1-p), fits FIVE independent parametric two-groups EB models (pi0*N(0,1) + (1-pi0)*N(mu1,sigma1^2), deterministic EM, fixed standard-normal null), and ranks all candidates by posterior signal probability q = 1 - lfdr in ONE global ranking (ties by candidate_index). Top 19,992 feed the canonical Ridge. No quota, no rho, no q multiplication.

## Comparison (saved references; only this model was run)

| Method | Final | Macro-F1 | Delta vs this |
|---|---|---|---|
| Canonical MR | 9,996 | 0.7543 | -0.0522 |
| MR-HIGH | 19,992 | 0.7477 | -0.0456 |
| R5-HIGH | 19,992 | 0.7772 | -0.0751 |
| HIER-HIGH | 19,992 | 0.7230 | -0.0209 |
| HIER-HIGH-SUP | 19,992 | 0.7823 | -0.0802 |
| HIER-CONTINUOUS-SUP | 19,992 | 0.7599 | -0.0578 |
| **HIER-EB-SUP** | 19,992 | **0.7021** | NA |

## EB parameters (per level)

| level | pi0 | mu1 | sigma1 | EM iters | sum(q) |
|---|---|---|---|---|---|
| 0 | 0.5000 | 11.677 | 2.907 | 11 | 19629.4 |
| 2 | 0.5000 | 6.406 | 3.434 | 10 | 27145.3 |
| 4 | 0.5000 | 4.655 | 3.154 | 11 | 48244.8 |
| 8 | 0.5000 | 2.644 | 3.014 | 12 | 80044.5 |
| 16 | 0.5000 | 1.673 | 3.775 | 13 | 147720.9 |

sum(q) is the estimated effective signal count - DIAGNOSTIC ONLY, never a quota.

## Emerged composition (diagnostic)

root 16180 (80.9%), K2 3812 (19.1%), K4 0 (0.0%), K8 0 (0.0%), K16 0 (0.0%).

CV (fold-internal EB over the unified pool): 0.6731 +/- 0.0456.

MR-HIGH reproduction gate: 0.7095 vs stored 0.7477 (diff 0.0382).

Runtime 36s; peak process memory 0.0 MB.
