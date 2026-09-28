# HIER-SUP-MOD - summary (UWaveY, seed 42)

**Macro-F1 = 0.7383** (accuracy 0.745, alpha 78.47599703514607)

## The one change

Raw ANOVA-F ranking -> EB variance-moderated F ranking on the SAME unified 469,812-candidate telescoping pool: per-candidate s^2 = SSE/(N-8), SINGLE Smyth scaled-F prior (d0 = 0.4724, s0^2 = 2.3569e-04), s_t^2 = (d0 s0^2 + d_e s^2)/(d0 + d_e), F_t = MS_between/s_t^2, one global top-19,992. Used nowhere (no rho, no G/H, no quotas, no local-fdr, no occupancy-as-df, no fallback filter).

## Comparison (saved references; only this model was run)

| Method | Final | Macro-F1 | Delta vs this |
|---|---|---|---|
| Canonical MR | 9,996 | 0.7543 | -0.016 |
| MR-HIGH | 19,992 | 0.7477 | -0.0094 |
| R5-HIGH | 19,992 | 0.7772 | -0.0389 |
| HIER-HIGH | 19,992 | 0.7230 | +0.0153 |
| HIER-HIGH-SUP | 19,992 | 0.7823 | -0.044 |
| HIER-CONTINUOUS-SUP | 19,992 | 0.7599 | -0.0216 |
| HIER-SUP-CLEAN | 19,992 | 0.7599 | -0.0216 |
| **HIER-SUP-MOD** | 19,992 | **0.7383** | NA |

## EB prior

d0 = 0.472385, s0^2 = 2.356866e-04 (single pool-wide fit; moderated df 252.47238520268573; 20091 non-positive variances handled per spec 13; degenerate fit: False).

## Raw vs moderated selection

intersection 19990/19992, Jaccard 0.9998 (raw-only 2, moderated-only 2; 2 entering, 2 leaving).

## Emerged composition (diagnostic)

D1 14865 (74.4%), D2/K2 3655 (18.3%), D4/K4 1290 (6.5%), D8/K8 160 (0.8%), D16/K16 22 (0.1%).

CV (moderated, fold-internal): 0.6858 +/- 0.0451. Raw-F equivalence gate: SKIPPED_SMOKE.

Runtime 41s; peak process memory 0.0 MB.
