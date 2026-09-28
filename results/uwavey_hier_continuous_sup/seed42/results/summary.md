# HIER-CONTINUOUS-SUP — summary (UWaveGestureLibraryY, seed 42)

**Macro-F1 = 0.7599** | accuracy 0.7635 | alpha 78.476 | runtime 347 s (production run; deterministic rerun 363 s)

## The one change

HIER-HIGH-SUP protects G at 17,993 (rho = 0.1 of 19,992) and lets
supervised selection act only on the 449,820 delta candidates for the
remaining 1,999 slots. HIER-CONTINUOUS-SUP removes that boundary:

- root candidates: full expanded MiniROCKET matrix, **19,992** (level-0 of
  the hierarchy; PPV_m^(1) = G_m) - no 17,993 preselection
- delta candidates: **449,820** (30 edges x 14,994 kernels, K = 2/4/8/16)
- unified pool C = [G_full || Delta_full] = **469,812** (asserted)
- ONE `top_f_select(C, y_dev, 19,992)` ranking -> exactly 19,992 features
- no rho, no N_G/N_H, no per-level quota - composition EMERGES from the
  ranking

Everything else locked and identical to HIER-HIGH-SUP: canonical split
(761/135/3,582), per-sample z-norm, expanded MiniROCKET
(MiniRocket(random_state=42, n_kernels=19,992) = 84 kernels x 238
quantiles), frozen seed-42 train-only hierarchy (reg16_tr byte-identical
to the stored uwavey_nested artifact), delta-bank definition, canonical
RidgeClassifierCV(logspace(-4, 4, 20)) protocol.

## Comparison (saved references; only this model was run)

| Method | Final feats | Macro-F1 | delta vs this |
|---|---|---|---|
| Canonical MR | 9,996 | 0.7543 | +0.0056 |
| MR-HIGH | 19,992 | 0.7477 | +0.0122 |
| R5-HIGH | 19,992 | 0.7772 | -0.0173 |
| HIER-HIGH | 19,992 | 0.7230 | +0.0369 |
| HIER-HIGH-SUP | 19,992 | 0.7823 | -0.0224 |
| **HIER-CONTINUOUS-SUP** | 19,992 | **0.7599** | - |

MR-HIGH reproduction gate: 0.7477 vs stored 0.7477 (diff 0.0000).
CV (fold-internal, unified pool, diagnostic only): 0.7624 +/- 0.0270.

## Emerged composition (diagnostic only - never fed back)

| level | pool | selected | share of final |
|---|---|---|---|
| root (0) | 19,992 | 14,453 | 72.3% |
| K=2 | 29,988 | 4,012 | 20.1% |
| K=4 | 59,976 | 1,436 | 7.2% |
| K=8 | 119,952 | 65 | 0.3% |
| K=16 | 239,904 | 26 | 0.1% |

The ranking selects 72.3% roots by F-statistic dominance - exactly the
protection the HIER-HIGH-SUP design hard-codes (84.5% at rho = 0.1), but
now as an *empirical outcome* rather than a design constraint.

## Interpretation (per spec section 16)

**Outcome C - unrestricted unified selection is less stable at this
sample size.** Against its strongest controlled comparator
(HIER-HIGH-SUP; same capacity 19,992, same seed, same MiniROCKET, same
hierarchy, same delta construction, same Ridge, same ANOVA-F machinery;
only candidate-pool competition changes), removing the G/H boundary costs
-0.0224 Macro-F1. The mechanism is selection dilution: n = 896 training
rows cannot estimate reliable F-statistics over 469,812 candidates, so
~27.7% of the budget flows to delta carriers that fold-internal CV rates
below the protected-root composition's incumbents (0.7624 +/- 0.0270 vs
HIER-HIGH-SUP's 0.7867 +/- 0.0267). The ranking must re-earn every root
slot from scratch, and that selection noise is not repaid.

Also note vs HIER-HIGH (the energy-allocated design): 0.7599 > 0.7230
(+0.0369) - supervised unified competition still beats the previous
label-free allocator. And vs Canonical MR: +0.0056, roughly parity with
plain MiniROCKET at 2x capacity.

**The evidence supports option (C), not (A)**: the fixed G/H boundary in
HIER-HIGH-SUP is not merely limiting - at this sample size it is
load-bearing. The free-composition result (72.3% roots) closely matches
the imposed allocation (84.5% roots is even more conservative), and the
gap is consistent with selection noise among 469,812 competing
candidates, so the null hypothesis of "unification recovers the same
performance" is not supported either. One dataset, one seed: no
universal-superiority claim is made.

## Leakage / determinism / tests

- MiniROCKET fit on train signals only; hierarchy built from train
  latents only (val/test transform-only); delta pool and root matrix are
  frozen-transform, label-free; final selection uses train+val labels
  (the canonical R5 final-stage protocol, identical to R5-HIGH and
  HIER-HIGH-SUP); CV diagnostic uses fold-train labels only; test labels
  touched only inside ridge_eval for scoring.
- Determinism: two independent production runs produced byte-identical
  results JSONs (only `runtime_s` differs), identical predictions and
  figures.
- Full repository suite: **661 passed, 1 skipped** (the opt-in --smoke
  e2e); focused suite: 19/19 including the smoke path.
