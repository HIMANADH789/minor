# HIER-SUP-CLEAN - summary (UWaveY, seed 42)

**Macro-F1 = 0.7599** (accuracy 0.7635, alpha 78.476)

## Formulation

One telescoping hierarchy of increments with PPV^(0) = 0:
Delta^(1)_m = PPV_m^(1) (the root increment, asserted bit-equal),
Delta_(m,c)^(K) = PPV_(m,c)^(K) - occupancy-weighted parent PPV for
K in {2,4,8,16}. Unified pool [D1 | D2 | D4 | D8 | D16] of 469,812
candidates; ONE supervised ANOVA-F ranking (train+val labels, canonical
R5 protocol) picks exactly 19,992 features with a fully emergent
composition. No rho, no G/H split, no quota, no energy allocation,
no empirical Bayes, no moderated-F.

## Result: canonicalization, not a new method

**Equivalence gate vs HIER-CONTINUOUS-SUP: verdict IDENTICAL, F1 delta 0.0.**
Every compared component matches: selected index set AND sequence,
per-candidate ANOVA-F, semantic metadata, emerged composition, test
predictions, alpha, Macro-F1. With PPV^(0) = 0 the "root block" IS
Delta^(1), so this experiment is the clean mathematical reformulation /
canonicalization of HIER-CONTINUOUS-SUP - not a performance improvement and
not a novel model. Per spec section 27: no novelty is manufactured from
notation.

## Comparison (saved references; only this model was run)

| Method | Final | Macro-F1 | Delta vs this |
|---|---|---|---|
| Canonical MR | 9,996 | 0.7543 | -0.0056 |
| MR-HIGH | 19,992 | 0.7477 | -0.0122 |
| R5-HIGH | 19,992 | 0.7772 | +0.0173 |
| HIER-HIGH | 19,992 | 0.7230 | -0.0369 |
| HIER-HIGH-SUP | 19,992 | 0.7823 | +0.0224 |
| HIER-CONTINUOUS-SUP | 19,992 | 0.7599 | 0.0000 |
| **HIER-SUP-CLEAN** | 19,992 | **0.7599** | NA |

## Emerged composition (diagnostic)

D1 14,453 (72.3%), D2/K2 4,012 (20.1%), D4/K4 1,436 (7.2%),
D8/K8 65 (0.3%), D16/K16 26 (0.1%).

CV (fold-internal over the unified pool): 0.7624 +/- 0.0270 (identical to
HIER-CONTINUOUS-SUP). MR-HIGH reproduction gate: 0.7477 vs stored 0.7477
(diff 0.0000). Runtime 290 s / 328 s (two runs); peak process memory
~10.3 GB (working set at the pool stage, externally observed).

## Verification

- Focused tests: 22/22 passed (incl. full-path smoke; equivalence gate
  SKIPPED_SMOKE as designed)
- Full repository suite: 708 passed, 1 skipped (opt-in smoke mark;
  686 prior + 22 new, zero regressions)
- Determinism: two independent production runs - all 13 artifacts
  byte-identical except wall-clock runtime_s (290.0 vs 328.0)
- Leakage: clean (train-only hierarchy/pool; final selection on train+val
  per canonical R5 protocol; test labels only inside ridge_eval)
- One locked-model fix disclosed: chain_from_counts_sums previously filled
  its per-level cnts dict only when want_deltas=True while
  hierarchy_chain reads it unconditionally - the want_deltas=False
  diagnostic path (used only by this experiment's identity check) raised
  KeyError. Fix moves one assignment out of the conditional; all
  delta-path behavior byte-identical (708-test suite green).
