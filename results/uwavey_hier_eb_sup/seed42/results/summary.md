# HIER-EB-SUP - summary (UWaveY, seed 42)

**Macro-F1 = 0.7510** (accuracy 0.7549, alpha 78.476, runtime ~330 s)

## The one change

HIER-CONTINUOUS-SUP ranks the unified 469,812-candidate pool by raw
ANOVA-F. HIER-EB-SUP keeps the identical statistic (bit-identical
f_classif F), converts it to exact p-values, then z = Phi^-1(1-p),
fits FIVE independent parametric two-groups EB models
(pi0*N(0,1) + (1-pi0)*N(mu1,sigma1^2), deterministic EM, predeclared
hyperparameters), and ranks all candidates by posterior signal
probability q = 1 - lfdr in ONE global ranking (ties by
candidate_index). Top 19,992 feed the canonical Ridge. No quota, no
rho, no q multiplication. Model named: parametric two-groups EB
local-FDR - NOT Efron locfdr.

## Comparison (saved references; only this model was run)

| Method | Final | Pool | Macro-F1 | Delta vs this |
|---|---|---|---|---|
| Canonical MR | 9,996 | - | 0.7543 | +0.0033 |
| MR-HIGH | 19,992 | - | 0.7477 | -0.0033 |
| R5-HIGH | 19,992 | 14,994 | 0.7772 | +0.0262 |
| HIER-HIGH | 19,992 | 449,820 | 0.7230 | -0.0280 |
| HIER-HIGH-SUP | 19,992 | 449,820 | 0.7823 | +0.0313 |
| HIER-CONTINUOUS-SUP | 19,992 | 469,812 | 0.7599 | +0.0089 |
| **HIER-EB-SUP** | 19,992 | 469,812 | **0.7510** | NA |

## EB parameters (per level)

| level | pi0 | mu1 | sigma1 | EM iters | sum(q) | selected |
|---|---|---|---|---|---|---|
| 0 | 0.5000 (floor) | 21.32 | 5.14 | 9 | 19,853 | 19,294 |
| 2 | 0.5000 (floor) | 12.32 | 5.91 | 8 | 29,032 | 698 |
| 4 | 0.5000 (floor) | 9.03 | 5.03 | 9 | 55,442 | 0 |
| 8 | 0.5000 (floor) | 5.56 | 3.92 | 10 | 95,235 | 0 |
| 16 | 0.5000 (floor) | 4.94 | 4.30 | 10 | 182,130 | 0 |

## Emerged composition (diagnostic)

root 19,294 (96.5%), K2 698 (3.5%), K4 0, K8 0, K16 0.

## Key diagnosis: the fixed-null two-groups model SATURATES on this pool

Every level's pi0 hit the predeclared 0.5 floor and sum(q) totals
~381,700 of 469,812 candidates. Under the fixed N(0,1) z-null, the
observed F/p distributions across ~470k candidates (strong multiplicity
plus overdispersion of the PPV-based F statistic at n=896) force the
alternative component to absorb most of the mass: mu1 stretches to
~21 (level 0) and lfdr reaches 0 for a huge cohort. Consequently
thousands of candidates share q = 1.0 exactly, the ranking's
discriminative power collapses at the top, and the deterministic
candidate_index tie-break takes over - which, with roots occupying
indices 0..19,991, effectively reverts to "all roots, then top-K2 by
index". The emergent 96.5%-root composition is therefore an artifact
of tie-breaking under saturation, not a posterior-derived allocation.
Under saturation, sum(q) is NOT interpretable as an effective signal
count (recorded for completeness only).

CV (fold-internal EB over the unified pool): 0.7760 +/- 0.0337.
MR-HIGH reproduction gate: 0.7477 vs stored 0.7477 (diff 0.0000).

## Interpretation (per spec 35) - outcome C

EB (0.7510) < raw ANOVA-F ranking (0.7599): the chosen parametric
two-groups EB model does NOT provide useful selection stabilization
here - it is misspecified for this pool scale (fixed N(0,1) null +
~470k multiplicity), saturates pi0/q, and degenerates into index-order
tie-breaking. This is a statement about THIS specified model (fixed
null, parametric Gaussian alternative), not about empirical-Bayes
selection in general: an empirical-null or nonparametric f1 variant
could behave differently (explicitly a separate future experiment per
spec 37). K4/K8/K16 selected = 0 is an emergent diagnostic of this
run only - it does NOT establish that those levels are useless (their
structural significance was established label-free in prior
experiments), and no multi-seed claims are made from this single run.
No universal-superiority claims.

## Leakage / determinism

Hierarchy train-only (reg16 byte-identical to the stored artifact);
pool label-free; EB fitted on train+val statistics only for the final
selection (canonical R5 boundary) and fold-train-only in the CV
diagnostic; test labels touched only inside ridge_eval; EB
hyperparameters predeclared, never tuned. Two independent production
runs: all 19 scientific artifacts byte-identical (only wall-clock
runtime_s differs: 330.6 vs 325.5 s). Peak process memory: ~6.5 GB
working set observed externally at the CV stage (in-process probe
returns 0.0 - documented psapi quirk).
