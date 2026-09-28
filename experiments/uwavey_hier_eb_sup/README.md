# HIER-EB-SUP

UWaveGestureLibraryY · seed 42 · final budget **19,992** · deterministic.

## Scientific question

The previous controlled experiment (HIER-CONTINUOUS-SUP) showed that unified
supervised selection over the full 469,812-candidate pool underperforms the
protected G/H composition of HIER-HIGH-SUP (0.7599 vs 0.7823), with the
diagnosis that raw ANOVA-F ranking over ~470k candidates is unstable at
n=896. This experiment tests whether an **empirical-Bayes two-groups
local-FDR selector** stabilizes that unified selection and improves over
raw ANOVA-F.

## One change (controlled ablation)

```
HIER-CONTINUOUS-SUP:  F = f_classif(C) -> top-19,992 by F
HIER-EB-SUP:          F = f_classif(C) -> p -> z = Phi^{-1}(1-p)
                      -> 5 independent per-level two-groups EB fits
                      -> lfdr -> q = 1-lfdr -> top-19,992 by q
```

Everything else is byte-locked to the accepted experiments: canonical
761/135/3,582 split, per-sample z-norm, aeon MiniRocket n_kernels=19,992
(84×238), frozen seed-42 SSL hierarchy (K=1→2→4→8→16, train-only fit,
transform-only val/test), occupancy-weighted telescoping delta bank
(449,820 candidates), unified pool [root | K2 | K4 | K8 | K16] = 469,812,
canonical `RidgeClassifierCV` protocol, fold-internal CV diagnostic.

The EB model is the **parametric two-groups model**
`f(z) = pi0*N(0,1) + (1-pi0)*N(mu1, sigma1^2)` fitted by deterministic EM
per level (fixed N(0,1) null; alternative constrained mu1>0). It is *not*
Efron's nonparametric locfdr. `sum(q)` per level is reported as an estimated
effective signal count — diagnostic only, never a quota. Features are fed to
Ridge as raw values (no q multiplication).

## Run

```bash
python experiments/uwavey_hier_eb_sup/runner.py            # production
python experiments/uwavey_hier_eb_sup/runner.py --smoke    # tiny deterministic smoke
python -m pytest tests/test_uwavey_hier_eb_sup.py          # focused suite (22 tests)
```

## Artifacts

`results/uwavey_hier_eb_sup/seed42/` — `results/` (json, comparison.csv,
summary.md), `diagnostics/` (EB parameters/sanity per level, selection
metadata with F/p/z/lfdr/q per selected feature, leakage report,
selection_protocol.md), `figures/` (per-level EB density histograms,
composition, comparison), `predictions/test_predictions.npy`.

Prior experiment artifacts are read-only references (test-enforced).
