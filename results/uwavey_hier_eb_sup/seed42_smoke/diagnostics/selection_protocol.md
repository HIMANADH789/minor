# Selection protocol - HIER-EB-SUP

## Unified pool (locked, identical to HIER-CONTINUOUS-SUP)

C = [G_full || Delta_full] with
- G_full: 19992 root candidates (level 0 of the hierarchy)
- Delta_full: 449820 hierarchical telescoping residuals (30 edges x 14994 kernels; K=2,4,8,16)
- total: 469812 candidates, canonical level-grouped layout [root | K2 | K4 | K8 | K16] (layout-guarded against col_meta)

## Statistic (bit-identical to the incumbent)

F, p = sklearn f_classif(C, y) - the same call top_f_select makes, so the F statistic and its exact p-values (same df, same class handling) are identical to the HIER-CONTINUOUS-SUP selector's statistic. Edge cases: non-finite F/p (constant features) -> F=0, p=1; p clipped to [1e-300, 1] before inversion; z = norm.isf(p) is a one-sided significance coordinate (NOT an effect direction), finite in [-8.22, 37.0].

## The empirical-Bayes model (the ONE change)

Per level l in {0, 2, 4, 8, 16} - FIVE independent fits, never pooled:

    f_l(z) = pi0_l * N(0,1) + (1 - pi0_l) * N(mu1_l, sigma1_l^2),  mu1_l > 0

fitted by deterministic EM (init pi0=0.90; mu1/sigma1 from the highest 10% z with degenerate fallback max(mean(z),0.5)/max(std(z),0.5); tol 1e-8 on max parameter change; 200 iterations max; clamps pi0 in [0.5, 1-1e-8], sigma1 >= 1e-3, mu1 >= 1e-3). lfdr = P(null | z) under the fitted mixture; q = 1 - lfdr. This is the parametric two-groups EB local-FDR model - NOT Efron's nonparametric locfdr.

## Stage 1 - CV diagnostic (never selects)

5-fold StratifiedKFold(shuffle=True, random_state=42) over train+val (896 rows). Per fold: f_classif on fold-train rows/labels ONLY -> z -> five fold-train EB fits -> q -> one global top-19,992 -> RidgeClassifierCV(ALPHAS) on fold-train -> macro-F1 on the fold. Fold-validation labels never touch the EB fit or the selection.

## Stage 2 - final selection (canonical R5 protocol)

EB scoring on the FULL dev matrix (train+val labels) - the same supervised boundary R5-HIGH, HIER-HIGH-SUP and HIER-CONTINUOUS-SUP use for their final selection. ONE global ranking by q (ties by candidate_index ascending), top 19,992. NO separate G stage, no rho, no N_G/N_H, no per-level quota, no q multiplication of features.

## Stage 3 - refit + one official test evaluation

RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) fit on the selected 19992-column representation over train+val; single evaluation on the official test split.

## Resulting composition (emerged, diagnostic only)

root 16180, K2 3812, K4 0, K8 0, K16 0 (sum 19992 = 19992). Estimated effective signal counts sum(q) per level are in eb_parameters_by_level.csv and are NOT quotas.
