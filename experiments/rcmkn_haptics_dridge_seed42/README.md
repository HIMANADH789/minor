# Haptics Differential-Ridge Control (seed 42)

Decisive regularization control for the finalized R2 model on Haptics.
Question: is the value of the full H block ([G || H], 9996 features)
controlled by feature identity or by block-level regularization strength?

All 4998 H features are kept for every gamma; only the H-block penalty
changes (alpha_H = gamma * alpha_G, alpha_G frozen at the canonical R2
value 4.281332398719396). gamma grid {1,2,4,8,16,32,64}, selected by
5-fold stratified CV on the 155-sample development set (spec sec 11),
tie tolerance 0.001 favors smaller gamma, official test evaluated exactly
once at the frozen gamma*.

## Result (summary)

    gamma CV curve (mean 5-fold Macro-F1):
        1: 0.5569   2: 0.5351   4: 0.5166   8: 0.4846
        16: 0.4614  32: 0.4499  64: 0.4381     (monotonically decreasing)

    gamma* = 1  ->  DRidge == R2 exactly
    val = 0.9014, test = 0.5500
    DRidge - M0 = +0.0526,  DRidge - R2 = +0.0000

The R2 result is NOT reproduced by differential regularization; the
H-block value at the canonical shared penalty is not excess capacity
to be pruned by a heavier penalty. See REPORT.md for the full analysis.

## Verification

gamma=1 identity gate: 0/308 prediction mismatches vs the canonical R2
stored test predictions; scaling trick vs direct primal block-penalty
solve: rel. max diff 1.33e-16; primal vs dual: 2.01e-16.
All 20 audit records pass (see audits.json).

## Artifacts

    config.json        frozen protocol + implementation notes
    audits.json        20 audit records (identity, leakage, dims, no gates)
    gamma_curve.json   per-gamma CV folds + block coefficient/df diagnostics
    diagnostics.json   G/H cross-correlation redundancy + identity check
    result.json        selected gamma, val/test, deltas
    predictions/       dridge_gamma1_pred_te.csv
    figures/           fig1..fig4 (val curve, test bars, beta norms, ratios)
    REPORT.md          16-section paper-ready report

## Reproduce

    cd ECG_Benchmark
    python -m experiments.rcmkn_haptics_dridge_seed42
    python -m pytest tests/test_rcmkn_haptics_dridge_seed42.py
