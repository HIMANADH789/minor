# Selection protocol - HIER-CONTINUOUS-SUP

## Unified pool (the defining change)

C = [G_full || Delta_full] with
- G_full: 19992 root candidates (full expanded PPV matrix; PPV_m^(1) = G_m, level 0 of the hierarchy)
- Delta_full: 449820 hierarchical telescoping residuals (30 edges x 14994 kernels; levels K=2,4,8,16)
- total: 469812 candidates (asserted C_dev.shape == (896, 469812))

## Stage 1 - CV diagnostic (never selects)

5-fold StratifiedKFold(shuffle=True, random_state=42) over train+val (896 rows). Per fold: top_f_select on the fold-train rows/labels ONLY (fold-internal selection over the full unified pool), held-out rows transformed with those identities, RidgeClassifierCV(ALPHAS) fit on fold-train, macro-F1 on the fold. Diagnostic only.

## Stage 2 - final selection (canonical R5 protocol)

top_f_select(C_dev, y_dev, 19,992) on the FULL dev matrix with train+val labels - exactly the protocol R5-HIGH and HIER-HIGH-SUP use for their final H selection. Selector semantics identical (sklearn f_classif, NaN->0, np.argsort(-f, kind='stable'), top-N). NO separate G stage, no rho, no N_G/N_H, no per-level quota.

## Stage 3 - refit + one official test evaluation

RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) fit on the selected 19992-column representation over train+val; single evaluation on the 3,582-row official test split.

## Resulting composition (emerged, diagnostic only)

root 14884, K2 3658, K4 1256, K8 164, K16 30 (sum 19992 = 19992).
