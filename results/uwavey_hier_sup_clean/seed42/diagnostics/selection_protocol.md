# Selection protocol - HIER-SUP-CLEAN

## Telescoping pool (clean formulation)

C = [D1 | D2 | D4 | D8 | D16] with
- D1: 19992 root increments Delta^(1) = PPV^(1) - PPV^(0) (PPV^(0) = 0, so Delta^(1) = PPV^(1) bit-exactly)
- D2/D4/D8/D16: 449820 occupancy-weighted telescoping residuals (30 edges x 14994 kernels)
- total: 469812 candidates (asserted C_dev.shape == (896, 469812))

## Stage 1 - CV diagnostic (never selects)

5-fold StratifiedKFold(shuffle=True, random_state=42) over train+val (896 rows). Per fold: top_f_select on the fold-train rows/labels ONLY (fold-internal selection over the full unified pool), held-out rows transformed with those identities, RidgeClassifierCV(ALPHAS) fit on fold-train, macro-F1 on the fold. Diagnostic only.

## Stage 2 - final selection (canonical R5 protocol)

top_f_select(C_dev, y_dev, 19,992) on the FULL dev matrix with train+val labels - exactly the protocol R5-HIGH, HIER-HIGH-SUP and HIER-CONTINUOUS-SUP use. Selector semantics identical (sklearn f_classif, NaN->0, np.argsort(-f, kind='stable'), top-N). ONE ranking over the complete pool: no G/H boundary, no rho, no protected budgets, no per-level quota.

## Stage 3 - refit + one official test evaluation

RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) fit on the selected 19992-column representation over train+val; single evaluation on the 3,582-row official test split.

## Resulting composition (emerged, diagnostic only)

D1 14453, D2/K2 4012, D4/K4 1436, D8/K8 65, D16/K16 26 (sum 19992 = 19992).
