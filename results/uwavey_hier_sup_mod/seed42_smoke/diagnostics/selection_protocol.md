# Selection protocol - HIER-SUP-MOD

## Pool (locked, identical to HIER-SUP-CLEAN)

C = [D1 | D2 | D4 | D8 | D16] with 19992 root increments (Delta^(1) = PPV^(1), PPV^(0) = 0) + 449820 occupancy-weighted telescoping deltas = 469812 candidates. One shared semantic column order for train/val/test (hard layout guard).

## Stage 1 - CV diagnostic (never selects)

5-fold StratifiedKFold(shuffle=True, random_state=42). Per fold: ANOVA pieces on fold-train rows/labels ONLY -> EB prior (d0, s0^2) fit on FOLD-TRAIN variances ONLY -> moderated F -> top-19,992 -> RidgeClassifierCV(ALPHAS) on fold-train -> macro-F1 on fold-val. Diagnostic only.

## Stage 2 - final production selection (canonical boundary)

ANOVA design on train+val (260 rows): d_between = 7, d_error = 252 for EVERY candidate. s_j^2 = SSE_j/252; SINGLE scaled-F EB prior over the full pool: d0 = 0.472385, s0^2 = 2.356866e-04 (Smyth log-moment matching; no validation tuning, no grid search). Posterior s_t^2 = (d0 s0^2 + d_e s^2)/(d0 + d_e); ranking F_t = MS_between/s_t^2 descending, ties by candidate_index; top 19,992. NO raw-F mixing, no FDR threshold, no quotas, no occupancy in the statistic.

## Stage 3 - refit + one official test evaluation

d0, s0^2 and the selected identities are FROZEN before the test transform; RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) on the selected 19992-column representation over train+val; single evaluation on the 3,582-row official test split.

## Emerged composition (diagnostic only)

D1 14865, D2/K2 3655, D4/K4 1290, D8/K8 160, D16/K16 22 (sum 19992 = 19992).

Raw vs moderated top-B overlap: intersection 19990/19992, Jaccard 0.9998 (raw-only 2, moderated-only 2).
