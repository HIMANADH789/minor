# CCWRU FINAL 3-SEED VALIDATION (test Macro-F1)

## CWRU_UNBAL

| Model | Seed 42 | Seed 43 | Seed 44 | Mean +- Std | Median | Min | Max |
|---|---|---|---|---|---|---|---|
| MiniRocket | 0.9917 | 0.9917 | 0.9917 | 0.9917 +- 0.0000 | 0.9917 | 0.9917 | 0.9917 |
| HERAMBA R5 | 0.9833 | 0.9792 | 0.9750 | 0.9792 +- 0.0042 | 0.9792 | 0.9750 | 0.9833 |

Matched-seed delta (HERAMBA R5 - MiniRocket):

| Seed | MR | HERAMBA R5 | Delta |
|---|---|---|---|
| 42 | 0.9917 | 0.9833 | -0.0084 |
| 43 | 0.9917 | 0.9792 | -0.0125 |
| 44 | 0.9917 | 0.9750 | -0.0167 |

Mean delta -0.0125, median -0.0125, std 0.0042; HERAMBA better on 0/3 seeds (descriptive only, n=3; no significance claim).

## CWRU_BAL

| Model | Seed 42 | Seed 43 | Seed 44 | Mean +- Std | Median | Min | Max |
|---|---|---|---|---|---|---|---|
| MiniRocket | 0.9947 | 0.9947 | 0.9947 | 0.9947 +- 0.0000 | 0.9947 | 0.9947 | 0.9947 |
| HERAMBA R5 | 0.9965 | 0.9912 | 0.9947 | 0.9941 +- 0.0027 | 0.9947 | 0.9912 | 0.9965 |

Matched-seed delta (HERAMBA R5 - MiniRocket):

| Seed | MR | HERAMBA R5 | Delta |
|---|---|---|---|
| 42 | 0.9947 | 0.9965 | +0.0018 |
| 43 | 0.9947 | 0.9912 | -0.0035 |
| 44 | 0.9947 | 0.9947 | +0.0000 |

Mean delta -0.0006, median +0.0000, std 0.0027; HERAMBA better on 1/3 seeds (descriptive only, n=3; no significance claim).

## Gates

| Gate | Observed | Reference | Tol | Verdict |
|---|---|---|---|---|
| MR CWRU_UNBAL seed42 | 0.9917 | 0.9917 | 0.0011 | PASS |
| R5 CWRU_UNBAL seed42 | 0.9833 | 0.9917 | 0.011 | PASS |
| MR CWRU_BAL seed42 | 0.9947 | 0.9947 | 0.0011 | PASS |
| R5 CWRU_BAL seed42 | 0.9965 | 0.9982 | 0.011 | PASS |

MR is deterministic across seeds by canonical convention (fixed extractor + deterministic Ridge): identical values across seeds 42/43/44 are expected and verified.
