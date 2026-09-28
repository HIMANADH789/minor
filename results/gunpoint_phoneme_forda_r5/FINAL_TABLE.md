# GunPoint / Phoneme / FordA -- MiniRocket vs HERAMBA R5 (test Macro-F1)

| Dataset | Model | Seed | Validation | Test | Fallback | Delta vs MR |
|---|---|---:|---:|---:|---|---:|
| GunPoint | MR | 42 | 1.0000 | 0.9933 | - | 0.0000 |
| GunPoint | MR | 43 | 1.0000 | 0.9933 | - | 0.0000 |
| GunPoint | MR | 44 | 1.0000 | 0.9933 | - | 0.0000 |
| GunPoint | HERAMBA R5 | 42 | 1.0000 | 1.0000 | no (HERAMBA selected on validation) | +0.0067 |
| GunPoint | HERAMBA R5 | 43 | 1.0000 | 1.0000 | no (HERAMBA selected on validation) | +0.0067 |
| GunPoint | HERAMBA R5 | 44 | 1.0000 | 1.0000 | no (HERAMBA selected on validation) | +0.0067 |
| Phoneme | MR | 42 | 0.1035 | 0.0808 | - | 0.0000 |
| Phoneme | MR | 43 | 0.1035 | 0.0808 | - | 0.0000 |
| Phoneme | MR | 44 | 0.1035 | 0.0808 | - | 0.0000 |
| Phoneme | HERAMBA R5 | 42 | 0.1464 | 0.1238 | no (HERAMBA selected on validation) | +0.0430 |
| Phoneme | HERAMBA R5 | 43 | 0.1464 | 0.1128 | no (HERAMBA selected on validation) | +0.0320 |
| Phoneme | HERAMBA R5 | 44 | 0.1445 | 0.1170 | no (HERAMBA selected on validation) | +0.0362 |
| FordA | MR | 42 | 0.9371 | 0.9499 | - | 0.0000 |
| FordA | MR | 43 | 0.9371 | 0.9499 | - | 0.0000 |
| FordA | MR | 44 | 0.9371 | 0.9499 | - | 0.0000 |
| FordA | HERAMBA R5 | 42 | 0.9389 | 0.9553 | no (HERAMBA selected on validation) | +0.0054 |
| FordA | HERAMBA R5 | 43 | 0.9463 | 0.9492 | no (HERAMBA selected on validation) | -0.0007 |
| FordA | HERAMBA R5 | 44 | 0.9445 | 0.9530 | no (HERAMBA selected on validation) | +0.0031 |

## Per-dataset aggregation

| Dataset | MR mean+-std | HERAMBA raw mean+-std | HERAMBA final mean+-std | per-seed d | mean d | HER wins |
|---|---|---|---|---|---|---|
| GunPoint | 0.9933 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | [0.0067, 0.0067, 0.0067] | +0.0067 | 3/3 |
| Phoneme | 0.0808 +- 0.0000 | 0.1179 +- 0.0056 | 0.1179 +- 0.0056 | [0.043, 0.032, 0.0362] | +0.0371 | 3/3 |
| FordA | 0.9499 +- 0.0000 | 0.9525 +- 0.0031 | 0.9525 +- 0.0031 | [0.0054, -0.0007, 0.0031] | +0.0026 | 2/3 |

Fallback summary: HERAMBA selected on validation in 9/9 seeds (0 fallbacks). Raw-HERAMBA failures are NOT hidden: raw test values are in FALLBACK_AUDIT.csv (notably Phoneme raw tests 0.1238/0.1128/0.1170 vs MR 0.0808 -- HERAMBA genuinely better; FordA seed43 raw 0.9492 slightly below MR 0.9499 despite higher validation).

n=3 seeds per dataset: descriptive statistics only; no significance claims.
