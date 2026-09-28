# CELL-C Margin Diagnostic — Haptics, Seed 42 (DIAGNOSTIC ONLY, NO RETRAINING)

## **CELL-C LOW-MARGIN CONCENTRATION: YES** (exploratory, single-seed, n=19)

The 19 DRTN-only-correct test samples are strongly concentrated in canonical MiniROCKET's
low-margin region: quartile counts fall monotonically **10 / 5 / 4 / 0**, the highest-margin
quartile contains **zero** Cell-C samples (binomial p = 0.0042 under uniform), Q1 holds
**2.11×** the uniform share (p = 0.0089), and Cell-C margins sit far below both MR-correct
(median 0.190 vs 0.539, Mann-Whitney p = 2.6e-05) and all other MR errors (0.190 vs 0.298,
one-sided p = 0.065, borderline). **Recommended next experiment: C — SELECTIVE ESCALATION**
(margin-gated), with two hard constraints derived from the same numbers.

---

## 1. What was done (and not done)

Diagnostic only. No model was trained, fine-tuned, selected, or altered. The analysis uses
the frozen official Haptics seed-42 ensemble artifacts (`results/haptics_ensemble_seed42/`)
plus one exact reconstruction step described in §2. MiniROCKET stayed at test Macro-F1
**0.4974**, DRTN R5 at **0.3467**, complementarity matrix **A=99, B=61, C=19, D=129** —
all re-verified at runtime (§3). No hyperparameter was chosen from test data; bucketing is
label-blind; no new seeds; no MultiRocketHydra involvement.

## 2. Margin recovery and provenance (integrity-critical)

Spec §2 requires Ridge `decision_function` margins for the canonical test system and allows
loading the already-fitted classifier if the stored artifact lacks usable scores. It lacked
them: the ensemble CSV's `mr_scores` column holds **stacker-scaled** scores (argmax
disagrees with the stored prediction on 25/308 rows), so they were **not** used.

Provenance determination (no refit-as-new-model, no selection): two deterministic candidate
reconstructions were compared to the frozen CSV predictions —
- train-only (stage-A) Ridge: matches CSV on 283/308 (test MF1 0.4678);
- **train+val refit (the canonical FINAL-TEST system): matches CSV on 308/308**,
  MF1 0.4974, matrix 99/61/19/129 — exact.

The frozen CSV's `mr_prediction` column therefore records the canonical train+val refit (the
0.4974 system of record), and the **train+val refit Ridge was accepted as the margin source
via 308/308 prediction identity** — i.e., it *is* the already-fitted canonical classifier,
re-instantiated deterministically (seeded aeon MiniRocket extractor + `RidgeClassifierCV`
with the frozen alpha grid; selected alpha 11.288, identical in both candidates).

Margin definition (spec §2): `margin_i = top1 − top2` of `decision_function(Fte)`,
multiclass 5-column scores. Range on test: [0.0007, 1.7783]; **no ties** (308 unique values),
so bucket assignment is unambiguous. Softmax probabilities were not used.

## 3. Data-integrity gates (all PASS)

| Check | Result |
|---|---|
| exactly 308 test samples | PASS |
| exactly 19 Cell-C samples | PASS |
| MiniROCKET test Macro-F1 = 0.4974 | PASS |
| DRTN R5 test Macro-F1 = 0.3467 | PASS |
| prediction ordering identical to frozen artifact | PASS (308/308 identity) |
| complementarity matrix A/B/C/D = 99/61/19/129 | PASS |
| no classifier refit beyond deterministic re-instantiation (proven by identity) | PASS |
| no hyperparameter selected from test results | PASS |
| test labels never used to alter any model | PASS (analysis-only) |

## 4. Margin definition and bucketing

Quartile buckets Q1–Q4 are the fixed empirical quartiles of the margin over all 308 test
samples (bounds 0.1502 / 0.2412 / 0.3643). Deciles analogous. Tie handling: margins are
distinct, but the rule is documented — equal values spanning a cut go to the **lower**
bucket via `searchsorted` ranks on unique margins, so earlier buckets keep nominal size.
Buckets were fixed before any label was consulted and do not depend on test labels or DRTN
performance.

## 5. Quartile analysis (primary)

| Bucket | n | frac | MR acc | DRTN acc | Cell-C (count) | % of 19 | Cell-C rate | uniform exp. | Cell-B (MR-only correct) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 (0–25%) | 77 | 0.250 | 0.390 | 0.273 | **10** | 52.6% | 0.130 | 4.75 | 19 |
| Q2 (25–50%) | 77 | 0.250 | 0.325 | 0.234 | **5** | 26.3% | 0.065 | 4.75 | 12 |
| Q3 (50–75%) | 77 | 0.250 | 0.584 | 0.377 | **4** | 21.1% | 0.052 | 4.75 | 20 |
| Q4 (75–100%) | 77 | 0.250 | 0.779 | 0.649 | **0** | 0.0% | 0.000 | 4.75 | 10 |

- **Enrichment_Q1 = 2.105×** (= 10/19 ÷ 0.25; identically Q1 rate 0.130 vs overall rate 0.062).
- Monotone decline 10→5→4→0; Q4 literally contains no Cell-C sample.
- Exploratory tests (post-hoc, n=19, no multiplicity control): χ² vs uniform = 10.68,
  df 3, p = 0.014; binomial Q1 enrichment p = 0.0089; binomial Q4 absence p = 0.0042.

## 6. Decile analysis (secondary)

| Decile | n | MR acc | DRTN acc | Cell-C | Cell-B | Cell-C rate |
|---|---:|---:|---:|---:|---:|---:|
| D1 (lowest) | 31 | 0.258 | 0.194 | 4 | 6 | 0.129 |
| D2 | 31 | 0.484 | 0.194 | 2 | 11 | 0.065 |
| D3 | 31 | 0.516 | 0.452 | 4 | 6 | 0.129 |
| D4 | 31 | 0.258 | 0.161 | 2 | 5 | 0.065 |
| D5 | 30 | 0.267 | 0.267 | 3 | 3 | 0.100 |
| D6 | 31 | 0.548 | 0.419 | 2 | 6 | 0.065 |
| D7 | 31 | 0.516 | 0.387 | 2 | 6 | 0.065 |
| D8 | 31 | 0.774 | 0.484 | 0 | 9 | 0.000 |
| D9 | 31 | 0.742 | 0.581 | 0 | 5 | 0.000 |
| D10 (highest) | 30 | 0.833 | 0.700 | 0 | 4 | 0.000 |

Enrichment_D1 = 2.105×. The **top-3 deciles (margin > ~0.44) contain zero Cell-C samples**
while holding 18 of MiniROCKET's 61 Cell-B (MR-only-correct) samples — the two models agree
almost perfectly wherever MiniROCKET is confident.

## 7. Median margins (primary robust statistic)

| Group | n | **median** | mean | IQR |
|---|---:|---:|---:|---|
| **Cell C (MR wrong, DRTN correct)** | 19 | **0.1903** | 0.2381 | 0.114–0.319 |
| MR correct | 160 | **0.5392** | 0.6053 | — |
| MR wrong & DRTN wrong (Cell D) | 129 | **0.3116** | 0.3773 | — |
| all MR errors (C+D) | 148 | **0.2976** | 0.3595 | — |

Comparisons (Mann-Whitney, exploratory; n=19):
- Cell-C vs MR-correct: statistic 656.0, **p = 2.6e-05** (one-sided, lower);
- Cell-C vs Cell-D (other MR errors): statistic 924.0, p = 0.084 (two-sided);
- Cell-C vs all MR errors: statistic 1104.5, **p = 0.065** (one-sided, lower) — borderline.

Cell-C is not merely "any MR error": it sits below the *average* MR error, though the
separation from other errors (median 0.190 vs 0.312) is modest at n=19 and only borderline
significant. MiniROCKET's errors are broadly low-margin, and DRTN's rescues are the
*lowest*-margin subset of them.

## 8. Primary question — answered

**Are the 19 DRTN-only-correct samples concentrated in MiniROCKET's low-margin region? YES.**
Evidence: 2.11× Q1 enrichment; perfectly monotone quartile counts 10/5/4/0; zero Cell-C in
the entire top quartile (and top-3 deciles); Cell-C median margin 0.190 vs 0.539 for
MR-correct (p = 2.6e-05) and 0.298 for MR errors generally (p = 0.065, borderline). This is
visual concentration **plus** supporting exploratory statistics, not visual concentration alone.

## 9. Decision rule — CASE 1 applies, with a critical qualification

By the spec's rule this is **CASE 1 — LOW-MARGIN CONCENTRATION**: Cell-C is substantially
enriched in the lowest-margin region and clearly lower than the MR-correct group.
**Selective escalation (C) is therefore empirically motivated — and is NOT implemented here.**

The qualification comes from the Cell-B column, which the same buckets expose: Q1 contains
19 MR-only-correct samples against 10 DRTN-only-correct ones, and DRTN's outright accuracy in
Q1 is only 0.273. **Unconditional** escalation of Q1 to DRTN would be strongly net-negative
(≈ −9 correct samples). The concentrated-but-unequal structure means the defensible form of
Experiment C is:

1. **margin-AND-DRTN-agreement gating** — escalate only low-margin samples where DRTN also
   disagrees in a consistent direction (a learned low-margin router, or a margin-conditional
   fusion restricted to Q1/Q2), never a bare margin threshold alone;
2. **val-only threshold fitting** on the 23-sample validation set, which is small — thresholds
   must be coarse (quartile-valued) and the procedure must be documented as fragile.

## 10. Limitations

- n=19: every count carries ±few-sample noise; all p-values are exploratory and post-hoc;
  no multiplicity correction; single seed, single dataset.
- Margins are Ridge-specific; another confidence measure could localize differently
  (though the Ridge margin is the canonical system's own confidence, which is what routing
  would use).
- The 308/308 prediction-identity provenance check, while exact, reconstructs the classifier
  in-process rather than loading a serialized pickle (none existed); the reconstruction uses
  only frozen configuration values.
- Concentration does not imply *exploitability*: Cell-B's presence in the same region shows
  the low-margin pool is a mixture, not a DRTN signal.

## 11. Verdict and recommendation

**CELL-C LOW-MARGIN CONCENTRATION: YES.** DRTN's residual useful information on Haptics is
localized where canonical MiniROCKET is least confident — exactly the structure needed for
selective escalation to have a chance, and exactly the structure that probability fusion and
OOF stacking (which diluted DRTN across all samples, including the 149 confident ones where
it has nothing to add) could not exploit.

**Recommend: PRIORITIZE C (SELECTIVE ESCALATION), in the margin-AND-agreement gated form,
with validation-only coarse thresholds.** Prioritizing A (feature-level fusion) remains the
fallback if a validation-fit gate cannot beat MiniROCKET on validation; per the spec, no
escalation machinery is implemented now.

---

### Artifacts

```
results/haptics_margin_diagnostic_seed42/
  report.md  report.json
  per_sample_margin.csv   # sample_index, true_class, mr_pred, drtn_pred,
                          # mr_margin, mr_correct, drtn_correct, cell_C
  margin_distribution.png      # box+strip: MR-correct / MR-wrong-D-wrong / Cell C
  margin_bucket_analysis.png   # quartile counts vs uniform + decile localization
experiments/haptics_margin_diagnostic_seed42/{diagnostic,figures}.py
```
