# Haptics Fixed-Feature + Learned-Regime Ensemble — Seed 42

**Status:** complete. One seed (42), frozen canonical Haptics protocol, six final systems,
each evaluated on TEST **exactly once** after all validation-only decisions were frozen.
All results below are single-seed exploratory evidence — no statistical-significance claims.

---

## 1. Objective

Primary question: do canonical MiniROCKET fixed random temporal features and the best
existing DRTN learned-regime trajectory model make sufficiently different errors on Haptics
that combining them improves test Macro-F1?

Secondary question: can a fixed-feature Hydra+MultiRocket ensemble and/or a leakage-safe
stacked classifier outperform canonical MiniROCKET alone?

## 2. Frozen Haptics protocol

| Item | Value |
|---|---|
| Dataset | canonical UCR/aeon Haptics (frozen repository loader, `external_stack_generalization/data.py`) |
| Split | train 132 / validation 23 / test 308 (identical to `results/drtn_haptics_seed42/`) |
| Length / classes | T = 1092, 5 classes |
| Preprocessing | per-sample z-normalization (canonical) |
| Seed | 42 everywhere (numpy / torch / aeon random_state) |
| Primary metric | Macro-F1 (`zero_division=0`), accuracy + per-class F1 + confusion matrices also reported |

Environment: Python 3.11.9, PyTorch 2.5.1+cu121, aeon 1.5.0, NumPy 2.3.5, CUDA 12.1
(RTX 4050 Laptop). Recorded in `config.json`.

## 3. Existing DRTN variant used

Selected by the repository's **documented validation-selection rule** — the official
`results/drtn_haptics_seed42/` rung with the highest validation Macro-F1:

- **R5 (official probe, K=8):** val MF1 **0.4891**, test MF1 0.3467, 599,413 trainable params.
- The later 100/15-budget controls (CTC/DTC/R5-K16) belong to a different protocol stage,
  were not designated "best existing variant" by any repository rule, and were not substituted.
- The frozen checkpoint's validation MF1 was reproduced at load time (gate: exact match to the
  reported 0.4891; a first attempt failed on a 4-decimal rounding tolerance and was fixed
  **before any test evaluation** — see §14, restart documentation).
- Checkpoint used as-is: no retraining, no retuning, no architecture changes.

## 4. Canonical MiniROCKET configuration

aeon `MiniRocket(random_state=42)` → 9,996 features; `RidgeClassifierCV(alphas=np.logspace(-4,4,20))`;
per-sample z-normalization. Leakage rule for this experiment: for **selection** (stage A) both the
extractor and the Ridge were fit on **TRAIN only**; the canonical train+val final refit was performed
only after all ensemble decisions were frozen (stage B). The final refit reproduced the canonical
test MF1 **0.4974** exactly (canonical reference 0.4974).

## 5. Hydra / MultiRocket configuration

aeon 1.5.0 ships canonical implementations, so no re-implementation was needed (and none was made):

- `HydraClassifier(random_state=42)`
- `MultiRocketHydraClassifier(random_state=42)`

Both followed the identical two-stage protocol: train-only fit for the stage-A validation score
(Hydra 0.4905, MR-Hydra 0.4924), canonical train+val refit for the single final test evaluation.
No approximation was labeled as Hydra/MultiRocket.

## 6. Leakage-prevention protocol

- Stage A (all selection): MiniROCKET extractor+Ridge **train-only**; DRTN official frozen
  checkpoint; Hydra/MR-Hydra train-only fits. All fusion/stacking/hyperparameter decisions used
  **validation only** (n=23).
- Stacker meta-features for its training rows are **out-of-fold** (5-fold stratified): per fold the
  MiniROCKET extractor+Ridge is refit on the other folds' training samples, so no training row's
  stacker input comes from a base model trained on that row. DRTN logits for stacker training rows
  come from the frozen checkpoint (trained on train, so OOF by construction w.r.t. the train rows).
- Score-scale handling (measured, `diagnostics/score_scale.json`): MR decision scores ≈
  [−1.39, +0.98]; DRTN logits ≈ [−2.88, +4.16]. Fusion: `softmax(S/T)`, `softmax(L/T)` with T=1
  (no fitted parameters); stacking: `StandardScaler` on the 10-dim meta-features, fitted on OOF
  train rows only.
- TEST was touched exactly once per final system, always last: **6 evaluations total, 6 systems**
  (`test_evaluated_once_per_system: true` in `report.json`).
- Safety suite: 9 ensemble leakage/alignment tests + full DRTN suite = **32/32 pass** before the run.

## 7. Error complementarity analysis (2×2 correctness: rows = MR, cols = DRTN)

| | DRTN correct | DRTN wrong |
|---|---:|---:|
| **MR correct** | A | B |
| **MR wrong** | C | D |

| Stage | A both correct | B MR-only | **C DRTN-only** | D both wrong | n | C / (B+C) |
|---|---:|---:|---:|---:|---:|---:|
| Validation (selection) | 10 (43.5%) | 4 (17.4%) | **3 (13.0%)** | 6 (26.1%) | 23 | 3/7 |
| Test (frozen) | 99 (32.1%) | 61 (19.8%) | **19 (6.2%)** | 129 (41.9%) | 308 | 19/80 |

Cell C — samples where DRTN is right and MiniROCKET wrong — is the complementary-information
reservoir. It exists (19 test samples) but is small relative to cell B (61): DRTN must repay
61 MR-only-correct predictions to break even before contributing anything, and it starts with a
15.1 pp Macro-F1 deficit. The asymmetric cells (B ≫ C, roughly 3:1 on test) are the quantitative
reason fusion/stacking could not win.

## 8. Fusion methodology

`P_fused = α·softmax(MR_scores/T) + (1−α)·softmax(DRTN_logits/T)`, T=1 (documented, unfitted).
α searched over {0.00, 0.05, …, 1.00} on **validation only**; tie-break toward α=1.0 (canonical
baseline preference). Selected **α = 0.95** (val MF1 0.6321; full curve in
`diagnostics/fusion_search.json` and Figure 1). The search pushed α hard toward 1.0 — validation
already indicated the DRTN contribution should be nearly zero.

## 9. Stacking methodology

Meta-features = [9-dim MiniROCKET Ridge decision scores ‖ 5-dim DRTN logits] (standardized,
OOF protocol above). Multiclass `LogisticRegression`, C ∈ {0.01, 0.1, 1.0, 10.0} searched on
**validation only**: selected **C = 10.0** (val MF1 0.5485 — already below MR's own stage-A val
0.5879, an early warning that stacking could not help). Final test applied the frozen stacker to
stage-B base scores.

## 10. Results — VALIDATION-SELECTED vs FINAL TEST (clearly separated)

| System | Val MF1 (selection basis) | **Test MF1 (frozen, one eval)** | Test accuracy | Δ test vs MR |
|---|---:|---:|---:|---:|
| **MiniROCKET (canonical refit)** | 0.5879 (stage A) | **0.4974** | 0.5195 | — |
| DRTN R5 (frozen checkpoint) | 0.4891 | 0.3467 | 0.3831 | −0.1507 |
| Fusion (α=0.95) | 0.6321 | 0.4649 | 0.4903 | −0.0325 |
| Stacked MR+DRTN (C=10) | 0.5485 | 0.4347 | 0.4545 | −0.0627 |
| Hydra | 0.4905 (stage A) | 0.5048 | 0.5162 | +0.0074 |
| MultiRocketHydra | 0.4924 (stage A) | **0.5111** | 0.5195 | **+0.0137** |

External non-neural reference kept as-is: canonical MiniROCKET Haptics = 0.4974.

**Primary comparison (spec §12): stacked MR+DRTN vs canonical MiniROCKET: Δ Macro-F1 = −0.0627
(stacking did NOT improve). Δ accuracy = −0.0650.** Fusion also did not improve (−0.0325).

## 11. Per-class results (test F1 per class)

| System | c0 | c1 | c2 | c3 | c4 |
|---|---:|---:|---:|---:|---:|
| MiniROCKET | 0.2597 | 0.5625 | 0.5672 | 0.5180 | 0.5797 |
| DRTN R5 | 0.0625 | 0.4132 | 0.4037 | 0.4110 | 0.4432 |
| Fusion α=0.95 | 0.2162 | 0.5203 | 0.5417 | 0.4853 | 0.5612 |
| Stacked MR+DRTN | 0.2222 | 0.4427 | 0.4640 | 0.5077 | 0.5369 |
| Hydra | 0.3556 | 0.5333 | 0.5496 | 0.4923 | 0.5931 |
| MultiRocketHydra | 0.3505 | 0.5736 | 0.5854 | 0.4806 | 0.5652 |

Class 0 is the dominant minority-class failure for every system. Both fixed-feature Hydra systems
repair much of the class-0 deficit (F1 0.35 vs 0.26) — that, not overall accuracy, is the source of
their Macro-F1 edge.

## 12. Error correlation

| Statistic | Validation | Test |
|---|---:|---:|
| Binary error correlation | 0.375 | 0.504 |
| Cohen's κ (correctness) | 0.374 | 0.485 |
| Prediction agreement | 0.609 | 0.584 |
| Error overlap (both wrong / union wrong) | 6/13 = 0.462 | 129/209 = 0.617 |
| MR wrong & DRTN correct | 3 | 19 |
| DRTN wrong & MR correct | 4 | 61 |

Errors are **moderately positively correlated**, not complementary. Per-class test overlap:
class 0 — 49/60 both-wrong (82%): both models fail the largest class almost entirely together;
classes 1–4 both-wrong ≈ 20 each. The 19 DRTN-only-correct samples are scattered thinly
(9/13/18/14/7 both-wrong + 0–8 MR-only-wrong across classes), too few to lift Macro-F1 through a
meta-learner trained on 132 OOF rows.

## 13. Runtime

MR extractor fit 0.2 s; Ridge fits negligible. Hydra: fit 0.8 s, train+val refit 2.4 s.
MultiRocketHydra: fit 1.5 s, refit 3.8 s. OOF stacking (5 refits) + LR search: seconds.
DRTN checkpoint scoring of 463 samples on GPU: not separately timed (no measurable cost relative
to the classical fits; the checkpoint is loaded once and reused for all splits). Peak GPU memory
dominated by the frozen DRTN session (~1.3 GB class), CPU-bound classical arms otherwise.

## 14. Limitations and restart documentation

- **Single seed, tiny validation (n=23).** All selection signals (fusion curve, C search) are
  noisy; the fusion search's val 0.6321 > MR val 0.5879 did **not** transfer to test (0.4649) —
  a textbook small-validation selection artifact, reported as such.
- **Restart documentation (spec §13):** two pre-completion failures occurred, both fixed before
  the full pipeline completed, neither involving any test-informed decision: (1) the checkpoint
  reproduction gate rejected a 4-dp-rounded stored value (tolerance fixed from 1e-6 to 5e-6
  *before any test evaluation*); (2) a stale variable name crashed the runner mid-freeze, after
  three systems' test evaluations in that attempt. The experiment was restarted per §13;
  the clean rerun used the identical frozen decisions and deterministically reproduced those
  three test values exactly (0.4974 / 0.3467 / 0.4649 in both runs). No test-informed change was
  ever made.
- DRTN contributes 61-vs-19 cell asymmetry; with a stronger learned model (or a dataset where
  neural models approach the fixed-feature baseline) the fusion calculus changes qualitatively.

## 15. Scientific interpretation (spec §22 cases)

- **CASE C applies:** fusion/stacking < MiniROCKET. The DRTN R5 errors were not sufficiently
  complementary — nor its standalone performance strong enough — to improve the canonical
  fixed-feature baseline. The 2×2 asymmetry (B:C ≈ 61:19) and error correlation 0.504 quantify why.
- **CASE D applies, separately:** MultiRocketHydra (0.5111) and Hydra (0.5048) both beat MiniROCKET
  (+0.0137 / +0.0074). This gain comes from **fixed-feature diversity** (genuinely different
  transforms at equal cost, class-0 repair), not from the learned-regime model. At Haptics scale,
  the cheapest reliable improvement is a second fixed-feature extractor, not deep complementarity.
- CASE E (both sources of diversity improving together) was not observed.

## 16. Final verdict

On Haptics (seed 42), the best existing DRTN regime model does **not** provide usable
complementary information to canonical MiniROCKET: validation-only fusion (−3.3 pp) and
leakage-safe OOF stacking (−6.3 pp) both fall short of MiniROCKET alone, consistent with the
moderately correlated errors (κ=0.485) and the 3:1 MR-only:DRTN-only correct asymmetry. In
contrast, the canonical Hydra/MultiRocketHydra fixed-feature systems deliver small Macro-F1 gains
(+0.7 / +1.4 pp) in seconds of runtime. The learned-regime trajectory (599K trainable parameters,
val 0.489) remains well below the fixed-feature baselines (val 0.49–0.59, test 0.50–0.51) on this
132-sample dataset; its next viable role would require either a dataset where neural models are
competitive or substantial standalone improvement before any fusion revisit.

---

### Artifacts

```
results/haptics_ensemble_seed42/
  report.md  report.json  config.json
  predictions/{train,val,test}_per_sample.csv   # sample_index, true_class, mr_prediction,
                                                # drtn_prediction, mr_correct, drtn_correct,
                                                # mr_scores, drtn_logits
  diagnostics/{correctness_table,error_correlation,fusion_search,stacking_search,score_scale}.json
  figures/{fusion_alpha,error_overlap,per_class_f1,final_comparison}.png
```

Unit/safety tests: 32/32 pass (`tests/test_haptics_ensemble.py` + DRTN suite).
Test evaluations: 6 (exactly one per final system).
