# TURS-FINAL: MiniROCKET + Statistically Validated KTM-W

Final four-dataset experiment of the TURS research line. Seed 42, S = 200, alpha = 0.05 (one-sided), decision rule frozen before test evaluation.

## 1. Executive Summary

- Canonical MiniROCKET reproduced the verified baseline on all four datasets within 0.02 Macro-F1: **PASS**.
- Statistical gate (circular-shift null, S = 200, validation only) decisions: **KEEP_KTM_W: 1** (ECG5000_BAL), **DROP_KTM_W: 3** (ECG5000_UNBAL, CWRU_UNBAL, CWRU_BAL).
- Final gated system mean test Macro-F1 = **0.8102** vs canonical MiniROCKET **0.8089** (mean delta **+0.0013**).
- Claim classification: **B. DATASET-SPECIFIC IMPROVEMENT**.

## 2. Motivation

MiniROCKET PPV features count *whether* kernel responses exceed learned biases but discard *where in the window* those responses occur. The KTM line hypothesized that window-relative activation timing carries additional, alignment-dependent predictive information. Earlier iterations suffered from two defects: a no-op top-K selection (K_TOP exceeded the 84 underlying kernel configurations) and null distributions with only S = 20 shifts (p-value resolution 1/21). This final experiment fixes both: all 84 kernel groups enter as features and the null uses S = 200.

## 3. Research Question

> Can window-relative MiniROCKET kernel activation timing provide additional predictive information beyond canonical MiniROCKET, and can a validation-only circular-shift statistical test identify when that information is genuinely useful?

The system determines, independently per dataset, KEEP_KTM_W or DROP_KTM_W **before** the test split is used. It is a dataset-level statistical selection system, not a mixture model: no alpha-blending, no per-sample routing, no learned gate.

## 4. Canonical MiniROCKET Baseline

aeon MiniRocket (n_kernels = 10000, random_state = 42, aeon 1.2.0), per-sample z-normalization, transformer fitted on the training split, RidgeClassifierCV(alphas = np.logspace(-4, 4, 20)) fitted on train+validation rows. This is the exact pipeline that produced the verified reference values.

| Dataset | Reference MF1 | Reproduced MF1 | |diff| | Status |
|---|---:|---:|---:|---|
| ECG5000_UNBAL | 0.5938 | 0.5938 | 0.0000 | PASS |
| ECG5000_BAL | 0.6553 | 0.6553 | 0.0000 | PASS |
| CWRU_UNBAL | 0.9917 | 0.9917 | 0.0000 | PASS |
| CWRU_BAL | 0.9947 | 0.9947 | 0.0000 | PASS |

## 5. KTM-W Definition

For each MiniROCKET **kernel group** (one of the 84 fixed tap-triple kernel configurations C(9,3); not "84 kernels" in the feature count sense), using the previously validated activation mask {t : C(t) > b} shared with aeon's PPV computation:

1. activated positions tau_1 < ... < tau_k inside the evaluated window;
2. normalized positions x_i = tau_i / T_e (T_e = number of evaluated positions, i.e. window-relative);
3. W = mean_i | x_(i) - i/(k+1) |  (sorted x_(i); the empirical 1D Wasserstein distance to Uniform[0,1]).

The per-feature W (one per MiniROCKET feature, ~9996 columns) from `src/features/ktm.py` is **averaged within each kernel group**, yielding exactly 84 KTM-W features W_1..W_84. KTM-W is deterministic, window-relative, derived from amplitudes only through the MiniROCKET activation masks, and adds no learned parameters. No mu, sigma^2, CPT phase, period estimation, or cyclic statistics are used. No top-K selection is applied (this fixes the earlier K_TOP no-op).

## 6. Statistical Gate

The gate is a hypothesis test, not a learned component: no neural network, no training, no parameters fitted beyond the two ridge models being compared. Performed independently per dataset on TRAIN + VALIDATION only.

## 7. Circular-Shift Null

For each of S = 200 repetitions, every validation sample is circularly shifted by an independent random offset s in {0,...,T-1}: x_shift[t] = x[(t - s) mod T]. Signal values, signal length, and labels are preserved; no values are permuted. KTM-W is recomputed on the shifted validation signals; the base MiniROCKET features are reused unchanged. The augmented ridge is refitted with the same standardization, classifier, alpha grid and validation protocol as the real model, giving Delta_null_j.

## 8. Empirical p-value

p = (1 + count(Delta_null_j >= Delta_real)) / (S + 1), S = 200 (one-sided; minimum resolution 1/201 ~ 0.005, fixing the earlier 1/21 problem). H0: real KTM-W does not improve validation performance beyond the circular-shift null. H1: it does.

## 9. Keep/Drop Rule (pre-registered)

KEEP_KTM_W iff p < 0.05 AND Delta_real > 0; otherwise DROP_KTM_W. The rule was frozen before test evaluation; no p <= 0.1, no "almost significant", no manual judgment, no test performance.

## 10. Leakage Prevention

Overall audit: **PASS**. Key points:

- gate_uses_train_val_only: PASS - run_detector signature contains no test arrays; MiniROCKET fitted on train; scores computed on val
- test_excluded_until_freeze: PASS - test split touched only by (a) canonical baseline reproduction against already-published reference values and (b) the single final evaluation after decisions were frozen; no gate statistic uses test
- standardization_fit_on_train_only: PASS - BlockScaler fitted on train rows only (unit tests 12/13)
- decision_frozen_before_final_test: PASS - freeze_decisions() executed and selected_models.json written before stage_final() ran
- S_fixed_at_200: PASS - S=200 declared before results; assert in run_detector; unit test 18
- threshold_fixed_at_0.05: PASS - keep_drop_rule uses 0.05 exactly
- no_posthoc_override: PASS - counterfactual models labelled POST-HOC / REFERENCE ONLY; never used to alter the decision
- no_per_example_routing: PASS - one constant decision per dataset (unit tests 16/17)

## 11. Unit Tests

18 pre-registered tests (`tests/test_turs_final.py`, copied to `results/turs_final/tests/`) cover KTM-W formula correctness, agreement with the previously validated implementation and aeon's activation masks, exactly 84 group features, circular-shift properties, shift reproducibility, label preservation, p-value formula/range, the exact keep/drop rule, blockwise train-only standardization, NaN/inf freedom, decision constancy, absence of per-example routing, and S = 200 execution. All tests passed before the benchmark ran.

## 12. Dataset Protocol

| Dataset | n_train | n_val | n_test | T | classes |
|---|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | 3400 | 600 | 1000 | 140 | 5 |
| ECG5000_BAL | 5226 | 923 | 1000 | 140 | 5 |
| CWRU_UNBAL | 1156 | 204 | 240 | 1024 | 4 |
| CWRU_BAL | 2727 | 482 | 567 | 1024 | 4 |

Canonical splits and preprocessing unchanged from the benchmark.

## 13. Validation Results

| Dataset | MF1 Base Val | MF1 Real KTM Val | Delta Real |
|---|---:|---:|---:|
| ECG5000_UNBAL | 0.6405 | 0.6254 | -0.0151 |
| ECG5000_BAL | 0.9546 | 0.9565 | +0.0019 |
| CWRU_UNBAL | 0.9804 | 0.9853 | +0.0049 |
| CWRU_BAL | 0.9896 | 0.9896 | +0.0000 |

## 14. Null Results

| Dataset | Delta Real | Null Mean | Null Median | Null Std | Null 5% | Null 95% | Max | Min | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | -0.0151 | -0.0686 | -0.0680 | 0.0285 | -0.1169 | -0.0244 | -0.0144 | -0.1601 | 0.0100 |
| ECG5000_BAL | +0.0019 | -0.0111 | -0.0111 | 0.0045 | -0.0185 | -0.0041 | -0.0010 | -0.0223 | 0.0050 |
| CWRU_UNBAL | +0.0049 | +0.0010 | +0.0000 | 0.0045 | -0.0049 | +0.0098 | +0.0147 | -0.0098 | 0.3532 |
| CWRU_BAL | +0.0000 | +0.0011 | +0.0000 | 0.0016 | -0.0020 | +0.0042 | +0.0042 | -0.0021 | 0.9353 |

See Figure 1 (`figures/fig1_real_vs_null_delta.png`) for the real delta against the null distributions.

## 15. Regime Decisions

| Dataset | MF1 Base Val | MF1 Real KTM Val | Delta Real | Null Mean | Null 95% | p | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| ECG5000_UNBAL | 0.6405 | 0.6254 | -0.0151 | -0.0686 | -0.0244 | 0.0100 | DROP_KTM_W |
| ECG5000_BAL | 0.9546 | 0.9565 | +0.0019 | -0.0111 | -0.0041 | 0.0050 | KEEP_KTM_W |
| CWRU_UNBAL | 0.9804 | 0.9853 | +0.0049 | +0.0010 | +0.0098 | 0.3532 | DROP_KTM_W |
| CWRU_BAL | 0.9896 | 0.9896 | +0.0000 | +0.0011 | +0.0042 | 0.9353 | DROP_KTM_W |

## 16. Final Test Results

| Dataset | Gate Decision | Base MR Test MF1 | Selected Final MF1 | Delta |
|---|---|---:|---:|---:|
| ECG5000_UNBAL | DROP_KTM_W | 0.5938 | 0.5938 | +0.0000 |
| ECG5000_BAL | KEEP_KTM_W | 0.6553 | 0.6606 | +0.0053 |
| CWRU_UNBAL | DROP_KTM_W | 0.9917 | 0.9917 | +0.0000 |
| CWRU_BAL | DROP_KTM_W | 0.9947 | 0.9947 | +0.0000 |

Counterfactual (non-selected) models are reported for reference only:

| Dataset | Counterfactual model | Test MF1 | Status |
|---|---|---:|---|
| ECG5000_UNBAL | MiniROCKET + KTM-W (post-hoc reference) | 0.5814 | POST-HOC / REFERENCE ONLY (newly evaluated; never used to alter the decision) |
| ECG5000_BAL | MiniROCKET (post-hoc reference) | 0.6553 | REFERENCE ONLY (existing frozen evaluation) |
| CWRU_UNBAL | MiniROCKET + KTM-W (post-hoc reference) | 0.9917 | POST-HOC / REFERENCE ONLY (newly evaluated; never used to alter the decision) |
| CWRU_BAL | MiniROCKET + KTM-W (post-hoc reference) | 0.9947 | POST-HOC / REFERENCE ONLY (newly evaluated; never used to alter the decision) |

## 17. Per-Class Results

Final selected model per dataset (full detail in `per_class_results.csv`):

**ECG5000_UNBAL** (MiniROCKET): accuracy 0.9540, Macro-F1 0.5938, weighted F1 0.9458, balanced accuracy 0.5577

- per-class precision: [0.9898, 0.918, 0.6667, 0.75, 0.0]
- per-class recall: [1.0, 0.983, 0.4211, 0.3846, 0.0]
- per-class F1: [0.9949, 0.9494, 0.5161, 0.5085, 0.0]
- confusion matrix: [[584, 0, 0, 0, 0], [0, 347, 2, 4, 0], [3, 7, 8, 1, 0], [2, 22, 0, 15, 0], [1, 2, 2, 0, 0]]
- canonical MiniROCKET baseline: accuracy 0.9540, Macro-F1 0.5938, weighted F1 0.9458

**ECG5000_BAL** (MiniROCKET + KTM-W): accuracy 0.9540, Macro-F1 0.6606, weighted F1 0.9505, balanced accuracy 0.6364

- per-class precision: [0.9949, 0.9319, 0.6471, 0.6538, 0.25]
- per-class recall: [0.9983, 0.9688, 0.5789, 0.4359, 0.2]
- per-class F1: [0.9966, 0.95, 0.6111, 0.5231, 0.2222]
- confusion matrix: [[583, 0, 0, 0, 1], [0, 342, 3, 8, 0], [1, 4, 11, 1, 2], [2, 19, 1, 17, 0], [0, 2, 2, 0, 1]]
- canonical MiniROCKET baseline: accuracy 0.9520, Macro-F1 0.6553, weighted F1 0.9498

**CWRU_UNBAL** (MiniROCKET): accuracy 0.9917, Macro-F1 0.9917, weighted F1 0.9917, balanced accuracy 0.9917

- per-class precision: [1.0, 1.0, 0.9677, 1.0]
- per-class recall: [1.0, 1.0, 1.0, 0.9667]
- per-class F1: [1.0, 1.0, 0.9836, 0.9831]
- confusion matrix: [[60, 0, 0, 0], [0, 60, 0, 0], [0, 0, 60, 0], [0, 0, 2, 58]]
- canonical MiniROCKET baseline: accuracy 0.9917, Macro-F1 0.9917, weighted F1 0.9917

**CWRU_BAL** (MiniROCKET): accuracy 0.9947, Macro-F1 0.9947, weighted F1 0.9947, balanced accuracy 0.9947

- per-class precision: [1.0, 1.0, 0.986, 0.9929]
- per-class recall: [1.0, 1.0, 0.993, 0.9859]
- per-class F1: [1.0, 1.0, 0.9895, 0.9894]
- confusion matrix: [[142, 0, 0, 0], [0, 141, 0, 0], [0, 0, 141, 1], [0, 0, 2, 140]]
- canonical MiniROCKET baseline: accuracy 0.9947, Macro-F1 0.9947, weighted F1 0.9947

## 18. Computational Cost

| Dataset | MR fit (s) | Base transform (s) | KTM-W extract (s) | Null gen (s) | Ridge CV detector (s) | Final fit (s) | Final inference (s) | Total (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | 0.2 | 0.3 | 2.9 | 1195.3 | 1095.6 | 7.0 | 7.0 | 2315.9 |
| ECG5000_BAL | 0.0 | 0.5 | 4.9 | 3064.7 | 2929.0 | 21.6 | 21.6 | 6072.0 |
| CWRU_UNBAL | 0.3 | 0.4 | 3.3 | 214.6 | 110.2 | 0.8 | 0.8 | 331.3 |
| CWRU_BAL | 0.1 | 0.8 | 7.6 | 788.8 | 551.8 | 4.1 | 4.2 | 1361.7 |

The statistical gate is setup-time computation only. At deployment there is no routing overhead: the selected model is a single fixed pipeline per dataset. For KEEP datasets, MiniROCKET+KTM-W inference adds only the 84 group statistics to the transform pass.

## 19. Limitations

- Four datasets only; the empirical p-value is a within-dataset null diagnostic, not a meta-analysis. The four decisions are reported descriptively and do not constitute a large sample of independent populations.
- The gate compares two ridge models on a single validation split; validation-set noise propagates to the decision.
- Kernel-group averaging discards per-feature timing heterogeneity within a group (a deliberate simplicity/robustness trade-off that fixes the 84-feature dimensionality).
- Circular shifts test alignment-dependence only; other nulls (block permutations, surrogate signals) could probe other mechanisms.

## 20. Final Scientific Conclusion

1. **Did canonical MiniROCKET reproduce the verified baseline?** Yes - all four datasets within 0.02 MF1.
2. **KEEP_KTM_W datasets:** ECG5000_BAL.
3. **DROP_KTM_W datasets:** ECG5000_UNBAL, CWRU_UNBAL, CWRU_BAL.
4. **Did the S >= 200 test support the original KTM-AV finding?** KTM-AV (S = 20) selected KEEP for ECG5000_UNBAL; at S = 200 that decision is not reproduced. See the per-dataset p-values for the full picture.
5. **Did the final held-out test confirm the selected augmentation?** Yes on 1/1 KEEP datasets (ECG5000_BAL showed a positive test delta).
6. **Final mean Macro-F1:** 0.8102.
7. **Mean change from canonical MiniROCKET:** +0.0013.
8. **Universal or dataset-specific?** Dataset-specific (the gate selected augmentation only where the validation alignment test detected timing information).
9. **Is KTM-W statistically defensible as a dataset-level augmentation?** Yes, on the datasets where the validation-only circular-shift test fired, with the pre-registered rule fixed before test evaluation.
10. **Is this simple gated MiniROCKET system strong enough to be the final TURS model?** Yes as a *dataset-level model-selection* system: it is leakage-free, cheap at deployment, fully reproducible, and never worse than canonical MiniROCKET by construction when the gate says DROP. Its value is precisely the absence of unjustified complexity.

## Claim Classification (sec. 40)

- Final result class: **B. DATASET-SPECIFIC IMPROVEMENT**
- Statistical gate validity: **STRONG** (S = 200 pre-registered, exact one-sided test, frozen rule, leakage audit PASS)
- KTM-W predictive contribution: **WEAK**
- Cross-dataset consistency: **STRONG** (4/4 decisions confirmed on test)
- Reproducibility: **STRONG** (fixed seeds, cached shifts, pinned library versions)

## Model Description (for publication)

> The proposed system augments canonical MiniROCKET with a compact kernel activation-timing statistic only when a validation-only circular-shift test provides evidence that the timing feature carries alignment-dependent predictive information. The gate is dataset-level statistical selection: it is not learned, not neural, not adaptive per-sample, and involves no dynamic routing.

## Claim Limit (sec. 31)

KTM-W is **not** claimed to universally improve MiniROCKET. KTM-W provided statistically validated dataset-specific augmentation on the subset of benchmark datasets for which the validation alignment test detected additional timing information.

## Reproducibility Record (sec. 34)

- seed = 42, S = 200, alpha = 0.05
- aeon 1.2.0, sklearn 1.6.1, numba 0.61.2, numpy 2.2.6
- RidgeClassifierCV(alphas = np.logspace(-4, 4, 20)) throughout; baseline and augmented models use identical classifier machinery
- All circular shift offsets: `configs/null_shifts.json`
- MiniROCKET configuration: n_kernels = 10000, random_state = 42, univariate, per-sample z-normalization
- Split identifiers, feature dimensions, standardization statistics, validation scores, null scores, p-values, decisions and final test results: `full_results.json`, `diagnostics.json`, `selected_models.json`, CSV tables
