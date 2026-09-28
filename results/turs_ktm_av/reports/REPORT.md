# TURS-KTM-AV: Alignment-Validated Kernel Timing Moments

## The Final Controlled Experiment in the KTM/Timing Line

---

## 1. Executive Summary

This experiment tests whether the original window-relative Kernel Timing Moment (KTM-W) carries statistically significant predictive information beyond canonical MiniROCKET, specifically when that information arises from **temporal alignment** rather than arbitrary temporal arrangement.

The protocol uses a circular-shift null distribution as a controlled comparison: we compute model performance with real KTM-W features and compare it against performance with KTM-W features derived from circularly-shifted signals. A one-sided empirical p-value determines whether real KTM-W improvement exceeds what shifted (alignment-destroying) variants achieve.

**Results:**

| Dataset | Δ_real | Null mean | Null 95% | p-value | Decision |
|---|---:|---:|---:|---:|---|
| ECG5000_UNBAL | +0.0035 | -0.3831 | -0.3427 | 0.0476 | **KEEP_KTM_W** |
| ECG5000_BAL | -0.0022 | -0.6713 | -0.6503 | 0.0476 | DROP_KTM_W |
| CWRU_UNBAL | -0.0196 | -0.0303 | -0.0098 | 0.3333 | DROP_KTM_W |
| CWRU_BAL | -0.0270 | -0.0366 | -0.0308 | 0.0952 | DROP_KTM_W |

**Final held-out test:**

| Dataset | MR Baseline | Decision | Final Model | Final Test MF1 | Δ |
|---|---:|---|---|---:|---:|
| ECG5000_UNBAL | 0.5938 | KEEP_KTM_W | MR + KTM-W | 0.6115 | +0.0177 |
| ECG5000_BAL | 0.6553 | DROP_KTM_W | MR | 0.6553 | 0.0000 |
| CWRU_UNBAL | 0.9917 | DROP_KTM_W | MR | 0.9917 | 0.0000 |
| CWRU_BAL | 0.9947 | DROP_KTM_W | MR | 0.9947 | 0.0000 |

KTM-W was distinguishable from its circular-shift null on ECG5000_UNBAL (p = 0.0476), producing a held-out test improvement of +0.0177 MF1. On ECG5000_BAL, the null was also statistically significant but Δ_real was negative, so KTM-W was correctly dropped. On both CWRU datasets, KTM-W failed the criterion, consistent with hypothesis.

---

## 2. Motivation

Previous experiments in the KTM/timing line tested various timing statistics (μ, σ², circular phase, CPT) in combination with routing, gating, and ensemble mechanisms. These added complexity without a controlled baseline comparison.

This experiment strips away all auxiliary components and tests **one precise hypothesis**:

> "Does the original window-relative KTM-W contain statistically significant predictive information beyond canonical MiniROCKET, as opposed to merely responding to arbitrary temporal arrangement?"

The key innovation is the **circular-shift null**: by circularly shifting validation signals, we preserve all amplitude statistics, signal length, local shape content, and global energy while disrupting the alignment between activation timing and the original window coordinate system. This isolates whether the **positional alignment** of activations (relative to the signal window) carries information, or whether any temporal statistic would suffice.

---

## 3. KTM-W Definition

For each MiniROCKET kernel m with activation indices τ₁ < ... < τₖ on a signal of length T:

1. **Normalize**: x_i = τ_i / T
2. **Compute**: W_m = mean_i |x_i - i/(k+1)|

This measures the average deviation of normalized activation positions from a uniform distribution. When activations are uniformly spaced, W ≈ 0. When they cluster at specific positions, W is larger.

Only W is computed. No μ, σ², circular phase, CPT, or period estimates.

The implementation is in `src/features/ktm.py` (`ktm_transform` function), returning one W value per MiniROCKET feature (9996 features from 84 kernel configurations × ~119 dilation/bias combinations each).

---

## 4. Top-K Selection

Selection uses the **absolute MiniROCKET-alone classifier coefficient magnitude**:

1. Fit canonical MiniROCKET + Ridge on training data
2. Obtain fitted Ridge coefficients for MiniROCKET features
3. Rank kernels by the absolute magnitude of their corresponding coefficients (aggregated within each of the 84 kernel configurations)
4. Select the top K kernel configurations

K = 1024 (fixed before evaluation). Since MiniRocket has 84 kernel configurations (C(9,3) combinations), K = 1024 > 84, meaning **all kernel configurations are selected**. This is equivalent to using all 9996 features.

The selection criterion comes exclusively from MiniROCKET coefficients — no KTM-W values, no validation performance, no test data.

**Why coefficient-based selection:** We test whether timing information attached to the **most predictive** MiniROCKET kernels carries additional signal. Kernel importance comes strictly from the base model.

---

## 5. Circular-Shift Null

For each validation sample with signal x of length T:

```
x_shift[t] = x[(t - s) mod T]
where s ~ Uniform({0, ..., T-1})
```

This is a circular re-indexing — values are not shuffled, permuted independently, or altered in any way. The shifted signal is an exact circular re-indexing of the original validation window.

**What the null preserves:**
- Amplitude distribution
- Local shape content
- Global energy (L² norm)
- Signal length
- MiniROCKET kernel responses up to temporal relocation

**What the null disrupts:**
- Alignment between activation timing and original window position
- The relationship between τ_i positions and the coordinate system

N_NULL_SHIFTS = 20 per validation sample, with a fixed deterministic RNG seed (SEED + dataset_index).

---

## 6. What Exactly Is Being Tested

The null does **not** merely compare mean(W_real) vs mean(W_null). The primary test is **model performance**.

For each null repetition j:
- Compute KTM-W on circularly-shifted validation signals
- Evaluate MR + KTM-W_null_j on validation
- Compute Δ_null_j = MF1(MR + KTM-W_null_j) - MF1(MR baseline)

Compare against:
- Δ_real_val = MF1(MR + real KTM-W) - MF1(MR baseline)

The question: does the real alignment provide more predictive improvement than shifted alternatives?

---

## 7. Validation Decision Rule

**Pre-registered criterion** (alpha = 0.05):

IF: p < 0.05 AND Δ_real_val > 0 THEN: KEEP KTM-W
Otherwise: DROP KTM-W

**Exact statistical test:**

One-sided empirical permutation-style p-value:

```
p = (1 + count(Δ_null_j >= Δ_real_val)) / (1 + N_NULL_SHIFTS)
```

This is the standard empirical p-value for testing whether the observed statistic exceeds the null distribution.

**H₀:** Real KTM-W provides no more validation improvement than the circular-shift null.

**H₁:** Real KTM-W provides greater validation improvement than the circular-shift null.

---

## 8. Leakage Prevention

The experiment enforces strict separation:

1. **Top-K selection** uses only MiniROCKET coefficients from training data
2. **KTM-W computation** on validation data uses the MiniROCKET transformer fitted on training data
3. **Null distribution** is built on validation data only — no test data is accessed
4. **The keep/drop decision** is made before any test evaluation
5. **Standardization** statistics are fit on training data only (for validation evaluation) or train+val (for final test evaluation)
6. **After the decision**, the final model is refit on train+val and evaluated ONCE on test

The test set is never used for:
- K selection
- Keep/drop decisions
- Null-shift configuration
- Threshold selection
- Any method modification

---

## 9. Unit Tests

13 unit tests were created in `tests/test_ktm_av.py` and all pass:

1. **test_circular_shift_preserves_multiset** — Circular shift preserves exact multiset of signal values
2. **test_circular_shift_preserves_length** — Circular shift preserves signal length
3. **test_deterministic_rng** — Same seed produces same shifts
4. **test_ktm_w_matches_reference** — KTM-W matches previously validated implementation
5. **test_activation_mask_matches_minirocket** — Activation masks match MiniROCKET
6. **test_top_k_from_mr_coefficients_only** — Top-K is derived only from MiniROCKET coefficient magnitude
7. **test_top_k_fixed_before_null** — Top-K is fixed before null/model evaluation
8. **test_null_shifts_preserve_labels** — Null shifts do not alter labels
9. **test_no_nans_infs** — No NaNs or Infs in features
10. **test_standardization_is_blockwise** — Standardization is blockwise (separate for MR and KTM-W)
11. **test_p_value_computation** — p-value computation is correct
12. **test_no_test_data_in_decision** — No test data enters the decision function
13. **test_smoke_experiment** — End-to-end smoke test with synthetic data

---

## 10. Per-Dataset Validation Results

### ECG5000_UNBAL (5 classes, T=140)
- **Data**: train=3400, val=600, test=1000
- **MR baseline (val)**: MF1 = 0.6405
- **MR + real KTM-W (val)**: MF1 = 0.6440
- **Δ_real = +0.0035**
- Null mean = -0.3831, null std = 0.0205, null median = -0.3831, null 95th = -0.3427
- **p = 0.0476**
- **DECISION: KEEP_KTM_W**

### ECG5000_BAL (5 classes, T=140)
- **Data**: train=5226, val=923, test=1000
- **MR baseline (val)**: MF1 = 0.9546
- **MR + real KTM-W (val)**: MF1 = 0.9524
- **Δ_real = -0.0022**
- Null mean = -0.6713, null std = 0.0145, null median = -0.6759, null 95th = -0.6503
- **p = 0.0476**
- **DECISION: DROP_KTM_W** (Δ_real ≤ 0, fails the positive-improvement criterion)

### CWRU_UNBAL (4 classes, T=1024)
- **Data**: train=1156, val=204, test=240
- **MR baseline (val)**: MF1 = 0.9804
- **MR + real KTM-W (val)**: MF1 = 0.9608
- **Δ_real = -0.0196**
- Null mean = -0.0303, null std = 0.0135, null median = -0.0296, null 95th = -0.0098
- **p = 0.3333**
- **DECISION: DROP_KTM_W**

### CWRU_BAL (4 classes, T=1024)
- **Data**: train=2727, val=482, test=567
- **MR baseline (val)**: MF1 = 0.9896
- **MR + real KTM-W (val)**: MF1 = 0.9626
- **Δ_real = -0.0270**
- Null mean = -0.0366, null std = 0.0049, null median = -0.0373, null 95th = -0.0308
- **p = 0.0952**
- **DECISION: DROP_KTM_W**

---

## 11. Null Distributions

### ECG5000_UNBAL
The null distribution is concentrated far below zero (mean = -0.3831), indicating that circularly-shifting ECG signals dramatically disrupts MiniROCKET + KTM-W performance. The real Δ (+0.0035) exceeds all 20 null values, yielding p = 0.0476. The real KTM-W provides positive improvement while the null KTM-W causes catastrophic degradation.

### ECG5000_BAL
Similar pattern: null distribution concentrated at mean = -0.6713. The real Δ (-0.0022) is negative, meaning real KTM-W slightly hurts performance. While the null is statistically distinguishable from real (p = 0.0476), the real Δ is negative, so the decision is DROP.

### CWRU_UNBAL
Null distribution centered near zero (mean = -0.0303) with overlap. The real Δ (-0.0196) falls within the null distribution (p = 0.3333). KTM-W is not distinguishable from shifted nulls.

### CWRU_BAL
Null distribution centered near zero (mean = -0.0366). The real Δ (-0.0270) falls within the null distribution (p = 0.0952). KTM-W is not distinguishable from shifted nulls.

---

## 12. Keep/Drop Decisions

| Dataset | Δ_real | Null mean | Null 95% | p-value | Decision |
|---|---:|---:|---:|---:|---|
| ECG5000_UNBAL | +0.0035 | -0.3831 | -0.3427 | 0.0476 | **KEEP_KTM_W** |
| ECG5000_BAL | -0.0022 | -0.6713 | -0.6503 | 0.0476 | DROP_KTM_W |
| CWRU_UNBAL | -0.0196 | -0.0303 | -0.0098 | 0.3333 | DROP_KTM_W |
| CWRU_BAL | -0.0270 | -0.0366 | -0.0308 | 0.0952 | DROP_KTM_W |

Decisions were made before any test data was accessed.

---

## 13. Final Held-Out Test Results

After decisions were frozen:

| Dataset | MR Baseline | Decision | Final Model | Final Test MF1 | Final Test Acc | Final Test WF1 | Δ |
|---|---:|---|---|---:|---:|---:|---:|
| ECG5000_UNBAL | 0.5938 | KEEP_KTM_W | MR + KTM-W | 0.6115 | 0.9570 | 0.9488 | +0.0177 |
| ECG5000_BAL | 0.6553 | DROP_KTM_W | MR | 0.6553 | 0.9520 | 0.9498 | 0.0000 |
| CWRU_UNBAL | 0.9917 | DROP_KTM_W | MR | 0.9917 | 0.9917 | 0.9917 | 0.0000 |
| CWRU_BAL | 0.9947 | DROP_KTM_W | MR | 0.9947 | 0.9947 | 0.9947 | 0.0000 |

The final model was refit on train+val and evaluated once on held-out test data. The decision was not revised after seeing test results.

**Verification of canonical baselines:**
- ECG5000_UNBAL: 0.5938 (matches reference 0.5938)
- ECG5000_BAL: 0.6553 (matches reference 0.6553)
- CWRU_UNBAL: 0.9917 (matches reference 0.9917)
- CWRU_BAL: 0.9947 (matches reference 0.9947)

---

## 14. Computational Cost

| Dataset | MR Fit | Top-K Selection | KTM Extract | Null Build | Total |
|---|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | 0.5s | 4.3s | 3.4s | 110.3s | 145.8s |
| ECG5000_BAL | 0.2s | 12.0s | 4.7s | 280.7s | 362.7s |
| CWRU_UNBAL | 0.3s | 0.5s | 3.6s | 24.1s | 31.9s |
| CWRU_BAL | 0.7s | 2.6s | 8.4s | 85.2s | 111.3s |
| **Total** | **1.7s** | **19.4s** | **20.1s** | **500.3s** | **651.7s (~10.9 min)** |

The bottleneck is the null construction (80 total calls to `ktm_transform` per dataset: 20 shifts × batched computation). The MR fit and KTM extraction are fast.

---

## 15. Interpretation

### On ECG5000_UNBAL
Real KTM-W carries alignment-dependent timing information that is statistically distinguishable from circularly-shifted nulls (p = 0.0476). The circular shift null dramatically degrades performance (null mean Δ = -0.3831), indicating that the positional alignment of activations carries substantial information. The real KTM-W provides a modest but genuine positive improvement (+0.0035 on validation, +0.0177 on test).

### On ECG5000_BAL
The null is also statistically distinguishable (p = 0.0476), but the real Δ is negative (-0.0022). This means that while the alignment matters, KTM-W slightly hurts performance on the balanced variant. The decision rule correctly drops KTM-W.

### On CWRU
KTM-W is not distinguishable from shifted nulls on either balanced or unbalanced CWRU. The null distributions overlap substantially with the real Δ. Both CWRU datasets are near ceiling (MF1 ≈ 0.99), leaving little room for improvement from timing features.

### Pattern
The null distributions reveal an important asymmetry:
- On ECG (shorter signals, T=140), circular shifting causes catastrophic performance collapse (null means of -0.38 and -0.67), suggesting MiniROCKET's activation positions are highly sensitive to temporal alignment.
- On CWRU (longer signals, T=1024), circular shifting causes mild degradation (null means of -0.03 and -0.04), suggesting activation positions are less alignment-dependent.

---

## 16. Limitations

1. **Only four datasets.** Results cannot support claims of universal superiority or inferiority.

2. **K = 84 kernel groups.** With only 84 kernel configurations in MiniRocket, K_TOP = 1024 does not truncate. This means the "top-K selection" step effectively selects all features. The timing statistics are computed for all 9996 features.

3. **KTM-W is per-feature, not per-kernel.** Despite the specification describing one W per kernel, the implementation computes one W per feature (each with its own dilation/bias). This provides 9996 timing features rather than 84. The null test is still valid but the interpretation of "timing per kernel" does not strictly hold.

4. **p = 0.0476 is marginally significant.** With N_NULL_SHIFTS = 20, the smallest achievable p-value is 1/21 ≈ 0.0476. ECG5000_UNBAL sits exactly at this floor. More null repetitions would provide finer resolution.

5. **The null tests alignment, not causality.** A positive result shows that alignment-dependent information exists, not that it causes the improvement. The improvement could be a side effect of how Ridge regularization interacts with correlated features.

6. **The balanced/unbalanced split matters.** ECG5000_UNBAL keeps KTM-W but ECG5000_BAL drops it, suggesting dataset-specific effects that cannot be generalized.

---

## 17. Final Conclusion

### Scientific Findings

1. **Does real KTM-W provide validation improvement beyond its circular-shift null?**

   Yes, on **ECG5000_UNBAL** (p = 0.0476, Δ_real = +0.0035 > 0). The alignment of activation timing with the signal window carries statistically significant predictive information.

   On **ECG5000_BAL**, the null is also significant (p = 0.0476) but Δ_real is negative (-0.0022), so KTM-W hurts performance despite the null being distinguishable.

   On **CWRU_UNBAL** and **CWRU_BAL**, KTM-W is not distinguishable from the shifted null (p = 0.3333 and p = 0.0952 respectively).

2. **On which datasets?**

   Only ECG5000_UNBAL shows a positive, statistically significant result. ECG5000_BAL shows significance but with negative Δ. Both CWRU datasets show no significance.

3. **Does the pre-registered keep/drop rule select KTM-W for ECG?**

   For **ECG5000_UNBAL**: Yes — KEEP_KTM_W (p = 0.0476 < 0.05, Δ_real = +0.0035 > 0).
   For **ECG5000_BAL**: No — DROP_KTM_W (Δ_real = -0.0022 ≤ 0, failing the positive-improvement criterion).

4. **Does it reject KTM-W for CWRU?**

   Yes — DROP_KTM_W on both CWRU_UNBAL (p = 0.3333) and CWRU_BAL (p = 0.0952), consistent with hypothesis.

5. **Does the final held-out test result support or contradict the validation decision?**

   **Supports.** The KEEP decision for ECG5000_UNBAL produced a test improvement of +0.0177 MF1 (0.5938 → 0.6115), confirming the validation finding. The DROP decisions for all other datasets resulted in unchanged test performance, as expected.

### Defensible Claim

> Under this validation-only alignment test, KTM-W was distinguishable from its circular-shift null on ECG5000_UNBAL (p = 0.0476) but not on ECG5000_BAL (negative Δ), CWRU_UNBAL (p = 0.3333), or CWRU_BAL (p = 0.0952). The keep/drop decision selected KTM-W for ECG5000_UNBAL and rejected it for the other three datasets. The held-out test result confirmed the validation decision, with a +0.0177 MF1 improvement on ECG5000_UNBAL and no change on the other datasets.

### This Is the Final Timing Experiment

No further KTM variants, CPT variants, learned phase, learned timing, period estimators, routers, or timing architectures will be created from this line of inquiry. The KTM/timing experimental program is complete.
