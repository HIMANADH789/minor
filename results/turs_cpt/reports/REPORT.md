# TURS-CPT: Cyclic Phase Timing — Experiment Report

## 1. Executive Summary

We tested whether grounding MiniROCKET kernel activation timing in the signal's own intrinsic cycle (CPT) provides more robust and complementary timing information than absolute window-relative timing (KTM). The experiment was run at two capacities (2K and 10K kernels) on four biomedical datasets.

**Primary result (10K)**: CPT **hurts performance on ALL four datasets**, and more severely than KTM. Mean Δ10K = -0.0453 (vs KTM's -0.0060). The degradation is catastrophic on CWRU (-0.0751 UNBAL, -0.0636 BAL) and significant on ECG5000 (-0.0218 UNBAL, -0.0206 BAL).

**Decomposition**: Phase features (concentration |R| and circular mean angle) are the primary source of harm. MR+phase alone drops CWRU_UNBAL from 0.9917 to 0.8293 (-0.1624). Concentration alone is less harmful but still negative.

**Verdict**: The hypothesis that cycle-anchored phase timing would be more robust than absolute timing is **NOT SUPPORTED**. CPT provides no useful timing information for these datasets and degrades performance more than KTM.

## 2. Motivation

MiniROCKET's PPV pooling discards temporal position information. KTM (absolute timing) showed modest benefits on ECG but harm on CWRU, suspected to be due to boundary dependence. CPT tests whether intrinsic-cycle phase anchoring removes this boundary dependence.

## 3. KTM Failure Being Addressed

KTM (μ/W) showed:
- ECG5000: +0.0068 to +0.0065 (modest benefit)
- CWRU: -0.0250 to -0.0124 (harm)
- Mean: -0.0060

The hypothesis was that absolute timing is anchored to arbitrary recording boundaries, and cyclic phase would be more robust.

## 4. CPT Architecture

For each kernel m with activations τ_1,...,τ_k:
1. Estimate intrinsic period P from the signal (autocorrelation primary, FFT fallback)
2. Compute cyclic phase φ_i = (τ_i mod P) / P
3. Compute resultant vector R_m = (1/k) Σ exp(2πiφ_i)
4. Extract concentration C_m = |R_m| and circular mean Θ_m = angle(R_m)

Top-K=1024 kernels selected by discriminative power on training data.

## 5. Mathematical Formulation

For active positions τ_i with estimated period P:

φ_i = (τ_i mod P) / P

z_i = exp(2πiφ_i)

R_m = (1/k) Σ_i z_i

C_m = |R_m| ∈ [0, 1] (phase concentration)

Θ_m = angle(R_m) → normalized: (angle mod 2π)/(2π) ∈ [0, 1)

Edge cases: k=0 → C=0, Θ=0; k=1 → C=1, Θ=phase of single activation.

## 6. Period Estimator

**Primary**: Autocorrelation with peak detection (prominence > 0.05, first prominent peak).

**Fallback**: Dominant FFT frequency (ignore DC, valid range 2 ≤ P ≤ T/2).

**Final fallback**: Median half-cycle from zero-crossings, or T/4.

One deterministic procedure across all datasets. No class labels used.

Period estimates:
- ECG5000: mean P≈40 (ACF 70%, plausible for ECG signals)
- CWRU: mean P≈15 (ACF 100%, but T=1024 — period seems too short)

## 7. Top-K Selection

Method: ANOVA F-score on training data only (no test leakage).

K=1024 fixed across all datasets.

Selected kernels:
- ECG5000: 336 kernels at 2K, 1024 at 10K
- CWRU: 480 kernels at 2K, 1024 at 10K

## 8. MiniROCKET Implementation

aeon 1.2.0 `MiniRocket(random_state=42, n_jobs=-1)`. CPT uses the same activation mask as MiniROCKET PPV (verified by count consistency in unit tests).

## 9. Dataset Protocol

Same as KTM: canonical benchmark_baselines.py splits, per-sample z-normalization, seed=42.

## 10. Standardization

Three blocks standardized separately (fit on train+val):
- F_MR: raw PPV (no standardization, matching canonical)
- F_concentration: z-scored
- F_phase: z-scored

## 11. Ridge Protocol

RidgeClassifierCV(alphas=np.logspace(-4,4,20)) on train+val, macro-F1 on test.

## 12. Leakage Audit

- Period estimation: data-only, no labels
- Top-K selection: training data only
- Standardization: fit on train+val (same data Ridge sees)
- No test data influence on any preprocessing

## 13. Unit-Test Results

19 tests passing:
- Period estimation (ACF detection, FFT fallback, never-None)
- Phase computation (zero/single activations, range, concentration)
- CPT transform (no NaN, range, shape, count consistency with KTM)
- Top-K selection (training-only, deterministic)
- Period summary

## 14. 2K Results

| Dataset | MR-2K | MR-2K+CPT | Δ |
|---|---|---|---|
| ECG5000_UNBAL | 0.5938 | 0.5624 | **-0.0314** |
| ECG5000_BAL | 0.6376 | 0.6016 | **-0.0360** |
| CWRU_UNBAL | 0.9542 | 0.8880 | **-0.0662** |
| CWRU_BAL | 0.9842 | 0.9701 | **-0.0141** |
| **Mean** | **0.7924** | **0.7555** | **-0.0369** |

## 15. 10K Results

| Dataset | MR-10K | MR-10K+CPT | Δ |
|---|---|---|---|
| ECG5000_UNBAL | 0.5938 | 0.5720 | **-0.0218** |
| ECG5000_BAL | 0.6553 | 0.6347 | **-0.0206** |
| CWRU_UNBAL | 0.9917 | 0.9166 | **-0.0751** |
| CWRU_BAL | 0.9947 | 0.9311 | **-0.0636** |
| **Mean** | **0.8089** | **0.7636** | **-0.0453** |

## 16. Decomposition (concentration-only / phase-only)

| Dataset | MR | MR+conc | MR+phase | MR+conc+phase |
|---|---|---|---|---|
| ECG5000_UNBAL | 0.5938 | 0.5851 | 0.5515 | 0.5720 |
| ECG5000_BAL | 0.6553 | 0.6494 | 0.5910 | 0.6347 |
| CWRU_UNBAL | 0.9917 | 0.9625 | 0.8293 | 0.9166 |
| CWRU_BAL | 0.9947 | 0.9647 | 0.8728 | 0.9311 |

**Phase features are the primary source of harm**: MR+phase alone drops CWRU_UNBAL from 0.9917 to 0.8293 (-0.1624). Concentration alone is less harmful but still negative (-0.0059 to -0.0292).

## 17. Period Quality Analysis

- ECG5000: 70% ACF-estimated, mean P≈40. Period estimates appear plausible.
- CWRU: 100% ACF-estimated, mean P≈15. Period seems too short for T=1024 signals — may be detecting high-frequency noise rather than true periodicity.

The short estimated period for CWRU may explain the catastrophic phase degradation: phase features computed relative to an unreliable period are pure noise.

## 18. ECG vs CWRU Analysis

**ECG5000**: CPT hurts by -0.0206 to -0.0218. Worse than KTM's +0.0065 to +0.0068. Phase anchoring does NOT help ECG classification.

**CWRU**: CPT hurts by -0.0636 to -0.0751. Much worse than KTM's -0.0124 to -0.0250. Phase anchoring makes CWRU classification WORSE.

**Hypothesis evaluation**: The prediction that CPT would improve ECG and reduce CWRU harm is **REFUTED**. CPT hurts both datasets, and hurts CWRU more than KTM.

## 19. Computational Cost

CPT adds:
- Period estimation: ~0.1-0.3s per dataset (parallelizable)
- Kernel ranking: ~0.1s
- CPT extraction: ~1-3s per dataset (similar to KTM)
- Total overhead: ~2-5s per dataset

The overhead is modest in absolute terms but the features are harmful.

## 20. Limitations

1. **Period estimation may be unreliable**: The ACF/FFT approach may not capture the true intrinsic period for all signals, especially CWRU with T=1024.

2. **Phase representation may be inappropriate**: The concentration/circular-mean formulation may not capture useful timing information even with perfect period estimates.

3. **Top-K selection**: Discriminative ranking on training data may not select the right kernels for phase-based timing.

4. **Single seed / single split**: No statistical significance testing.

5. **Linear classifier only**: RidgeClassifierCV may not extract value from non-linear phase relationships.

## 21. Final Conclusion

**Does cycle-anchored kernel activation timing recover useful temporal information while avoiding the boundary-dependent failure of absolute timing?**

**Answer: NO.**

CPT provides **no useful timing information** for these datasets and degrades performance more severely than KTM. The hypothesis that intrinsic-cycle phase would be more robust than absolute timing is **NOT SUPPORTED**.

**Classification**:
- **Predictive value beyond full MiniROCKET**: NEGATIVE
- **Phase robustness**: NOT SUPPORTED
- **Cross-dataset consistency**: STRONG (consistently negative on all 4 datasets)

**The experiment is a clear, informative negative result.** Cycle-anchored phase timing does not solve the timing problem identified by KTM. The degradation is worse on CWRU (where period estimates appear unreliable) and also negative on ECG5000 (where period estimates appear plausible). This suggests that the issue is not merely boundary dependence but a fundamental limitation of phase-based timing features for these datasets.

**Possible explanations**:
1. Period estimation is unreliable for these signal types
2. Phase features are inherently noisy/uninformative for these classification tasks
3. MiniROCKET's PPV features already capture enough temporal information implicitly
4. The timing information is simply not useful for these specific datasets

The experiment definitively shows that **intrinsic-cycle phase timing is not a viable supplement to MiniROCKET** under this formulation.

---

*Experiment conducted with aeon 1.2.0, scikit-learn 1.6.1, numba 0.61.2, numpy 2.2.6 on Windows.*
*Protocol: canonical benchmark_baselines.py splits, MiniRocket(random_state=42), RidgeClassifierCV(alphas=logspace(-4,4,20)).*
