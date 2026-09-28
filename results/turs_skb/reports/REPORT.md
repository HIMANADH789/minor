# TURS-SKB - Structured Kernel Bank: Final Report

Run completed 2026-09-14 09:15 | seed 42 primary | canonical aeon MiniROCKET 10K base | outputs: results/turs_skb/

## 1. Executive summary

- M0 canonical reproduced the reference benchmark on all 4 datasets (max |delta| 0.0000 <= tol 0.02).

- M6 (MR + all five structured families, 12709 features vs 9996) mean Macro-F1 0.7809 vs M0 0.8089 (mean delta -0.0280; 0/4 datasets positive; 0 FDR-significant dataset-level M6 improvements).

- Best single family by mean delta: **energy** (-0.0027).

- Final classification: **NEGATIVE**.

## 2. Research question

Can mathematically structured fixed temporal kernels provide complementary information to canonical MiniROCKET while retaining its efficient fixed-feature + linear-readout framework? The novelty tested is ONLY the kernel family; the downstream pipeline is unchanged.

## 3. Why MiniROCKET kernels may be limiting

Canonical MiniROCKET draws sparse two-valued kernels (groups of +1 and -1 taps) with random lengths {7,9,11} and data-driven dilation; its bias set is 9 train-response quantiles. This family is random, sparse, two-valued, and optimized for generic shape matching - it does not deliberately encode smooth bump morphology (F1), exact finite-difference operators (F2), analytic wavelets (F3), band-localized oscillations (F4), or burst energy (F5). If those structured function classes carry predictive signal that the sparse family misses, a fixed structured bank should add it.

## 4. Mathematical definition of each family

- **F1 Morphological/shape**: unit-L2 Gaussian bumps `k(u)=exp(-u^2/2s^2)`, difference-of-Gaussians `k(u)=exp(-u^2/2s1^2)-a*exp(-u^2/2s2^2)` (s2=s1*1.6), and Gaussian-derivative operators (1st/2nd). Grid: sigma in [1.0, 2.0, 4.0, 8.0, 16.0] x dilation in [1, 2, 4, 8] x polarity for bumps; alpha in [0.5, 1.0, 1.5] for DoG; orders [1, 2].

- **F2 Derivative**: discrete operators `[-1,0,+1]` (1st) and `[+1,-2,+1]` (2nd), unit-L2, dilations (1, 2, 4, 8, 16, 32, 64, 128), both polarities. Deliberately small (32 kernels) to avoid feature-count dominance.

- **F3 Wavelet**: `k_a(u) = a^{-1/2} psi(u/a)` discretized for psi in {Haar, Mexican hat, Morlet} x a in [1, 2, 4, 8] x dilations {1..128} x polarity (192 kernels).

- **F4 Gabor**: `k(u)=exp(-u^2/(2s^2)) cos(2 pi f u + phi)`, f in [0.125, 0.0625, 0.03125] (periods 8/16/32 taps), sigma in [2.0, 4.0, 8.0, 16.0], phi in [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469] (sine quadrature via phase), dilations [1, 2, 4, 8] (192 kernels).

- **F5 Energy/burst (NONLINEAR)**: `B(t) = E_w(t) - lam*E_4w(t)` with `E_w(t) = sum_j q_j x_{t+j}^2`, q = sum-normalized tent window (NOT a signed linear convolution; documented as a nonlinear energy operator). Grid: w in [4, 8, 16, 32] x lam in [0.0, 0.5, 1.0] (12 operators).

## 5. Exact kernel-generation procedure

All banks are built from the predefined grids above with NO randomness, NO fitting, and NO label access; every base kernel is unit-L2-normalized; dilation is recorded per bank entry and capped at fit time to the largest power of two that fits the series length (aeon's clip rule). Identical (kernel, effective-dilation) pairs are collapsed and counted, never duplicated. Bank counts: morphological=140, derivative=32, wavelet=192, gabor=192, energy=12.

## 6. Feature extraction

Response `r_k(t) = sum_j k_j x_{t + j d}` (kernel taps strided by dilation d, evaluated at every valid position) - identical response model to MiniROCKET. Each kernel yields 9 PPV features against its 9 golden-ratio quantile biases (the aeon MiniROCKET threshold abstraction, mirrored exactly): `PPV = mean_t 1[r(t) > b]`, biases fitted on TRAIN responses only. No mu/W/variance/phase features (spec sec. 12).

## 7. Standardization

MR block: RAW (bit-identical to canonical M0 features - the canonical baseline itself applies no scaler to MiniROCKET PPV features; fairness requires M1-M6 to inherit that exact block). Each structured family block: z-standardized with TRAIN-only statistics (BlockStandardizer, near-zero-variance columns left unscaled). All blocks verified finite with mean/std/min/max logged (sanity_blocks in full_results.json).

## 8. Ridge protocol

RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)) fitted on TRAIN+VAL (canonical baseline_bench protocol), single TEST evaluation. Screening fits use TRAIN-only ridge evaluated on VAL and never touch TEST.

## 9. Dataset protocol

Canonical four-dataset benchmark split (train_test_split seed 42, stratified; per-signal z-norm). Reference values: ECG5000_UNBAL=0.5938, ECG5000_BAL=0.6553, CWRU_UNBAL=0.9917, CWRU_BAL=0.9947.

## 10. Experimental variants

| Model | Blocks | Feature dim (mean) |
|---|---|---:|
| M0 | MR | 9996 |
| M1 | MR + morphological | 10873 |
| M2 | MR + derivative | 10131 |
| M3 | MR + wavelet | 10455 |
| M4 | MR + gabor | 11130 |
| M5 | MR + energy | 10104 |
| M6 | MR + all five | 12709 |

## 11. Main results (test Macro-F1, seed 42)

| Dataset | MR-10K | +Morph | +Deriv | +Wavelet | +Gabor | +Energy | +All5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | 0.5938 | 0.5963 | 0.5874 | 0.6058 | 0.5919 | 0.5958 | 0.5736 |
| ECG5000_BAL | 0.6553 | 0.6382 | 0.6516 | 0.6013 | 0.6118 | 0.6484 | 0.6074 |
| CWRU_UNBAL | 0.9917 | 0.9750 | 0.9833 | 0.9791 | 0.9457 | 0.9875 | 0.9583 |
| CWRU_BAL | 0.9947 | 0.9930 | 0.9982 | 0.9930 | 0.9912 | 0.9930 | 0.9842 |

## 12. Per-family gains (dMF1 vs M0)

| Dataset | dMorph | dDeriv | dWavelet | dGabor | dEnergy | dAll5 |
|---|---:|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | +0.0025 | -0.0064 | +0.0120 | -0.0019 | +0.0020 | -0.0202 |
| ECG5000_BAL | -0.0171 | -0.0037 | -0.0540 | -0.0435 | -0.0069 | -0.0479 |
| CWRU_UNBAL | -0.0167 | -0.0084 | -0.0126 | -0.0460 | -0.0042 | -0.0334 |
| CWRU_BAL | -0.0017 | +0.0035 | -0.0017 | -0.0035 | -0.0017 | -0.0105 |

Mean over datasets: dMorph=-0.0083, dDeriv=-0.0038, dWavelet=-0.0141, dGabor=-0.0237, dEnergy=-0.0027, dAll5=-0.0280 (median -0.0268).

Pairwise (screening-gated) results: none run.

## 13. Statistical inference (paired McNemar on held-out test correctness)

| Comparison | Dataset | dMF1 | p | q(FDR) | Effect size | Verdict |
|---|---|---:|---:|---:|---:|---|
| M0 vs M1_Morph | ECG5000_UNBAL | +0.0025 | 0.7237 | 0.8684 |  | not significant |
| M0 vs M2_Deriv | ECG5000_UNBAL | -0.0064 | 0.4795 | 0.8684 |  | not significant |
| M0 vs M3_Wavelet | ECG5000_UNBAL | +0.0120 | 0.6171 | 0.8684 |  | not significant |
| M0 vs M4_Gabor | ECG5000_UNBAL | -0.0019 | 0.6831 | 0.8684 |  | not significant |
| M0 vs M5_Energy | ECG5000_UNBAL | +0.0020 | 1 | 1 |  | not significant |
| M0 vs M6_ALL5 | ECG5000_UNBAL | -0.0202 | 0.7237 | 0.8684 | -3.3226 | not significant |
| M0 vs M1_Morph | ECG5000_BAL | -0.0171 | 0.1824 | 0.5473 |  | not significant |
| M0 vs M2_Deriv | ECG5000_BAL | -0.0037 | 0.6171 | 0.9022 |  | not significant |
| M0 vs M3_Wavelet | ECG5000_BAL | -0.0540 | 0.02334 | 0.1401 |  | not significant |
| M0 vs M4_Gabor | ECG5000_BAL | -0.0435 | 0.7518 | 0.9022 |  | not significant |
| M0 vs M5_Energy | ECG5000_BAL | -0.0069 | 1 | 1 |  | not significant |
| M0 vs M6_ALL5 | ECG5000_BAL | -0.0479 | 0.4795 | 0.9022 | -4.2035 | not significant |
| M0 vs M1_Morph | CWRU_UNBAL | -0.0167 | 0.2888 | 0.5566 |  | not significant |
| M0 vs M2_Deriv | CWRU_UNBAL | -0.0084 | 0.6171 | 0.7405 |  | not significant |
| M0 vs M3_Wavelet | CWRU_UNBAL | -0.0126 | 0.3711 | 0.5566 |  | not significant |
| M0 vs M4_Gabor | CWRU_UNBAL | -0.0460 | 0.005546 | 0.03327 |  | significant-but-negative |
| M0 vs M5_Energy | CWRU_UNBAL | -0.0042 | 1 | 1 |  | not significant |
| M0 vs M6_ALL5 | CWRU_UNBAL | -0.0334 | 0.02686 | 0.08057 | -8.4438 | not significant |
| M0 vs M1_Morph | CWRU_BAL | -0.0017 | 1 | 1 |  | not significant |
| M0 vs M2_Deriv | CWRU_BAL | +0.0035 | 0.4795 | 1 |  | not significant |
| M0 vs M3_Wavelet | CWRU_BAL | -0.0017 | 1 | 1 |  | not significant |
| M0 vs M4_Gabor | CWRU_BAL | -0.0035 | 0.6171 | 1 |  | not significant |
| M0 vs M5_Energy | CWRU_BAL | -0.0017 | 1 | 1 |  | not significant |
| M0 vs M6_ALL5 | CWRU_BAL | -0.0105 | 0.0771 | 0.4626 | -192.4332 | not significant |

Six primary comparisons per dataset (M1-M5, M6) corrected with BH-FDR within dataset; screening-gated pairwise tests corrected separately. Effect size for the primary M0-vs-M6 comparison is Cohen's d over the 5-seed MF1 pairs (sec. 18).

## 14. Multiple-comparison correction

BH-FDR at alpha=0.05. Verdicts above use q, not raw p. No family is declared significant on raw p alone.

## 15. Representation diversity

| Family | Feature count (mean) | CKA vs MR (mean) | Mean |corr| vs MR (mean) |
|---|---:|---:|---:|
| morphological | 877 | 0.601 | 0.310 |
| derivative | 135 | 0.903 | 0.391 |
| wavelet | 459 | 0.588 | 0.350 |
| gabor | 1134 | 0.484 | 0.277 |
| energy | 108 | 0.547 | 0.332 |

CKA is a DIVERSITY diagnostic only: low CKA means the block is mathematically distinct, not that it is useful; predictive increment is judged in sections 12-13.

## 16. Kernel-response diagnostics

Figure 5 (fig5_kernel_responses.png / _energy.png) shows the pre-registered representative kernels' responses on one TRAIN ECG and one TRAIN CWRU signal (selected before any test result was seen; no cherry-picking). Mechanistic expectations (pre-registered) vs outcome:

- morphological (shape-rich ECG beat morphology): pre-registered target datasets ['ECG5000_UNBAL', 'ECG5000_BAL']; observed mean delta on targets -0.0073, elsewhere -0.0092 -> **REJECTED**.
- derivative (transitions/edges (QRS edges, fault steps)): pre-registered target datasets ['ECG5000_UNBAL', 'ECG5000_BAL', 'CWRU_UNBAL', 'CWRU_BAL']; observed mean delta on targets -0.0038, elsewhere +nan -> **REJECTED (single-domain family)**.
- wavelet (multiscale morphology of ECG beats): pre-registered target datasets ['ECG5000_UNBAL', 'ECG5000_BAL']; observed mean delta on targets -0.0210, elsewhere -0.0072 -> **REJECTED**.
- gabor (oscillatory vibration structure of bearings): pre-registered target datasets ['CWRU_UNBAL', 'CWRU_BAL']; observed mean delta on targets -0.0248, elsewhere -0.0227 -> **REJECTED**.
- energy (impulsive/burst fault energy): pre-registered target datasets ['CWRU_UNBAL', 'CWRU_BAL']; observed mean delta on targets -0.0029, elsewhere -0.0024 -> **REJECTED**.

## 17. Computational cost

| Dataset | Model | Feature dim | Ridge fit (s) | Inference (s) | SKB extract (s) |
|---|---|---:|---:|---:|---:|
| ECG5000_UNBAL | M0 | 9996 | 12.6 | 0.01 | - |
| ECG5000_UNBAL | M6 | 11787 | 12.4 | 0.01 | 1.1 |
| ECG5000_BAL | M0 | 9996 | 38.9 | 0.02 | - |
| ECG5000_BAL | M6 | 11787 | 42.2 | 0.01 | 1.2 |
| CWRU_UNBAL | M0 | 9996 | 1.2 | 0.00 | - |
| CWRU_UNBAL | M6 | 13632 | 1.3 | 0.01 | 7.0 |
| CWRU_BAL | M0 | 9996 | 6.9 | 0.01 | - |
| CWRU_BAL | M6 | 13632 | 12.3 | 0.01 | 31.3 |

Feature-extraction cost of the full structured bank is seconds-scale per split; the ridge fit on the enlarged matrix dominates. Gain per added feature is reported per dataset in feature_dimensions.csv (delta MF1 per 1K added features).

## 18. Seed robustness (M0 vs M6, seeds 42-46)

| Dataset | mean d | std | median | min | max | n_pos | n_neg | permutation p | Cohen's d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | -0.0226 | 0.0068 | -0.0202 | -0.0321 | -0.0162 | 0 | 5 | 0.05647 | -3.323 |
| ECG5000_BAL | -0.0589 | 0.0140 | -0.0601 | -0.0729 | -0.0417 | 0 | 5 | 0.06597 | -4.204 |
| CWRU_UNBAL | -0.0317 | 0.0038 | -0.0334 | -0.0334 | -0.0250 | 0 | 5 | 0.06797 | -8.444 |
| CWRU_BAL | -0.0105 | 0.0001 | -0.0105 | -0.0106 | -0.0105 | 0 | 5 | 0.06497 | -192.433 |

## 19. Failure analysis

Families with negative increments: morphological: negative delta on 3/4 datasets; derivative: negative delta on 3/4 datasets; wavelet: negative delta on 3/4 datasets; gabor: negative delta on 4/4 datasets; energy: negative delta on 3/4 datasets. Structured blocks can shift the ridge solution away from the canonical optimum when their added features are noise-dominated at near-ceiling accuracy (CWRU) or when family scale dominates after standardization.

## 20. Limitations

- Single canonical split per dataset for primary results (5-seed robustness covers MiniROCKET draw variance only; the split itself is fixed by the benchmark protocol).
- Family feature counts differ by design (deriv/energy smaller, documented per spec sec. 13); per-family comparisons therefore conflate family identity with feature budget, mitigated by the matched ~9-features-per-kernel abstraction and the M6 gain-per-feature accounting.
- Biases are train-quantile-fitted (MiniROCKET's own mechanism); a fully predefined deterministic threshold convention was the alternative and is documented as such (spec sec. 6).
- The energy family is a nonlinear operator; its comparison to linear families is by feature semantics, not by convolution algebra.

## 21. Final scientific conclusion

The structured kernel bank slightly DEGRADES canonical MiniROCKET performance; the hypothesis is not supported.

## Final decision-rule classification (sec. 44)

**NEGATIVE**

## Final scientific questions

- **Q1** Does any structured family significantly improve over canonical MiniROCKET? No - no family improves canonical MiniROCKET.

- **Q2** Which family is most useful? **energy** (mean delta -0.0027).

- **Q3** Consistent across ECG and CWRU? morphological: ECG -0.0073 vs CWRU -0.0092 (consistent); derivative: ECG -0.0051 vs CWRU -0.0024 (consistent); wavelet: ECG -0.0210 vs CWRU -0.0072 (consistent); gabor: ECG -0.0227 vs CWRU -0.0248 (consistent); energy: ECG -0.0024 vs CWRU -0.0029 (consistent).

- **Q4** Genuinely different representations? Mean CKA vs MR: morphological=0.601, derivative=0.903, wavelet=0.588, gabor=0.484, energy=0.547 (distinct blocks; usefulness judged by predictive increment).

- **Q5** Additive benefit from combining families? dAll5=-0.0280 vs best single family -0.0027.

- **Q6** Justified by cost? Feature dim 9996 -> 12709 (~1.3x) for -0.0280 mean MF1; gain per 1K added features: ECG5000_UNBAL=-0.011, ECG5000_BAL=-0.027, CWRU_UNBAL=-0.009, CWRU_BAL=-0.003.

- **Q7** M6 vs M0 under controlled evaluation? Mean 0.7809 vs 0.8089; 0 dataset(s) FDR-significant; seed-robustness p-values in sec. 18.
