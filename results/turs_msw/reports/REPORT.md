# TURS-MSW — Multi-Scale Windowed MiniROCKET: Final Report

Generated 2026-09-14 14:47 | seed 42 | datasets: EpilepticSeizures, Haptics, Phoneme | primary metric: TEST Macro-F1

## 1. Executive summary

- **EpilepticSeizures**: M0=0.9194, M1=0.9246 (delta +0.0052), M2=0.9245 (delta +0.0051), M3=0.9233 (delta +0.0039)
- **Haptics**: M0=0.4974, M1=0.4883 (delta -0.0091), M2=0.4633 (delta -0.0341), M3=0.4825 (delta -0.0149)
- **Phoneme**: M0=0.0808, M1=0.0881 (delta +0.0073), M2=0.0876 (delta +0.0068), M3=0.0938 (delta +0.0130)
- M3 (TURS-MSW) improves on 2/3 datasets; 2 FDR-significant improvements.

## 2. Research question

Does preserving coarse temporal locality before PPV pooling provide predictive information beyond canonical full-capacity MiniROCKET? The kernel bank, biases, dilations, responses and activation masks are EXACTLY the canonical fitted MiniRocket's; only the pooling regions change.

## 3. MiniROCKET baseline (verified)

| dataset | M0 val MF1 | M0 test MF1 | alpha | F | equivalence max delta (train/val/test) |
|---|---:|---:|---:|---:|---|
| EpilepticSeizures | 0.9499 | 0.9194 | 11.29 | 9996 | 0.00e+00/0.00e+00/0.00e+00 |
| Haptics | 0.5879 | 0.4974 | 11.29 | 9996 | 0.00e+00/0.00e+00/0.00e+00 |
| Phoneme | 0.1035 | 0.0808 | 206.9 | 9996 | 0.00e+00/0.00e+00/0.00e+00 |

The global block of the regional extractor is bit-identical to the canonical aeon output on every split (max delta < 1e-12), satisfying the sec. 22 equivalence requirement. M0 was additionally verified against the frozen external-generalization baseline (results/external_stack_generalization), tolerance 5e-3 (sec. 21).

## 4-10. Motivation, formulation, windows, extraction, protocol

- **Motivation**: canonical PPV destroys WHERE activations occur; regional PPVs retain coarse position information.
- **Formulation** (sec. 40): per kernel k, activation I_k(t)=1[r_k(t)>b_k]; global P = mean over [0,T); medium P_{k,m} = mean over M_m; local P_{k,l} = mean over L_l; per-feature vector [P_G, P_M1, P_M2, P_L1..P_L4]; features concatenated scale-major, RidgeClassifierCV unchanged.
- **Windows** (deterministic, fractions of T): medium w=ceil(3T/4), regions [0,w), [T-w,T) (~50% neighbour overlap, full coverage); local w=ceil(T/2), stride floor(T/4), regions [j*stride, min(j*stride+w, T)) for j=0..3 (50% overlap, full coverage). For padding1==1 kernels, windows are mapped onto aeon's valid axis C[padding:T-padding] (same responses, re-indexed).
- **Extraction**: verbatim mirror of aeon's `_static_transform_uni` (same C_alpha/C_gamma accumulation, same padding parity), with prefix-sum activation counts for O(1) window means. Responses are computed ONCE per split; no per-window recomputation (sec. 19). Test features are processed in chunks (sec. 36) and never materialized per variant.
- **Standardization**: all MSW features are PPV rates in [0,1] — the same natural scale as canonical M0 — so the canonical RAW ridge protocol is retained for every variant (no scaling fitted on any split). Block statistics reported below.
- **Ridge protocol**: RidgeClassifierCV(alphas=logspace(-4,4,20)); train-only fit -> val MF1; final fit TRAIN+VAL; single test eval.
- **Dataset protocol**: canonical UCR splits from the audited external-generalization loaders; val = provided canonical val (EpilepticSeizures) or deterministic per-class stratified 15% of train, seed 42 (Haptics, Phoneme); per-sample z-norm; test untouched until final evaluation.

Block statistics (train, raw):

| block | min | max | mean | std | near-zero-var cols | NaN | Inf |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | 0.0 | 1.0 | 0.500116 | 0.290273 | 8 | 0 | 0 |
| medium | 0.0 | 1.0 | 0.499944 | 0.288139 | 16 | 0 | 0 |
| local | 0.0 | 1.0 | 0.489059 | 0.303172 | 950 | 0 | 0 |

## 11. Experimental variants

| variant | blocks | features (per dataset) |
|---|---|---|
| M0 | global | EpilepticSeizures: 9996, Haptics: 9996, Phoneme: 9996 |
| M1 | global+medium | EpilepticSeizures: 29988, Haptics: 29988, Phoneme: 29988 |
| M2 | global+local | EpilepticSeizures: 49980, Haptics: 49980, Phoneme: 49980 |
| M3 | global+medium+local | EpilepticSeizures: 69972, Haptics: 69972, Phoneme: 69972 |
| M4 | medium+local (diag) | EpilepticSeizures: 59976, Haptics: 59976, Phoneme: 59976 |

## 12. Main results (test Macro-F1)

| Dataset | M0 MR-10K | M1 G+Med | M2 G+Loc | M3 G+Med+Loc | M4 Med+Loc |
|---|---:|---:|---:|---:|---:|
| EpilepticSeizures | 0.9194 | 0.9246 | 0.9245 | 0.9233 | 0.9245 |
| Haptics | 0.4974 | 0.4883 | 0.4633 | 0.4825 | 0.4774 |
| Phoneme | 0.0808 | 0.0881 | 0.0876 | 0.0938 | 0.0945 |

| Dataset | delta M1 | delta M2 | delta M3 |
|---|---:|---:|---:|
| EpilepticSeizures | +0.0052 | +0.0051 | +0.0039 |
| Haptics | -0.0091 | -0.0341 | -0.0149 |
| Phoneme | +0.0073 | +0.0068 | +0.0130 |

## 13. Statistical inference (McNemar on test correctness; BH-FDR within dataset over M1/M2/M3)

| Dataset | Comparison | delta MF1 | chi2 | p | q | Cohen's d | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| EpilepticSeizures | M0 vs M1 | +0.0052 | 21.2459 | 4e-06 | 1.2e-05 | 0.0444 | SIGNIFICANT |
| EpilepticSeizures | M0 vs M2 | +0.0051 | 6.6928 | 0.00968 | 0.01452 | 0.025 | SIGNIFICANT |
| EpilepticSeizures | M0 vs M3 | +0.0039 | 4.2044 | 0.04032 | 0.04032 | 0.02 | SIGNIFICANT |
| Haptics | M0 vs M1 | -0.0091 | 0.5714 | 0.4497 | 0.4533 | -0.0646 | not significant |
| Haptics | M0 vs M2 | -0.0341 | 2.2273 | 0.1356 | 0.4068 | -0.0975 | not significant |
| Haptics | M0 vs M3 | -0.0149 | 0.5625 | 0.4533 | 0.4533 | -0.057 | not significant |
| Phoneme | M0 vs M1 | +0.0073 | 9.7232 | 0.00182 | 0.00182 | 0.074 | SIGNIFICANT |
| Phoneme | M0 vs M2 | +0.0068 | 10.5274 | 0.001176 | 0.001764 | 0.0763 | SIGNIFICANT |
| Phoneme | M0 vs M3 | +0.0130 | 20.0057 | 8e-06 | 2.3e-05 | 0.105 | SIGNIFICANT |

## 14. Representation complementarity (train)

- **EpilepticSeizures**: CKA(global, medium)=0.991661, CKA(global, local)=0.918335, CKA(medium, local)=0.9159; mean |corr| global-vs-medium=0.216501, global-vs-local=0.192432.
- **Haptics**: CKA(global, medium)=0.972822, CKA(global, local)=0.868514, CKA(medium, local)=0.91678; mean |corr| global-vs-medium=0.267529, global-vs-local=0.243723.
- **Phoneme**: CKA(global, medium)=0.957867, CKA(global, local)=0.582273, CKA(medium, local)=0.728287; mean |corr| global-vs-medium=0.275873, global-vs-local=0.211625.
- Interpretation guard: distinct representations are NOT automatically useful; predictive increment is judged in sections 12-13.

## 15. Temporal-locality sanity test (sec. 28)

Mirrored-burst pair (same activations, opposite location), EpilepticSeizures-fitted kernels: mean |delta PPV| global=0.025331, medium=0.03142, local=0.045596. Regional features change while global stays (near-)invariant — the mechanistic sanity check holds.

## 16. Shift diagnostic + stability (sec. 29/30)

- **EpilepticSeizures** circular shift T/2: mean |delta| global=0.042604, medium=0.05963, local=0.096084 (global should change far less).
  - noise: global=0.002152, regional=0.002356 (diagnostic only).
  - shift1: global=0.004038, regional=0.006384 (diagnostic only).
- **Haptics** circular shift T/2: mean |delta| global=0.052684, medium=0.091315, local=0.161871 (global should change far less).
  - noise: global=0.00662, regional=0.007429 (diagnostic only).
  - shift1: global=0.00272, regional=0.003583 (diagnostic only).
- **Phoneme** circular shift T/2: mean |delta| global=0.015028, medium=0.034461, local=0.07025 (global should change far less).
  - noise: global=0.001341, regional=0.001711 (diagnostic only).
  - shift1: global=0.001032, regional=0.00148 (diagnostic only).

## 17. Class-level analysis

- **EpilepticSeizures**: M0 class F1=[0.8706, 0.9683]; M3 class F1=[0.8771, 0.9696]. M0 acc=0.949, wF1=0.9489, balAcc=0.9179; M3 acc=0.9512, wF1=0.9513, balAcc=0.9241.
- **Haptics**: M0 class F1=[0.2597, 0.5625, 0.5672, 0.518, 0.5797]; M3 class F1=[0.2368, 0.5496, 0.5588, 0.5038, 0.5634]. M0 acc=0.5195, wF1=0.4989, balAcc=0.5182; M3 acc=0.5065, wF1=0.4839, balAcc=0.5055.
- **Phoneme**: M0 class F1=[0.0, 0.0, 0.1071, 0.1266, 0.0, 0.0, 0.0, 0.1, 0.152, 0.1154, 0.0, 0.0308, 0.2029, 0.1988, 0.1818, 0.3174, 0.0, 0.0, 0.0, 0.0, 0.0625, 0.4264, 0.4102, 0.0, 0.3571, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0377, 0.0, 0.3226]; M3 class F1=[0.0, 0.0, 0.1754, 0.2083, 0.0, 0.0, 0.0, 0.0, 0.131, 0.1345, 0.0, 0.1667, 0.1304, 0.2721, 0.1739, 0.3416, 0.0, 0.0, 0.0, 0.0, 0.0488, 0.5033, 0.4092, 0.0, 0.4245, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1154, 0.0, 0.4231]. M0 acc=0.2484, wF1=0.1832, balAcc=0.0977; M3 acc=0.2801, wF1=0.2064, balAcc=0.113.

## 18. Runtime (sec. 33/34)

| dataset | M0 total (s) | M3 total (s) | overhead | M0 dim | M3 dim |
|---|---:|---:|---:|---:|---:|
| EpilepticSeizures | 0.75 | 6.32 | 8.43x | 9996 | 69972 |
| Haptics | 0.24 | 0.8 | 3.33x | 9996 | 69972 |
| Phoneme | 0.62 | 2.5 | 4.03x | 9996 | 69972 |

M3 total includes the shared MiniRocket kernel fit, regional extraction (train+val+test) and its ridge fits; M0 total includes kernel fit, aeon transforms and its ridge fits. Trainable neural parameters added: 0 (Ridge coefficients only; dimension above).

## 19. Seed robustness (M0 vs M3, seeds 42-46)

| dataset | mean delta | std | n_pos | n_neg | min | max |
|---|---:|---:|---:|---:|---:|---:|
| EpilepticSeizures | +0.0037 | 0.0021 | 5 | 0 | +0.0017 | +0.0071 |
| Haptics | -0.0111 | 0.0062 | 0 | 5 | -0.0194 | -0.0031 |
| Phoneme | +0.0124 | 0.0026 | 5 | 0 | +0.0086 | +0.0151 |

## 20. Limitations

- Fixed 2/4-region pyramid is one of many possible locality layouts; no tuning of window counts was performed (by design).
- Feature dimension grows 7x, inflating ridge cost; no capacity-matched control was run (contingent on a substantial M3 gain, per sec. 36).
- Single split per dataset (canonical); multi-seed varies only the MiniRocket draw.

## 21. Final conclusion

- Classification: **MODERATE** (M3 positive on 2/3, 2 FDR-significant).
- Q1 medium pooling helps: EpilepticSeizures +0.0052, Haptics -0.0091, Phoneme +0.0073.
- Q2 local pooling helps: EpilepticSeizures +0.0051, Haptics -0.0341, Phoneme +0.0068.
- Q3 combined (M3): EpilepticSeizures +0.0039, Haptics -0.0149, Phoneme +0.0130.
- Q4 EpilepticSeizures: delta M3=+0.0039. Q5 Haptics: -0.0149. Q6 Phoneme: +0.0130.
- Q7: regional blocks are mathematically distinct from global PPV (sections 14-16) — locality information EXISTS in the representation; whether it helps classification is answered by Q1-Q3.
- Q8: partially justified — gains are significant but inconsistent across datasets.

## Prior-art note (sec. 44)

- MultiRocket (Dempster & Webb; included in aeon) extends MiniRocket with additional global pooling operators (MPV, LSPV, IASPV) — but all of them pool over the whole series; it does NOT introduce temporal regions.
- 'Structured temporal representation' (Schlegel et al., 2025) studies structured aggregation in ROCKET-family transforms; learnable temporal pooling (DTP, AAAI 2021) learns position-aware pooling with neural parameters.
- No prior work found in the available project/research materials that applies a FIXED deterministic multi-scale regional PPV pyramid to the canonical MiniROCKET activation masks. Any novelty claim should nonetheless be validated by a dedicated literature review; this experiment is framed as a controlled representation study, not as a novelty claim.
