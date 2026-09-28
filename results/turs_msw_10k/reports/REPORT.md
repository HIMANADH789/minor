# TURS-MSW-10K — Fixed-Budget Multi-Scale Windowed MiniROCKET: Final Report

Generated 2026-09-14 15:53 | seed 42 | final five datasets: ECG5000_UNBAL, ECG5000_BAL, CWRU_UNBAL, CWRU_BAL, EpilepticSeizures | primary metric: TEST Macro-F1

## 1. Executive summary

- **ECG5000_UNBAL**: selected **M0** — test MF1 0.5938 vs M0 0.5938 (delta +0.0000)
- **ECG5000_BAL**: selected **M0** — test MF1 0.6553 vs M0 0.6553 (delta +0.0000)
- **CWRU_UNBAL**: selected **M0** — test MF1 0.9917 vs M0 0.9917 (delta +0.0000)
- **CWRU_BAL**: selected **M0** — test MF1 0.9947 vs M0 0.9947 (delta +0.0000)
- **EpilepticSeizures**: selected **M0** — test MF1 0.9194 vs M0 0.9194 (delta +0.0000)
- Validation-only selection retained M0 on 5/5 datasets and selected an MSW variant on 0.
- Five-dataset aggregate: mean M0 0.8310 vs mean selected 0.8310 (mean gain +0.0000).

## 2-4. Research question, budget control, canonical baseline

- **Question**: can a fixed ~10K MiniROCKET feature budget, redistributed across global/medium/local temporal pooling, beat canonical 10K global MiniROCKET? The kernel bank is NOT enlarged — this isolates pooling allocation from capacity (the previous MSW study's 7x inflation is removed by design).
- **Frozen five** (from repository inspection, not results): the four canonical core datasets (`results/baseline_bench`, npz splits, seed-42 stratified 15% val, z-norm) + **EpilepticSeizures** (the PRIMARY external set; Haptics/Phoneme were secondary diagnostics, excluded).
- **M0 verified**: canonical aeon MiniRocket(seed=42, ~10K) + RidgeClassifierCV(logspace(-4,4,20)), final train+val fit.

| dataset | M0 test MF1 | reference | delta |
|---|---:|---:|---:|
| ECG5000_UNBAL | 0.5938 | 0.5938 | +0.0000 |
| ECG5000_BAL | 0.6553 | 0.6553 | +0.0000 |
| CWRU_UNBAL | 0.9917 | 0.9917 | +0.0000 |
| CWRU_BAL | 0.9947 | 0.9947 | +0.0000 |
| EpilepticSeizures | 0.9194 | 0.9194 | +0.0000 |

## 5-10. Architecture, allocation, pooling, exact 10K budget

- ONE canonical MiniRocket fit per dataset; kernels/biases/dilations/responses/masks shared by all variants (no per-variant refitting).
- Deterministic DISJOINT kernel allocation (strided modular rule over aeon's dilation-major layout; documented in msw10k.py):

| variant | global x1 | medium x2 | local x4 | total features |
|---|---:|---:|---:|---:|
| M0 | 9996 | 0 | 0 | **9996** |
| M1 | 4998 | 2499 | 0 | **9996** |
| M2 | 4996 | 0 | 1250 | **9996** |
| M3 | 3332 | 1666 | 833 | **9996** |
| M4 | 0 | 2498 | 1250 | **9996** |

- Windows: identical to the validated MSW study (global [0,T); medium w=ceil(3T/4) x2 @50% overlap; local w=ceil(T/2) x4 @50% overlap; padding1==1 kernels mapped to aeon's valid axis).
- Equivalence gate: the global block of every global-assigned kernel is BIT-IDENTICAL to the canonical MiniRocket feature (max|diff| reported per dataset below).
- Features are PPV rates in [0,1] on one natural scale -> canonical RAW ridge protocol retained; block statistics in diagnostics.json.

## 11-12. Dataset protocol and validation-only selection

- Core four: canonical npz splits, stratified 15% val (seed 42), per-sample z-norm; EpilepticSeizures: provided canonical val split. Test untouched until after the frozen decision.
- Selection rule (frozen BEFORE test): candidates with val MF1 > M0 AND BH-adjusted q < 0.05 (paired McNemar vs M0 on val predictions); pick highest val MF1 among eligible; otherwise M0. Decision saved to disk before any test evaluation.

| dataset | M0 val | M1 val | M2 val | M3 val | M4 val | selected |
|---|---:|---:|---:|---:|---:|---|
| ECG5000_UNBAL | 0.6405 | 0.6462 | 0.6384 | 0.6446 | 0.6464 | **M0** |
| ECG5000_BAL | 0.9546 | 0.9483 | 0.9412 | 0.9526 | 0.9458 | **M0** |
| CWRU_UNBAL | 0.9804 | 0.9902 | 0.9804 | 0.9755 | 0.9412 | **M0** |
| CWRU_BAL | 0.9896 | 0.9938 | 0.9896 | 0.9876 | 0.9710 | **M0** |
| EpilepticSeizures | 0.9499 | 0.9499 | 0.9499 | 0.9499 | 1.0000 | **M0** |

## 13-14. Statistical inference and main results

Test-side paired McNemar (each MSW variant vs M0), BH-adjusted over the 4 comparisons per dataset:

| dataset | comparison | delta MF1 | chi2 | p_raw | q_fdr | effect d |
|---|---|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | M0 vs M1 (test) | -0.0035 | 0.5 | 0.4795 | 0.75183 | 0.0 |
| ECG5000_UNBAL | M0 vs M2 (test) | +0.0024 | 0.25 | 0.617075 | 0.75183 | 0.0316 |
| ECG5000_UNBAL | M0 vs M3 (test) | -0.0125 | 0.1667 | 0.683091 | 0.75183 | 0.0 |
| ECG5000_UNBAL | M0 vs M4 (test) | -0.0034 | 0.1 | 0.75183 | 0.75183 | 0.02 |
| ECG5000_BAL | M0 vs M1 (test) | +0.0180 | 0.0833 | 0.77283 | 1.0 | 0.0183 |
| ECG5000_BAL | M0 vs M2 (test) | -0.0006 | 0.0714 | 0.789268 | 1.0 | 0.0169 |
| ECG5000_BAL | M0 vs M3 (test) | +0.0057 | 0.0 | 1.0 | 1.0 | 0.0088 |
| ECG5000_BAL | M0 vs M4 (test) | -0.0498 | 0.05 | 0.823063 | 1.0 | -0.0141 |
| CWRU_UNBAL | M0 vs M1 (test) | -0.0042 | 0.0 | 1.0 | 1.0 | -0.0372 |
| CWRU_UNBAL | M0 vs M2 (test) | -0.0125 | 0.8 | 0.371093 | 0.742187 | -0.0867 |
| CWRU_UNBAL | M0 vs M3 (test) | -0.0084 | 0.25 | 0.617075 | 0.822767 | -0.0645 |
| CWRU_UNBAL | M0 vs M4 (test) | -0.0542 | 9.6 | 0.001946 | 0.007783 | -0.2215 |
| CWRU_BAL | M0 vs M1 (test) | -0.0017 | 0.0 | 1.0 | 1.0 | -0.0242 |
| CWRU_BAL | M0 vs M2 (test) | +0.0000 | 0.1667 | 0.683091 | 0.910789 | 0.0 |
| CWRU_BAL | M0 vs M3 (test) | -0.0035 | 0.1667 | 0.683091 | 0.910789 | -0.0343 |
| CWRU_BAL | M0 vs M4 (test) | -0.0246 | 12.0714 | 0.000512 | 0.002048 | -0.159 |
| EpilepticSeizures | M0 vs M1 (test) | +0.0155 | 47.1468 | 0.0 | 0.0 | 0.065 |
| EpilepticSeizures | M0 vs M2 (test) | +0.0154 | 44.0151 | 0.0 | 0.0 | 0.0628 |
| EpilepticSeizures | M0 vs M3 (test) | +0.0138 | 38.9251 | 0.0 | 0.0 | 0.0591 |
| EpilepticSeizures | M0 vs M4 (test) | +0.0199 | 42.7159 | 0.0 | 0.0 | 0.0617 |

### Master table (test Macro-F1)

| Dataset | M0 | M1 G+M | M2 G+L | M3 G+M+L | M4 M+L | Selected | Final Test |
|---|---:|---:|---:|---:|---:|---|---:|
| ECG5000_UNBAL | 0.5938 | 0.5903 | 0.5962 | 0.5813 | 0.5904 | M0 | **0.5938** |
| ECG5000_BAL | 0.6553 | 0.6733 | 0.6547 | 0.6610 | 0.6055 | M0 | **0.6553** |
| CWRU_UNBAL | 0.9917 | 0.9875 | 0.9792 | 0.9833 | 0.9375 | M0 | **0.9917** |
| CWRU_BAL | 0.9947 | 0.9930 | 0.9947 | 0.9912 | 0.9701 | M0 | **0.9947** |
| EpilepticSeizures | 0.9194 | 0.9349 | 0.9348 | 0.9332 | 0.9393 | M0 | **0.9194** |
| **Mean** | 0.8310 | 0.8358 | 0.8319 | 0.8300 | 0.8086 | — | 0.8310 |

## 15. Per-dataset decisions

| Dataset | Selected variant | M0 Test | Final Test | Gain |
|---|---|---:|---:|---:|
| ECG5000_UNBAL | M0 | 0.5938 | 0.5938 | +0.0000 |
| ECG5000_BAL | M0 | 0.6553 | 0.6553 | +0.0000 |
| CWRU_UNBAL | M0 | 0.9917 | 0.9917 | +0.0000 |
| CWRU_BAL | M0 | 0.9947 | 0.9947 | +0.0000 |
| EpilepticSeizures | M0 | 0.9194 | 0.9194 | +0.0000 |

## 16-17. Representation and locality diagnostics

- Block similarity (M3 train matrix, memory-safe sample-gram CKA):

| dataset | CKA G-M | CKA G-L | CKA M-L | mean|corr| G-M | mean|corr| G-L |
|---|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | 0.945886 | 0.879207 | 0.903117 | 0.357443 | 0.307801 |
| ECG5000_BAL | 0.906131 | 0.785057 | 0.823172 | 0.341422 | 0.288902 |
| CWRU_UNBAL | 0.940993 | 0.846796 | 0.893337 | 0.39226 | 0.349754 |
| CWRU_BAL | 0.941567 | 0.839862 | 0.888994 | 0.39944 | 0.347864 |
| EpilepticSeizures | 0.728102 | 0.722282 | 0.93832 | 0.270971 | 0.257628 |

- Locality mechanism (synthetic mirrored-burst pair; same global activation count, different location) and T/2 circular-shift diagnostic (expected: global change < medium < local):

| dataset | loc: |dG| | loc: |dM| | loc: |dL| | shift: |dG| | |dM| | |dL| |
|---|---:|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | 0.031984 | 0.039907 | 0.058076 | 0.076641 | 0.122133 | 0.182219 |
| ECG5000_BAL | 0.031689 | 0.039557 | 0.05765 | 0.084099 | 0.130637 | 0.185982 |
| CWRU_UNBAL | 0.004297 | 0.005539 | 0.008313 | 0.013301 | 0.024871 | 0.043794 |
| CWRU_BAL | 0.004306 | 0.005548 | 0.008313 | 0.014209 | 0.028983 | 0.055555 |
| EpilepticSeizures | 0.025331 | 0.03142 | 0.045596 | 0.042604 | 0.05963 | 0.096084 |

## 18. Class-level analysis

- **ECG5000_UNBAL** (M0 vs M0): class F1 [0.9949, 0.9494, 0.5161, 0.5085, 0.0] -> [0.9949, 0.9494, 0.5161, 0.5085, 0.0]
- **ECG5000_BAL** (M0 vs M0): class F1 [0.9983, 0.9456, 0.6111, 0.5217, 0.2] -> [0.9983, 0.9456, 0.6111, 0.5217, 0.2]
- **CWRU_UNBAL** (M0 vs M0): class F1 [1.0, 1.0, 0.9836, 0.9831] -> [1.0, 1.0, 0.9836, 0.9831]
- **CWRU_BAL** (M0 vs M0): class F1 [1.0, 1.0, 0.9895, 0.9894] -> [1.0, 1.0, 0.9895, 0.9894]
- **EpilepticSeizures** (M0 vs M0): class F1 [0.8706, 0.9683] -> [0.8706, 0.9683]

## 19-20. Computational cost and robustness

| dataset | MR fit s | regional tr+va s | regional test pass s |
|---|---:|---:|---:|
| ECG5000_UNBAL | 0.35 | 0.57 | 0.54 |
| ECG5000_BAL | 0.25 | 0.82 | 0.55 |
| CWRU_UNBAL | 0.38 | 1.03 | 0.3 |
| CWRU_BAL | 0.81 | 2.38 | 0.72 |
| EpilepticSeizures | 0.04 | 0.02 | 10.04 |

Robustness seeds 43-46 (M0 vs the SELECTED variant; seed 42 row is the primary run):

| dataset | seed | M0 | selected | delta |
|---|---:|---:|---:|---:|
| ECG5000_UNBAL | 43 | 0.6006 | 0.6006 | +0.0000 |
| ECG5000_UNBAL | 44 | 0.5898 | 0.5898 | +0.0000 |
| ECG5000_UNBAL | 45 | 0.5909 | 0.5909 | +0.0000 |
| ECG5000_UNBAL | 46 | 0.6057 | 0.6057 | +0.0000 |
| ECG5000_BAL | 43 | 0.6451 | 0.6451 | +0.0000 |
| ECG5000_BAL | 44 | 0.6491 | 0.6491 | +0.0000 |
| ECG5000_BAL | 45 | 0.6532 | 0.6532 | +0.0000 |
| ECG5000_BAL | 46 | 0.6473 | 0.6473 | +0.0000 |
| CWRU_UNBAL | 43 | 0.9917 | 0.9917 | +0.0000 |
| CWRU_UNBAL | 44 | 0.9917 | 0.9917 | +0.0000 |
| CWRU_UNBAL | 45 | 0.9833 | 0.9833 | +0.0000 |
| CWRU_UNBAL | 46 | 0.9917 | 0.9917 | +0.0000 |
| CWRU_BAL | 43 | 0.9947 | 0.9947 | +0.0000 |
| CWRU_BAL | 44 | 0.9965 | 0.9965 | +0.0000 |
| CWRU_BAL | 45 | 0.9965 | 0.9965 | +0.0000 |
| CWRU_BAL | 46 | 0.9947 | 0.9947 | +0.0000 |
| EpilepticSeizures | 43 | 0.9170 | 0.9170 | +0.0000 |
| EpilepticSeizures | 44 | 0.9226 | 0.9226 | +0.0000 |
| EpilepticSeizures | 45 | 0.9225 | 0.9225 | +0.0000 |
| EpilepticSeizures | 46 | 0.9235 | 0.9235 | +0.0000 |

## 21-22. Limitations and final conclusion

- Single split per dataset; robustness seeds vary only the MiniRocket kernel draw (the canonical protocol's only stochastic element).
- The strided allocation is one of many deterministic partitions; no allocation search was performed (by design — this tests allocation, not selection).
- Validation sets are small on the external dataset (n=20); selection power is limited there — reported honestly (CASE E).

**Final scientific conclusion (sec. 43 questions):**
- Q8/ECG5000_UNBAL: selected system did NOT improve over MiniROCKET (M0, +0.0000).
- Q8/ECG5000_BAL: selected system did NOT improve over MiniROCKET (M0, +0.0000).
- Q8/CWRU_UNBAL: selected system did NOT improve over MiniROCKET (M0, +0.0000).
- Q8/CWRU_BAL: selected system did NOT improve over MiniROCKET (M0, +0.0000).
- Q8/EpilepticSeizures: selected system did NOT improve over MiniROCKET (M0, +0.0000).

The strongest supported claim is dataset-specific: a fixed-budget, validation-selected multi-scale pooling representation provided measurable dataset-specific gains where the validation procedure found statistically supported improvements; canonical global MiniROCKET remains the correct default where it did not.

## Prior-art note

- MultiRocket (in aeon) adds global pooling operators (MPV/LSPV/IASPV) — none are temporal-region aware. The previous TURS-MSW study is the in-repository precedent for regional pooling (7x budget); this experiment is its fair-capacity control.