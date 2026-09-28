# MiniROCKET vs TURS-Stack vs TURS-SKB — Detailed Metric Comparison

Compiled 2026-09-14 from:
- `results/turs_skb/full_results.json`, `summary.json`, `reports/REPORT.md` (MiniROCKET M0 + structured bank M1–M6, seed 42, 5-seed robustness)
- `results/turs_stack/<DS>/full_results.json` (TURS-Stack full run, seed 42, combiner fit on val only)
- `results/baseline_bench/<DS>.json` (reference values, bit-reproduced by M0)

All numbers are TEST split, canonical benchmark protocol (seed 42, stratified split, z-norm, train+val fit).

## 1. Headline: Test Macro-F1

| Dataset | MiniROCKET 10K (M0) | TURS-SKB M6 (+All5) | TURS-Stack soft-vote | TURS-Stack val-selected combiner | Winner |
|---|---:|---:|---:|---:|---|
| ECG5000_UNBAL | 0.5938 | 0.5736 | 0.6150 | 0.5883 (stacking) | **TURS-Stack soft-vote (+2.1pp)** |
| ECG5000_BAL | 0.6553 | 0.6074 | 0.6631 | 0.6725 (diag-stacking) | **TURS-Stack diag-stacking (+1.7pp)** |
| CWRU_UNBAL | 0.9917 | 0.9583 | 0.9539 | 0.9539 (soft-vote) | **MiniROCKET (+3.8pp)** |
| CWRU_BAL | 0.9947 | 0.9842 | 0.9877 | 0.9859 (static weights) | **MiniROCKET (+0.7pp)** |
| **Mean** | **0.8089** | 0.7809 | 0.8049 | 0.8002 | MiniROCKET |

Pattern: **TURS-Stack wins both ECG datasets; MiniROCKET wins both CWRU datasets.** The learned 4-branch ensemble only pays off where class morphology (ECG beats) is hard; on near-ceiling fault classification the fixed-feature + ridge pipeline is better and simpler.

## 2. Full metric table — MiniROCKET 10K (M0, ridge readout)

| Metric | ECG5000_UNBAL | ECG5000_BAL | CWRU_UNBAL | CWRU_BAL |
|---|---:|---:|---:|---:|
| Test Macro-F1 | 0.5938 | 0.6553 | 0.9917 | 0.9947 |
| Test accuracy | 0.9540 | 0.9520 | 0.9917 | 0.9947 |
| Test weighted-F1 | 0.9458 | 0.9498 | 0.9917 | 0.9947 |
| Test balanced accuracy | 0.5577 | 0.6398 | 0.9917 | 0.9947 |
| Class F1s | [0.9949, 0.9494, 0.5161, 0.5085, 0.0] | [0.9983, 0.9456, 0.6111, 0.5217, 0.2] | [1.0, 1.0, 0.9836, 0.9831] | [1.0, 1.0, 0.9895, 0.9894] |
| Class precision | [0.9898, 0.9180, 0.6667, 0.7500, 0.0] | [0.9983, 0.9313, 0.6471, 0.6000, 0.2] | [1.0, 1.0, 0.9677, 1.0] | [1.0, 1.0, 0.9860, 0.9929] |
| Class recall | [1.0, 0.9830, 0.4211, 0.3846, 0.0] | [0.9983, 0.9603, 0.5789, 0.4615, 0.2] | [1.0, 1.0, 1.0, 0.9667] | [1.0, 1.0, 0.9930, 0.9859] |
| Feature dim | 9,996 | 9,996 | 9,996 | 9,996 |
| Ridge fit (train+val, s) | 12.6 | 38.9 | 1.2 | 6.9 |
| Test inference (s) | 0.0122 | 0.0177 | 0.0030 | 0.0071 |
| Trainable neural params | 0 | 0 | 0 | 0 |

ECG5000_UNBAL failure mode: minority classes 3–5 collapse (class 5 F1 = 0 — never predicted); accuracy 0.954 is dominated by the majority beat classes, hence macro-F1 0.594 vs accuracy 0.954.

## 3. Full metric table — TURS-Stack (4-branch ensemble, 164K params)

| Metric | ECG5000_UNBAL | ECG5000_BAL | CWRU_UNBAL | CWRU_BAL |
|---|---:|---:|---:|---:|
| Soft-vote test Macro-F1 | 0.6150 | 0.6631 | 0.9539 | 0.9877 |
| Soft-vote test accuracy | 0.9560 | 0.9490 | 0.9542 | 0.9877 |
| Best combiner test MF1 | 0.6287 (static w.) | 0.6725 (diag-stack) | 0.9581 (static w.) | 0.9877 (soft-vote) |
| Val-selected combiner test MF1 (pre-registered protocol) | 0.5883 | 0.6725 | 0.9539 | 0.9859 |
| Hard-vote / stacking / diag-stack test | 0.6100 / 0.5883 / 0.5853 | 0.6562 / 0.6056 / 0.6725 | 0.9539 / 0.9580 / 0.9580 | 0.9877 / 0.9877 / 0.9877 |
| Branch test MF1 (lite/rv/cs/cmr) | 0.6116 / 0.6077 / 0.6174 / 0.6287 | 0.5867 / 0.6468 / 0.6676 / 0.6141 | 0.9581 / 0.9581 / 0.9623 / 0.9581 | 0.9859 / 0.9859 / 0.9877 / 0.9859 |
| Val soft-vote MF1 (selection) | 0.6639 | 0.9590 | 0.9835 | 0.9930 |
| Val → test gap (soft-vote) | −4.9pp | −29.6pp | −3.0pp | −0.5pp |
| Params (total / 4x-independent) | 164,811 / 659,244 | 164,811 / 659,244 | 164,615 / 659,048 | 164,615 / 659,048 |
| Train time (s, RTX 4050) | 292.5 | 426.7 | 773.6 | 1,962.8 |
| Best epoch (of 30) | 23 | 29 | 26 | 22 |

Notes:
- Val→test generalization gap is the dominant TURS-Stack weakness on ECG (−29.6pp on ECG5000_BAL: val 0.9590 → test 0.6631). Combiner selection on val did not transfer (stacking looked best at 0.8106 val, gave 0.5883 test).
- All 12 correctness gates passed on all datasets (trunk called once/forward, branch independence, beta simplex, no cross-window state, serialization, etc.).
- Class-5 failure mirrors MiniROCKET on ECG5000_UNBAL: predicted-class distribution [588, 372, 19, 21, 0] vs true [584, 353, 19, 39, 5] — class 5 never predicted by any branch.
- CWRU_UNBAL: 60/60/68/52 predictions vs 60/60/60/60 true — mild majority-class bias.

## 4. TURS-SKB structured bank vs MiniROCKET (this benchmark's primary question)

Test Macro-F1, seed 42:

| Dataset | M0 | +Morph | +Deriv | +Wavelet | +Gabor | +Energy | +All5 (M6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ECG5000_UNBAL | 0.5938 | 0.5963 | 0.5874 | 0.6058 | 0.5919 | 0.5958 | 0.5736 |
| ECG5000_BAL | 0.6553 | 0.6382 | 0.6516 | 0.6013 | 0.6118 | 0.6484 | 0.6074 |
| CWRU_UNBAL | 0.9917 | 0.9750 | 0.9833 | 0.9791 | 0.9457 | 0.9875 | 0.9583 |
| CWRU_BAL | 0.9947 | 0.9930 | 0.9982 | 0.9930 | 0.9912 | 0.9930 | 0.9842 |
| Mean delta vs M0 | — | −0.0083 | −0.0038 | −0.0141 | −0.0237 | −0.0027 | **−0.0280** |

Statistical inference (paired McNemar on held-out test, BH-FDR within dataset over the 6 primary comparisons):
- No comparison reaches significance on any dataset (all q ≥ 0.05; e.g. ECG5000_UNBAL M6 q = 0.868).
- Val-only screening (train-fit ridge → val MF1) found **no family beating M0 on any dataset** → per the pre-registered protocol no family/pair would have been deployed; pairwise models were never triggered.
- 5-seed robustness (seeds 42–46, M6 vs M0): negative on 20/20 seed-dataset pairs; mean delta −0.0226±0.0068 (ECG-U), −0.0589±0.0140 (ECG-B), −0.0317±0.0038 (CWRU-U), −0.0105±0.0001 (CWRU-B); paired-permutation p ≈ 0.056–0.068 per dataset; effect sizes Cohen's d −3.3 to −192.
- All 5 pre-registered mechanistic expectations REJECTED (no family helps its target domain).
- Cost of M6: +1,791 features (ECG) / +3,636 features (CWRU) for strictly negative or null returns; gain per 1K added features is negative everywhere.

## 5. Model-cost summary

| | MiniROCKET 10K | TURS-SKB M6 | TURS-Stack |
|---|---|---|---|
| Trainable neural params | 0 | 0 | 164,615–164,811 (75% saved vs 4 independent branches) |
| Readout | RidgeClassifierCV (10K→C) | RidgeClassifierCV (11.8K–13.6K→C) | per-branch heads + 5 combiners |
| Fit time (all 4 datasets) | ridge 1.2–38.9 s/dataset (CPU) | + 4.6–142.4 s/dataset SKB extraction (CPU) | 292–1,963 s/dataset (GPU) |
| Test inference | 3–18 ms | 3–18 ms | GPU forward per window |
| Hyperparameter surface | kernel count | kernel grids (frozen) | branches, combiners, epochs, lr |

## 6. Conclusions

1. **ECG (morphology-limited):** TURS-Stack soft-vote 0.6150/0.6631 beats MiniROCKET 0.5938/0.6553 (+2.1/+0.8pp); the learned branches add complementary shape sensitivity that the fixed kernel bank misses. But the val→test gap (up to −29.6pp) and combiner instability (stacking overfits val) make the learned pipeline fragile.
2. **CWRU (fault signatures):** MiniROCKET is better on both (0.9917/0.9947 vs 0.9539/0.9877) at 0 trainable params and CPU-only fitting; TURS-Stack's trunk spends 164K parameters to underperform a linear readout on fixed features.
3. **TURS-SKB (structured kernel enrichment):** decisively NEGATIVE — mean −2.8pp for M6, no family/significant improvement anywhere, 20/20 negative across seeds. Canonical MiniROCKET's sparse random kernels + quantile-PPV already capture the predictive temporal structure available to fixed-feature + linear-readout pipelines at this scale.
4. Best single SKB family per dataset (test, post-hoc): wavelet 0.6058 (ECG-U), derivative 0.6516 (ECG-B), energy 0.9875 (CWRU-U), derivative 0.9982 (CWRU-B) — mean 0.8108, marginally above M0's 0.8089, but this is test-set selection and not protocol-admissible (val screening selected none).
