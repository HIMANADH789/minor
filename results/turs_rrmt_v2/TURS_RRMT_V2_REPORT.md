# TURS-RRMT-V2: Readout Surgery on the Frozen V1 Representation

*Runtime 0.9 min · seed 42 · V1 A4 backbone frozen (fixed bank M=128, J=4 flavors, top-2 routing) · canonical split/protocol*

## 1. Question

V1 concluded that the RRMT **representation** is strong (Ridge on A4 features beats the neural head) but the MLP head wastes it. V2 keeps the representation frozen and swaps ONLY the readout:

- **R0** V1 MLP head (baseline) — **R1** frozen sklearn Ridge (post-hoc)
- **R2** differentiable ridge — **R3** class-weighted ridge
- **R4** soft routing (τ=2) + ridge — **R5** kernelized routed readout
- **R6** M=256 bank + ridge — **R7** M=512 bank + ridge

## 2. Headline results (test Macro-F1)

| Dataset | V1 A4 (R0) | R1 | **R2** | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.4171 | 0.4872 | 0.5469 | 0.4359 | **0.6471** | 0.4749 | 0.5462 | 0.4788 |
| ECG5000_BAL | 0.5501 | **0.5855** | 0.5427 | 0.5155 | 0.5775 | 0.4924 | 0.5542 | 0.5469 |
| CWRU_UNBAL | 0.6945 | 0.9121 | 0.9202 | 0.9202 | 0.9249 | 0.7540 | 0.9250 | **0.9327** |
| CWRU_BAL | 0.8316 | 0.9344 | 0.9344 | 0.9344 | 0.9273 | 0.7806 | **0.9505** | 0.9399 |

## 3. V2 vs V1 neural head (R0) — paired McNemar with BH-FDR

| Dataset | Comparison | Δ MF1 | McNemar p | q (FDR) | Verdict |
|---|---|---|---|---|---|
| ECG5000_UNBAL | ECG5000_UNBAL: R1 vs R0 | +0.0701 | 0.0056 | 0.0082 | SIGNIFICANT IMPROVEMENT |
| ECG5000_UNBAL | ECG5000_UNBAL: R2 vs R0 | +0.1298 | 0.3020 | 0.3253 | n.s. (improves) |
| ECG5000_UNBAL | ECG5000_UNBAL: R3 vs R0 | +0.0188 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| ECG5000_UNBAL | ECG5000_UNBAL: R4 vs R0 | +0.2300 | 0.0072 | 0.0096 | SIGNIFICANT IMPROVEMENT |
| ECG5000_UNBAL | ECG5000_UNBAL: R5 vs R0 | +0.0578 | 0.0009 | 0.0016 | SIGNIFICANT IMPROVEMENT |
| ECG5000_UNBAL | ECG5000_UNBAL: R6 vs R0 | +0.1291 | 0.0030 | 0.0049 | SIGNIFICANT IMPROVEMENT |
| ECG5000_UNBAL | ECG5000_UNBAL: R7 vs R0 | +0.0617 | 0.0455 | 0.0554 | n.s. (improves) |
| ECG5000_BAL | ECG5000_BAL: R1 vs R0 | +0.0353 | 0.0051 | 0.0079 | SIGNIFICANT IMPROVEMENT |
| ECG5000_BAL | ECG5000_BAL: R2 vs R0 | -0.0074 | 0.1198 | 0.1342 | n.s. (degrades) |
| ECG5000_BAL | ECG5000_BAL: R3 vs R0 | -0.0346 | 0.0000 | 0.0000 | SIGNIFICANT DEGRADATION |
| ECG5000_BAL | ECG5000_BAL: R4 vs R0 | +0.0274 | 0.8973 | 0.8973 | n.s. (improves) |
| ECG5000_BAL | ECG5000_BAL: R5 vs R0 | -0.0577 | 0.0001 | 0.0001 | SIGNIFICANT DEGRADATION |
| ECG5000_BAL | ECG5000_BAL: R6 vs R0 | +0.0041 | 0.0191 | 0.0243 | SIGNIFICANT IMPROVEMENT |
| ECG5000_BAL | ECG5000_BAL: R7 vs R0 | -0.0032 | 0.4497 | 0.4663 | n.s. (degrades) |
| CWRU_UNBAL | CWRU_UNBAL: R1 vs R0 | +0.2176 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_UNBAL | CWRU_UNBAL: R2 vs R0 | +0.2257 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_UNBAL | CWRU_UNBAL: R3 vs R0 | +0.2257 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_UNBAL | CWRU_UNBAL: R4 vs R0 | +0.2303 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_UNBAL | CWRU_UNBAL: R5 vs R0 | +0.0595 | 0.0825 | 0.0962 | n.s. (improves) |
| CWRU_UNBAL | CWRU_UNBAL: R6 vs R0 | +0.2305 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_UNBAL | CWRU_UNBAL: R7 vs R0 | +0.2382 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_BAL | CWRU_BAL: R1 vs R0 | +0.1028 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_BAL | CWRU_BAL: R2 vs R0 | +0.1028 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_BAL | CWRU_BAL: R3 vs R0 | +0.1028 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_BAL | CWRU_BAL: R4 vs R0 | +0.0958 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_BAL | CWRU_BAL: R5 vs R0 | -0.0510 | 0.0059 | 0.0082 | SIGNIFICANT DEGRADATION |
| CWRU_BAL | CWRU_BAL: R6 vs R0 | +0.1189 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |
| CWRU_BAL | CWRU_BAL: R7 vs R0 | +0.1084 | 0.0000 | 0.0000 | SIGNIFICANT IMPROVEMENT |

## 4. Key findings

- **ECG5000_UNBAL**: best readout = **R4**. Frozen ridge (R1) 0.487 vs MLP head (R0) 0.417; differentiable ridge (R2) 0.547; soft routing (R4) 0.6471.
- **ECG5000_BAL**: best readout = **R1**. Frozen ridge (R1) 0.585 vs MLP head (R0) 0.550; differentiable ridge (R2) 0.543; soft routing (R4) 0.5775.
- **CWRU_UNBAL**: best readout = **R7**. Frozen ridge (R1) 0.912 vs MLP head (R0) 0.695; differentiable ridge (R2) 0.920; soft routing (R4) 0.9249.
- **CWRU_BAL**: best readout = **R6**. Frozen ridge (R1) 0.934 vs MLP head (R0) 0.832; differentiable ridge (R2) 0.934; soft routing (R4) 0.9273.

- **Uncertainty (D3)**: ECG5000_UNBAL: routing-entropy AUROC 0.6582 vs confidence AUROC 0.8449; ECG5000_BAL: routing-entropy AUROC 0.6131 vs confidence AUROC 0.887; CWRU_UNBAL: routing-entropy AUROC 0.3855 vs confidence AUROC 0.8209; CWRU_BAL: routing-entropy AUROC 0.516 vs confidence AUROC 0.8035
- **Soft vs hard routing (D6)**: ECG5000_UNBAL: agreement 0.975, McNemar p=0.01391; ECG5000_BAL: agreement 0.948, McNemar p=0.05527; CWRU_UNBAL: agreement 0.9625, McNemar p=1.0; CWRU_BAL: agreement 0.9806, McNemar p=0.34278

## 5. Diagnostic detail (D1-D6)

### ECG5000_UNBAL
- **D1 replay fidelity** (recomputed MF1 vs recorded): R0: 0.4171 (Δ 0.0), R1: 0.4872 (Δ 0.0), R2: 0.5469 (Δ 0.0), R3: 0.4359 (Δ 0.0), R4: 0.6471 (Δ 0.0), R5: 0.4749 (Δ 0.0), R6: 0.5462 (Δ 0.0), R7: 0.4788 (Δ 0.0)
- **D2 error analysis**: minority classes [2, 4]; worst minority-MF1 variant R0 = 0.0
- **D3 uncertainty** routing_entropy: AUROC 0.6582 CI95 [0.5828, 0.7235]
- **D3 uncertainty** confidence_neg: AUROC 0.8449 CI95 [0.8082, 0.8793]
- **D4 flavor usage**: flavor_0: mass 0.2175, flavor_1: mass 0.4451, flavor_2: mass 0.246, flavor_3: mass 0.0915
- **D5 lambda sensitivity**: best λ=10.0 val-MF1 0.5835, spread 0.0972
- **D6 soft-vs-hard**: agreement 0.975 over 25 disagreements, McNemar p=0.01391

### ECG5000_BAL
- **D1 replay fidelity** (recomputed MF1 vs recorded): R0: 0.5501 (Δ 0.0), R1: 0.5855 (Δ 0.0), R2: 0.5427 (Δ 0.0), R3: 0.5155 (Δ 0.0), R4: 0.5775 (Δ 0.0), R5: 0.4924 (Δ 0.0), R6: 0.5542 (Δ 0.0), R7: 0.5469 (Δ 0.0)
- **D2 error analysis**: minority classes [2]; worst minority-MF1 variant R5 = 0.2456
- **D3 uncertainty** routing_entropy: AUROC 0.6131 CI95 [0.5539, 0.6709]
- **D3 uncertainty** confidence_neg: AUROC 0.887 CI95 [0.8604, 0.9128]
- **D4 flavor usage**: flavor_0: mass 0.1365, flavor_1: mass 0.3211, flavor_2: mass 0.4726, flavor_3: mass 0.0698
- **D5 lambda sensitivity**: best λ=1.0 val-MF1 0.8229, spread 0.0358
- **D6 soft-vs-hard**: agreement 0.948 over 52 disagreements, McNemar p=0.05527

### CWRU_UNBAL
- **D1 replay fidelity** (recomputed MF1 vs recorded): R0: 0.6945 (Δ 0.0), R1: 0.9121 (Δ 0.0), R2: 0.9202 (Δ 0.0), R3: 0.9202 (Δ 0.0), R4: 0.9249 (Δ 0.0), R5: 0.7540 (Δ 0.0), R6: 0.9250 (Δ 0.0), R7: 0.9327 (Δ 0.0)
- **D2 error analysis**: minority classes [2]; worst minority-MF1 variant R0 = 0.5769
- **D3 uncertainty** routing_entropy: AUROC 0.3855 CI95 [0.32, 0.4601]
- **D3 uncertainty** confidence_neg: AUROC 0.8209 CI95 [0.767, 0.8665]
- **D4 flavor usage**: flavor_0: mass 0.3155, flavor_1: mass 0.5317, flavor_2: mass 0.0704, flavor_3: mass 0.0823
- **D5 lambda sensitivity**: best λ=0.001 val-MF1 0.9211, spread 0.0611
- **D6 soft-vs-hard**: agreement 0.9625 over 9 disagreements, McNemar p=1.0

### CWRU_BAL
- **D1 replay fidelity** (recomputed MF1 vs recorded): R0: 0.8316 (Δ 0.0), R1: 0.9362 (Δ 0.001786), R2: 0.9344 (Δ 0.0), R3: 0.9344 (Δ 0.0), R4: 0.9273 (Δ 0.0), R5: 0.7806 (Δ 0.0), R6: 0.9505 (Δ 0.0), R7: 0.9399 (Δ 0.0)
- **D2 error analysis**: minority classes [2, 3]; worst minority-MF1 variant R5 = 0.718
- **D3 uncertainty** routing_entropy: AUROC 0.516 CI95 [0.4688, 0.5648]
- **D3 uncertainty** confidence_neg: AUROC 0.8035 CI95 [0.766, 0.8426]
- **D4 flavor usage**: flavor_0: mass 0.4571, flavor_1: mass 0.4133, flavor_2: mass 0.0561, flavor_3: mass 0.0735
- **D5 lambda sensitivity**: best λ=0.01 val-MF1 0.9364, spread 0.0746
- **D6 soft-vs-hard**: agreement 0.9806 over 11 disagreements, McNemar p=0.34278

## 6. Interpretation discipline

- All readouts share the SAME frozen V1 representation and the SAME splits; differences are attributable to the readout alone.
- McNemar tests are paired on identical test samples; BH-FDR is applied within the V2_vs_R0 family (and within each diagnostic family separately).
- R6/R7 reuse the seed-fixed pattern bank at larger M with an UNTRAINED router/projection — they probe bank capacity, not full model capacity at that M.
- This is machine-learning benchmarking, NOT clinical validation.

## 7. Limitations

- Single seed (42); no multi-seed variance.
- The differentiable-ridge path (R2/R3) fits W* in closed form on the frozen features; end-to-end joint training was NOT run in V2.
- R5's kernel features reduce each flavor to its time-mean — a deliberately severe compression (10 dims).

## 8. Artifacts

- Per-variant results: `results/turs_rrmt_v2/<DS>/<V>_results.json`
- Replay predictions: `results/turs_rrmt_v2/<DS>/predictions/`
- Comparison CSV: `results/turs_rrmt_v2/tables/model_comparison.csv`
- Hypothesis tests: `results/turs_rrmt_v2/tables/hypothesis_tests.csv`
- V1 handoff: `TURS_RRMT_HANDOFF.md`
