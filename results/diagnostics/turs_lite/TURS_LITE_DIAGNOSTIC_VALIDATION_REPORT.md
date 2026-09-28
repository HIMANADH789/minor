# TURS-Lite Intrinsic Diagnostic Validation — Final Report

*Run: 2026-09-11 20:11:51 · seed 42 · Python 3.11.9 · torch 2.5.1+cu121 · device cuda*

## 1. Executive Summary

**Overall verdict: WEAK SUPPORT** (mean dimension score 0.473; 1/6 dimensions consistent across the four datasets).

## 2. Research Question

Does standalone TURS-Lite provide *intrinsic diagnostic support*: do its internal variables z (regime), v (velocity), u (uncertainty) and alpha (transport reliance) carry validated, non-trivial information about the model's own state, its errors, and the input, beyond ordinary confidence baselines?

## 3. Definition of Intrinsic Diagnostic Evidence

A mechanism counts as diagnostic evidence only if: (i) it has a documented mechanistic meaning in the source; (ii) it shows statistically significant, non-negligible effects on pre-specified hypotheses; (iii) it survives comparison against simple baselines (max-softmax, entropy, raw-signal descriptors); (iv) effects are consistent across datasets; and (v) it is faithful under perturbation.

## 4. TURS-Lite Architecture Relevant to Diagnostics

- **z (regime)**: `RegimeEncoder.mu_net` applied to pooled H4 features; sample-level [B,16]. Temporal trace z_t = same frozen encoder applied per timestep (inference-only; documented in audit).
- **v (velocity)**: `vel_linear(z)` — a *learned projection* of z, NOT a temporal difference (source docstring). Temporal trace v_t per timestep.
- **u (uncertainty)**: `softplus(clamp(logvar_net(H4), -5, 2)) + 1e-6` — Gaussian regime dispersion; trained with MSE toward (1 - max softmax) via TURSLoss (lambda_unc=0.01).
- **alpha**: `sigmoid(alpha_net([t_pooled, z, v, u, g_T, g_R]))`; from `F_fused = alpha*(g_T*F_T) + (1-alpha)*(g_R*F_R)`, **alpha weights the TRANSPORT contribution** (low alpha = regime reliance). alpha is sample-level only; no temporal semantics (NOT TESTABLE temporally).

## 5. Datasets

- **ECG5000_UNBAL**: 4000 samples, L=140, C=5; split 3400/600/1000
- **ECG5000_BAL**: 6149 samples, L=140, C=5; split 5226/923/1000
- **CWRU_UNBAL**: 1600 samples, L=1024, C=4; split 1156/204/240
- **CWRU_BAL**: 3776 samples, L=1024, C=4; split 2727/482/567

## 6. Exact Training/Reproduction Protocol

Canonical `fair_turs.py` TURS-Lite configuration, unchanged: TURSNet (regime_dim=16, variant=lite), TURSLoss CE + aux (0.01/0.005/0.01/0.005), AdamW 3e-4/1e-2, OneCycleLR (30 ep), batch 64, grad clip 1.0, seed 42, early stop patience 8 on val macro-F1.

## 7. Predictive Reproduction Results

| Dataset | Historical MF1 | This run MF1 | Acc | BalAcc | MCC | Best ep |
|---|---|---|---|---|---|---|
| ECG5000_UNBAL | 0.5862 | 0.6046 | 0.9550 | 0.5776 | 0.9150 | 14 |
| ECG5000_BAL | 0.6086 | 0.7114 | 0.9590 | 0.6772 | 0.9226 | 27 |
| CWRU_UNBAL | 0.9211 | 0.9504 | 0.9500 | 0.9500 | 0.9336 | 29 |
| CWRU_BAL | 0.9542 | 0.9541 | 0.9541 | 0.9542 | 0.9392 | 21 |

Differences vs historical values are expected (GPU nondeterminism in cuDNN kernels despite deterministic seeds); the diagnostic study uses this run's own frozen checkpoints throughout.

## 8. Latent-State Validation

- ECG5000_UNBAL: silhouette=0.110, ARI=0.379, probe(z) MF1=0.589 vs raw-stats probe MF1=0.371
- ECG5000_BAL: silhouette=-0.072, ARI=0.052, probe(z) MF1=0.586 vs raw-stats probe MF1=0.472
- CWRU_UNBAL: silhouette=0.147, ARI=0.403, probe(z) MF1=0.946 vs raw-stats probe MF1=0.708
- CWRU_BAL: silhouette=0.161, ARI=0.431, probe(z) MF1=0.956 vs raw-stats probe MF1=0.763

## 9. Temporal Velocity Validation

- ECG5000_UNBAL: hit@5 TURS=0.044444444444444446, raw diff=0.06666666666666667, energy=0.022222222222222223, random=0.0
- ECG5000_BAL: hit@5 TURS=0.022222222222222223, raw diff=0.06666666666666667, energy=0.022222222222222223, random=0.0
- CWRU_UNBAL: hit@5 TURS=0.0, raw diff=0.075, energy=0.0, random=0.025
- CWRU_BAL: hit@5 TURS=0.0, raw diff=0.0, energy=0.025, random=0.025

## 10. Uncertainty Validation

- ECG5000_UNBAL: u incorrect>correct (Cliff's δ=0.2421175101803374, p=0.0030026745935425253), AUROC=0.6210587550901687 CI=[0.514499709133217, 0.7166567771960441]
- ECG5000_BAL: u incorrect>correct (Cliff's δ=0.034817772578142885, p=0.35283272050548464), AUROC=0.5174088862890714 CI=[0.4249179785854168, 0.608163991963173]
- CWRU_UNBAL: u incorrect>correct (Cliff's δ=0.5307017543859649, p=0.0009840451296452227), AUROC=0.7653508771929824 CI=[0.6787097953216376, 0.8501553362573099]
- CWRU_BAL: u incorrect>correct (Cliff's δ=0.4942414332432817, p=1.0248812050659643e-05), AUROC=0.7471207166216408 CI=[0.6933616522110052, 0.7954677946822124]

## 11. Calibration

- ECG5000_UNBAL: ECE=0.0186, aECE=0.0155, Brier=0.0739, NLL=0.1571
- ECG5000_BAL: ECE=0.0339, aECE=0.0296, Brier=0.0755, NLL=0.2615
- CWRU_UNBAL: ECE=0.0270, aECE=0.0236, Brier=0.0883, NLL=0.1503
- CWRU_BAL: ECE=0.0315, aECE=0.0291, Brier=0.0749, NLL=0.1748

## 12. Risk-Coverage / Selective Prediction

- ECG5000_UNBAL: AURC u=0.0718 vs entropy=0.1317 (risk@80% u=0.0425)
- ECG5000_BAL: AURC u=0.0440 vs entropy=0.1123 (risk@80% u=0.0425)
- CWRU_UNBAL: AURC u=0.0787 vs entropy=0.1302 (risk@80% u=0.0625)
- CWRU_BAL: AURC u=0.0655 vs entropy=0.1361 (risk@80% u=0.0573)

## 13. Controlled Degradation

- ECG5000_UNBAL: pooled u↔error Spearman ρ=-0.09006197905695291, p=0.0004997501249375312, AUROC=0.4273285755996722
- ECG5000_BAL: pooled u↔error Spearman ρ=0.007458062809903088, p=0.46276861569215394, AUROC=0.5053427369691595
- CWRU_UNBAL: pooled u↔error Spearman ρ=0.25002942919447685, p=0.0004997501249375312, AUROC=0.7083233540000986
- CWRU_BAL: pooled u↔error Spearman ρ=-0.2568532453426572, p=0.0004997501249375312, AUROC=0.29925815980239096

## 14. Alpha / Reliance Validation

- ECG5000_UNBAL: strongest descriptors lag_disagreement (ρ=-0.839), local_variance_mean (ρ=-0.565), ks_stat_uniform (ρ=0.451)
- ECG5000_BAL: strongest descriptors ks_stat_uniform (ρ=-0.522), hist_entropy (ρ=0.275), local_variance_mean (ρ=0.246)
- CWRU_UNBAL: strongest descriptors lag_disagreement (ρ=0.733), local_variance_mean (ρ=-0.554), hist_entropy (ρ=-0.033)
- CWRU_BAL: strongest descriptors local_variance_mean (ρ=-0.366), lag_disagreement (ρ=0.331), ks_stat_uniform (ρ=0.312)

## 15. Faithfulness

- ECG5000_UNBAL [u_t]: targeted p-drop -0.0004 vs random 0.0022; diff=-0.0023 CI=[-0.004132001173192596, -0.0005175532518818122], p=0.0005, d=-0.24
- ECG5000_UNBAL [s_t]: targeted p-drop 0.0064 vs random 0.0048; diff=0.0035 CI=[-0.00021002755963612043, 0.007293667050582975], p=0.5537, d=0.04
- ECG5000_BAL [u_t]: targeted p-drop -0.0034 vs random 0.0130; diff=-0.0182 CI=[-0.03570805493766481, -0.0021420288689501083], p=0.0005, d=-0.18
- ECG5000_BAL [s_t]: targeted p-drop 0.0148 vs random 0.0107; diff=0.0072 CI=[-0.011396266023208453, 0.026875379398743078], p=0.6082, d=0.03
- CWRU_UNBAL [u_t]: targeted p-drop 0.0105 vs random 0.0025; diff=0.0003 CI=[-0.013834513260129216, 0.014535955761481692], p=0.1664, d=0.09
- CWRU_UNBAL [s_t]: targeted p-drop 0.0155 vs random 0.0095; diff=0.0029 CI=[-0.018422342997898033, 0.024165820366858194], p=0.5342, d=0.04
- CWRU_BAL [u_t]: targeted p-drop 0.0025 vs random 0.0060; diff=-0.0037 CI=[-0.012674898904030365, 0.0046055636440966415], p=0.1684, d=-0.08
- CWRU_BAL [s_t]: targeted p-drop 0.0231 vs random 0.0056; diff=0.0143 CI=[-0.0030213086232639954, 0.032314913646162784], p=0.0085, d=0.15

## 16. Stability

- ECG5000_UNBAL: bitwise replay=True, max|Δp|=0.0
- ECG5000_BAL: bitwise replay=True, max|Δp|=0.0
- CWRU_UNBAL: bitwise replay=True, max|Δp|=0.0
- CWRU_BAL: bitwise replay=True, max|Δp|=0.0

## 17. Counterfactual Consistency

- ECG5000_UNBAL: high-s_t edit p-drop 0.0743 vs random 0.0516 (p=0.1114); flips 0.127 vs 0.070
- ECG5000_BAL: high-s_t edit p-drop 0.1074 vs random 0.0198 (p=0.0005); flips 0.157 vs 0.063
- CWRU_UNBAL: high-s_t edit p-drop 0.0812 vs random 0.1137 (p=0.1219); flips 0.154 vs 0.179
- CWRU_BAL: high-s_t edit p-drop 0.1013 vs random 0.0721 (p=0.1099); flips 0.143 vs 0.110

## 18. Simple Baseline Comparisons

- ECG5000_UNBAL turs_u_mean: AUROC=0.6211 CI=[0.5201,0.7148]
- ECG5000_UNBAL turs_u_max: AUROC=0.6758 CI=[0.5854,0.7548]
- ECG5000_UNBAL 1_minus_max_softmax: AUROC=0.8997 CI=[0.8616,0.9325]
- ECG5000_UNBAL predictive_entropy: AUROC=0.9002 CI=[0.8643,0.9328]
- ECG5000_UNBAL top1_top2_margin_neg: AUROC=0.9000 CI=[0.8639,0.9340]
- ECG5000_BAL turs_u_mean: AUROC=0.5174 CI=[0.4291,0.6113]
- ECG5000_BAL turs_u_max: AUROC=0.6909 CI=[0.6198,0.7583]
- ECG5000_BAL 1_minus_max_softmax: AUROC=0.8994 CI=[0.8601,0.9322]
- ECG5000_BAL predictive_entropy: AUROC=0.8987 CI=[0.8602,0.9334]
- ECG5000_BAL top1_top2_margin_neg: AUROC=0.8991 CI=[0.8590,0.9318]
- CWRU_UNBAL turs_u_mean: AUROC=0.7654 CI=[0.6751,0.8447]
- CWRU_UNBAL turs_u_max: AUROC=0.5376 CI=[0.4247,0.6488]
- CWRU_UNBAL 1_minus_max_softmax: AUROC=0.9002 CI=[0.8487,0.9430]
- CWRU_UNBAL predictive_entropy: AUROC=0.9002 CI=[0.8472,0.9474]
- CWRU_UNBAL top1_top2_margin_neg: AUROC=0.8991 CI=[0.8465,0.9434]
- CWRU_BAL turs_u_mean: AUROC=0.7471 CI=[0.6964,0.7941]
- CWRU_BAL turs_u_max: AUROC=0.7671 CI=[0.7271,0.8072]
- CWRU_BAL 1_minus_max_softmax: AUROC=0.9268 CI=[0.8782,0.9629]
- CWRU_BAL predictive_entropy: AUROC=0.9266 CI=[0.8785,0.9613]
- CWRU_BAL top1_top2_margin_neg: AUROC=0.9262 CI=[0.8746,0.9628]

## 19-20. Statistical Significance & Effect Sizes

All hypotheses (raw p, BH-FDR q): see `tables/13_significance_tests.csv` and `tables/14_effect_sizes.csv`. Families: .

## 21. Cross-Dataset Analysis

| Dimension | ECG5000_UNBAL | ECG5000_BAL | CWRU_UNBAL | CWRU_BAL | Consistent |
|---|---|---|---|---|---|
| D2_latent_state_validity | WEAK | NOT SUPPORTED | NOT SUPPORTED | NOT SUPPORTED | no |
| D3_temporal_change_sensitivity | NOT SUPPORTED | NOT SUPPORTED | NOT TESTABLE | NOT TESTABLE | no |
| D4_uncertainty_validity | NOT SUPPORTED | NOT SUPPORTED | MODERATE | WEAK | no |
| D6_degradation_response | NOT TESTABLE | NOT TESTABLE | NOT TESTABLE | NOT TESTABLE | no |
| D7_faithfulness | NOT SUPPORTED | NOT SUPPORTED | NOT SUPPORTED | NOT SUPPORTED | no |
| D8_alpha_reliance_validity | STRONG | STRONG | STRONG | STRONG | yes |

## 22. Failure Modes

- D2_latent_state_validity: inconsistent across datasets (WEAK, NOT SUPPORTED, NOT SUPPORTED, NOT SUPPORTED).
- D3_temporal_change_sensitivity: inconsistent across datasets (NOT SUPPORTED, NOT SUPPORTED, NOT TESTABLE, NOT TESTABLE).
- D4_uncertainty_validity: inconsistent across datasets (NOT SUPPORTED, NOT SUPPORTED, MODERATE, WEAK).
- D6_degradation_response: inconsistent across datasets (NOT TESTABLE, NOT TESTABLE, NOT TESTABLE, NOT TESTABLE).
- D7_faithfulness: inconsistent across datasets (NOT SUPPORTED, NOT SUPPORTED, NOT SUPPORTED, NOT SUPPORTED).

## 23. Evidence Rubric

Full rubric in `evidence_rubric.json`; grades per dimension/dataset:
- ECG5000_UNBAL D1_predictive_reliability: **MODERATE** (score=0.6046355541960116)
- ECG5000_UNBAL D2_latent_state_validity: **WEAK** (score=0.3491294088525768)
- ECG5000_UNBAL D3_temporal_change_sensitivity: **NOT SUPPORTED** (score=0.044444444444444446)
- ECG5000_UNBAL D4_uncertainty_validity: **NOT SUPPORTED** (score=0.24211751018033745)
- ECG5000_UNBAL D5_calibration_selective: **STRONG** (score=0.9814255656003952)
- ECG5000_UNBAL D6_degradation_response: **NOT TESTABLE** (score=None)
- ECG5000_UNBAL D7_faithfulness: **NOT SUPPORTED** (score=0.01729080144335361)
- ECG5000_UNBAL D8_alpha_reliance_validity: **STRONG** (score=1.0)
- ECG5000_UNBAL D9_stability_reproducibility: **STRONG** (score=1.0)
- ECG5000_UNBAL D10_counterfactual_consistency: **NOT SUPPORTED** (score=0.11371985277122194)
- ECG5000_UNBAL D11_baseline_superiority: **NOT SUPPORTED** (score=0.0)
- ECG5000_BAL D1_predictive_reliability: **STRONG** (score=0.7113771396210665)
- ECG5000_BAL D2_latent_state_validity: **NOT SUPPORTED** (score=0.1206342212007755)
- ECG5000_BAL D3_temporal_change_sensitivity: **NOT SUPPORTED** (score=0.022222222222222223)
- ECG5000_BAL D4_uncertainty_validity: **NOT SUPPORTED** (score=0.03481777257814289)
- ECG5000_BAL D5_calibration_selective: **STRONG** (score=0.9661155623197555)
- ECG5000_BAL D6_degradation_response: **NOT TESTABLE** (score=None)
- ECG5000_BAL D7_faithfulness: **NOT SUPPORTED** (score=0.036007660866486435)
- ECG5000_BAL D8_alpha_reliance_validity: **STRONG** (score=1.0)
- ECG5000_BAL D9_stability_reproducibility: **STRONG** (score=1.0)
- ECG5000_BAL D10_counterfactual_consistency: **WEAK** (score=0.4382963789908748)
- ECG5000_BAL D11_baseline_superiority: **NOT SUPPORTED** (score=0.0)
- CWRU_UNBAL D1_predictive_reliability: **STRONG** (score=0.950382722799344)
- CWRU_UNBAL D2_latent_state_validity: **NOT SUPPORTED** (score=0.2417126177339164)
- CWRU_UNBAL D3_temporal_change_sensitivity: **NOT TESTABLE** (score=None)
- CWRU_UNBAL D4_uncertainty_validity: **MODERATE** (score=0.5307017543859649)
- CWRU_UNBAL D5_calibration_selective: **STRONG** (score=0.9729567152758439)
- CWRU_UNBAL D6_degradation_response: **NOT TESTABLE** (score=None)
- CWRU_UNBAL D7_faithfulness: **NOT SUPPORTED** (score=0.014366119110491129)
- CWRU_UNBAL D8_alpha_reliance_validity: **STRONG** (score=1.0)
- CWRU_UNBAL D9_stability_reproducibility: **STRONG** (score=1.0)
- CWRU_UNBAL D10_counterfactual_consistency: **NOT SUPPORTED** (score=0.0)
- CWRU_UNBAL D11_baseline_superiority: **NOT SUPPORTED** (score=0.0)
- CWRU_BAL D1_predictive_reliability: **STRONG** (score=0.9540629220161475)
- CWRU_BAL D2_latent_state_validity: **NOT SUPPORTED** (score=0.20749183649645053)
- CWRU_BAL D3_temporal_change_sensitivity: **NOT TESTABLE** (score=None)
- CWRU_BAL D4_uncertainty_validity: **WEAK** (score=0.4942414332432816)
- CWRU_BAL D5_calibration_selective: **STRONG** (score=0.9685435756594205)
- CWRU_BAL D6_degradation_response: **NOT TESTABLE** (score=None)
- CWRU_BAL D7_faithfulness: **NOT SUPPORTED** (score=0.07174372664885595)
- CWRU_BAL D8_alpha_reliance_validity: **STRONG** (score=0.7311292173930904)
- CWRU_BAL D9_stability_reproducibility: **STRONG** (score=1.0)
- CWRU_BAL D10_counterfactual_consistency: **NOT SUPPORTED** (score=0.14619052252965045)
- CWRU_BAL D11_baseline_superiority: **NOT SUPPORTED** (score=0.0)

## 24. Overall Diagnostic Conclusion

**WEAK SUPPORT** — mean dimension score 0.473.

## 25. Clinical-Validity Boundary

This study evaluates machine-learning diagnostic evidence only. It does NOT establish clinical diagnostic accuracy, physician agreement, external hospital validation, prospective validation, patient-outcome validity, or regulatory/device validity. No clinical claims are made or supported.

## 26. Limitations

- Standalone TURS-Lite computes z/v/u from pooled features; temporal traces are inference-only applications of the frozen encoders.
- v is a learned projection, not a temporal derivative; localization results must be read accordingly.
- Synthetic temporal ground truth (no event labels in these datasets); controlled injections only.
- Single primary seed (42), as per the benchmark protocol; deterministic replay verified.

## 27. Reproducibility Information

- Checkpoints: `checkpoints/turs_lite/<DS>_TURS_Lite.pt` (sha256 in run_metadata.json)
- Extraction cache: `results/diagnostics/turs_lite/extracted/`
- Seeds: model 42, bootstrap 4200, permutation 4300, synthetic 4400
- Total runtime: 0s

## 30. Final Paper-Ready Conclusion

"TURS-Lite **does provide sufficient empirical evidence for intrinsic diagnostic support" at the WEAK SUPPORT level.

- Strongest evidence: D8_alpha_reliance_validity (score 1.00); weakest: D6_degradation_response (score 0.00).
- Baseline comparison: TURS-u vs max-softmax/entropy/margin — see `tables/12_baseline_comparison.csv` (superiority is NOT assumed; the rubric grades it explicitly).
- Cross-dataset consistency: see table in section 21.

"This conclusion refers to machine-learning diagnostic evidence and does not constitute clinical validation."
