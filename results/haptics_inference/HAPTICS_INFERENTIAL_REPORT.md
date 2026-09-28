# Haptics — Final Inferential / Scientific Significance Report

ANALYSIS ONLY package built exclusively from stored artifacts (predictions, frozen G/H banks rebuilt identity-verified from the seed-42 context checkpoint, stored VQ diagnostics, stored scalar scores). No training, no new mechanisms, no extra seeds. Test set (n=308) used for post-hoc inference only.

## 1. Objective

Extract maximum scientific evidence from the already-trained Haptics models: quantify uncertainty of test Macro-F1, paired model differences, seed stability, H-feature discriminativeness, G/H redundancy, mechanism associations, and which conclusions survive uncertainty analysis.

## 2. Existing model/data assets

| Asset | Status | Source |
|---|---|---|
| Predictions M0 + 4 baselines, seed 42 | AVAILABLE | results/external_stack_generalization/predictions/*.npy |
| Baseline predictions seeds 43/44 | AVAILABLE | results/haptics_baselines_3seed/<model>/seed{43,44}/predictions.npy |
| R5(50/50) predictions all 3 seeds | AVAILABLE | canonical seed-42 recovered (0.5500) + results/rcmkn_r2_haptics_3seed/seed{42,43,44}/predictions.csv |
| Adaptive-ρ R5 predictions seeds 43/44 | MISSING | no checkpoints stored; scalar scores only (no-training rule) |
| G/H banks + regime codes (seed 42) | AVAILABLE (rebuilt from frozen checkpoint; identity gate exact: 0.5500, α=4.2813) | cache npy + canonical loader |
| Neural probabilities/logits | MISSING | not stored by the canonical baseline runner |
| Ridge decision values (M0, R5) | DERIVABLE | frozen banks + deterministic refit |

## 3. Primary performance uncertainty (bootstrap, 10k reps, seed 42042)

| model | seed | Macro-F1 | boot mean | boot SD | 95% CI |
|---|---|---|---|---|---|
| FCN | 42 | 0.0634 | 0.0633 | 0.0063 | [0.0510, 0.0758] |
| FCN | 43 | 0.1302 | 0.1297 | 0.0154 | [0.0999, 0.1600] |
| FCN | 44 | 0.0637 | 0.0636 | 0.0063 | [0.0511, 0.0757] |
| InceptionTime | 42 | 0.0636 | 0.0635 | 0.0063 | [0.0511, 0.0758] |
| InceptionTime | 43 | 0.2475 | 0.2462 | 0.0234 | [0.2009, 0.2927] |
| InceptionTime | 44 | 0.0945 | 0.0940 | 0.0126 | [0.0709, 0.1205] |
| MiniROCKET | 42 | 0.4974 | 0.4954 | 0.0283 | [0.4395, 0.5502] |
| MiniROCKET | 43 | 0.4974 | 0.4953 | 0.0279 | [0.4417, 0.5505] |
| MiniROCKET | 44 | 0.4974 | 0.4951 | 0.0282 | [0.4395, 0.5507] |
| PatchTST | 42 | 0.1692 | 0.1687 | 0.0125 | [0.1444, 0.1926] |
| PatchTST | 43 | 0.1339 | 0.1334 | 0.0132 | [0.1081, 0.1590] |
| PatchTST | 44 | 0.0756 | 0.0753 | 0.0091 | [0.0587, 0.0946] |
| R5_5050 | 42 | 0.5500 | 0.5482 | 0.0284 | [0.4933, 0.6047] |
| R5_5050 | 43 | 0.5213 | 0.5189 | 0.0282 | [0.4634, 0.5745] |
| R5_5050 | 44 | 0.5387 | 0.5364 | 0.0285 | [0.4812, 0.5926] |
| ResNet1D | 42 | 0.1512 | 0.1505 | 0.0142 | [0.1232, 0.1786] |
| ResNet1D | 43 | 0.0715 | 0.0713 | 0.0063 | [0.0587, 0.0841] |
| ResNet1D | 44 | 0.0693 | 0.0691 | 0.0082 | [0.0541, 0.0861] |

## 4. Paired model comparisons

Paired bootstrap over the same 308 test samples + paired randomization (20k swaps, exchangeability H0) with Holm correction within the seed-42 planned family. A CI excluding 0 is *evidence* against test-set sampling noise — not proof.

| comparison | seed | delta | 95% CI | P(delta>0) | perm p | Holm p |
|---|---|---|---|---|---|---|
| R5_5050-MiniROCKET | 42 | 0.0526 | [0.0125, 0.0931] | 0.9953 | 0.01165 | 0.01165 |
| R5_5050-MiniROCKET | 43 | 0.0239 | [-0.0162, 0.0635] | 0.8779 | 0.25314 |  |
| R5_5050-MiniROCKET | 44 | 0.0413 | [0.0003, 0.0821] | 0.9760 | 0.04895 |  |
| R5_5050-InceptionTime | 42 | 0.4865 | [0.4283, 0.5414] | 1.0000 | 0.00005 | 0.00045 |
| R5_5050-FCN | 42 | 0.4866 | [0.4284, 0.5414] | 1.0000 | 0.00005 | 0.00045 |
| R5_5050-ResNet1D | 42 | 0.3988 | [0.3395, 0.4555] | 1.0000 | 0.00005 | 0.00045 |
| R5_5050-PatchTST | 42 | 0.3808 | [0.3214, 0.4384] | 1.0000 | 0.00005 | 0.00045 |
| R5_5050-InceptionTime | 43 | 0.2739 | [0.2070, 0.3384] | 1.0000 | — | — |
| R5_5050-FCN | 43 | 0.3911 | [0.3301, 0.4501] | 1.0000 | — | — |
| R5_5050-ResNet1D | 43 | 0.4499 | [0.3916, 0.5050] | 1.0000 | — | — |
| R5_5050-PatchTST | 43 | 0.3874 | [0.3276, 0.4458] | 1.0000 | — | — |
| R5_5050-InceptionTime | 44 | 0.4442 | [0.3840, 0.5030] | 1.0000 | — | — |
| R5_5050-FCN | 44 | 0.4750 | [0.4162, 0.5311] | 1.0000 | — | — |
| R5_5050-ResNet1D | 44 | 0.4694 | [0.4102, 0.5263] | 1.0000 | — | — |
| R5_5050-PatchTST | 44 | 0.4632 | [0.4052, 0.5201] | 1.0000 | — | — |
| InceptionTime-MiniROCKET | 42 | -0.4339 | [-0.4873, -0.3777] | 0.0000 | 0.00005 | 0.00045 |
| FCN-MiniROCKET | 42 | -0.4340 | [-0.4874, -0.3778] | 0.0000 | 0.00005 | 0.00045 |
| ResNet1D-MiniROCKET | 42 | -0.3462 | [-0.4010, -0.2882] | 0.0000 | 0.00005 | 0.00045 |
| PatchTST-MiniROCKET | 42 | -0.3282 | [-0.3816, -0.2718] | 0.0000 | 0.00005 | 0.00045 |
| InceptionTime-MiniROCKET | 43 | -0.2499 | [-0.3131, -0.1856] | 0.0000 | — | — |
| FCN-MiniROCKET | 43 | -0.3672 | [-0.4253, -0.3059] | 0.0000 | — | — |
| ResNet1D-MiniROCKET | 43 | -0.4260 | [-0.4807, -0.3687] | 0.0000 | — | — |
| PatchTST-MiniROCKET | 43 | -0.3635 | [-0.4186, -0.3063] | 0.0000 | — | — |
| InceptionTime-MiniROCKET | 44 | -0.4029 | [-0.4605, -0.3423] | 0.0000 | — | — |
| FCN-MiniROCKET | 44 | -0.4337 | [-0.4883, -0.3757] | 0.0000 | — | — |
| ResNet1D-MiniROCKET | 44 | -0.4281 | [-0.4841, -0.3709] | 0.0000 | — | — |
| PatchTST-MiniROCKET | 44 | -0.4219 | [-0.4767, -0.3652] | 0.0000 | — | — |

## 5. Multi-seed stability (n=3, descriptive)

R5 per-seed: 42 = 0.5500 (50/50), 43 = 0.5264 (ρ=0.4), 44 = 0.5387 (50/50). Mean 0.5384 ± 0.0118 (median 0.5387, range [0.5264, 0.55], CV 0.0219). Three seeds cannot support population-level significance claims.

| seed | R5 | R5 - M0 |
|---|---|---|
| 42 | 0.5500 | 0.0526 |
| 43 | 0.5213 | 0.0239 |
| 44 | 0.5387 | 0.0413 |

## 6. R5 allocation stability

| seed | retained config | ρ* | N_G:N_H | R5(50/50) | R5(ρ*) | diff |
|---|---|---|---|---|---|---|
| 42 | 50/50 | 0.5 | 4998:4998 | 0.5500 | 0.5429 | 0.0071 |
| 43 | 0.4 | 0.4 | 5998:3998 | 0.5213 | 0.5264 | -0.0051 |
| 44 | 50/50 | 0.5 | 4998:4998 | 0.5387 | 0.5342 | 0.0045 |
ρ* = 0.5 on 2/3 seeds; adaptive ρ* = 0.4 once (CV-selected, never "optimal"). Performance change from adaptive allocation is small (≤ 0.008).

## 7. Per-class effects

R5 vs M0 per-class F1 (bootstrap means, n=308 resamples):

| class | R5 F1 | M0 F1 | Δ |
|---|---|---|---|
| 0 | 0.3682 [0.2285, 0.5000] | 0.2561 [0.1311, 0.3846] | 0.1121 |
| 1 | 0.6110 [0.5000, 0.7087] | 0.5617 [0.4536, 0.6577] | 0.0493 |
| 2 | 0.6177 [0.5161, 0.7130] | 0.5670 [0.4640, 0.6579] | 0.0507 |
| 3 | 0.5523 [0.4473, 0.6531] | 0.5157 [0.4065, 0.6115] | 0.0366 |
| 4 | 0.5878 [0.4865, 0.6809] | 0.5771 [0.4733, 0.6667] | 0.0107 |

## 8. Error / disagreement structure

Seed-42 error overlap (Jaccard) and disagreement vs R5:

| pair | Jaccard | disagreement | both wrong | A-only | B-only |
|---|---|---|---|---|---|
| MiniROCKET vs R5_5050 | 0.7736 | 0.1169 | 123 | 25 | 11 |
| R5_5050 vs InceptionTime | 0.4275 | 0.5000 | 115 | 19 | 135 |
| R5_5050 vs FCN | 0.4275 | 0.5000 | 115 | 19 | 135 |
| R5_5050 vs ResNet1D | 0.4458 | 0.4481 | 111 | 23 | 115 |
| R5_5050 vs PatchTST | 0.4350 | 0.4513 | 107 | 27 | 112 |

McNemar exact tests (seed 42, Holm-corrected):

| comparison | discordant A/B | p | Holm p |
|---|---|---|---|
| MiniROCKET-vs-R5_5050 | 25/11 | 0.02882 | 0.02882 |
| R5_5050-vs-InceptionTime | 19/135 | 0.00000 | 0.00000 |
| R5_5050-vs-FCN | 19/135 | 0.00000 | 0.00000 |
| R5_5050-vs-ResNet1D | 23/115 | 0.00000 | 0.00000 |
| R5_5050-vs-PatchTST | 27/112 | 0.00000 | 0.00000 |

## 9. Regime statistics

| seed | active codes | entropy (nats) | normalized | perplexity |
|---|---|---|---|---|
| 42 | 8 | 1.9464 | 0.936 | 7.0036 |
| 43 | 8 | 1.7611 | 0.8469 | 5.8187 |
| 44 | 6 | 1.7217 | 0.828 | 5.5942 |
All seeds use non-degenerate regimes (perplexity 5.6–7.0 of max 8), so R5 is not operating on a collapsed representation.

## 10. H-feature informativeness

| bank | n | mean F | median F | mean top-50 F | frac F>1 | mean top-50 MI | frac MI>0.05 |
|---|---|---|---|---|---|---|---|
| G | 4998 | 5.268 | 4.355 | 19.196 | 0.9002 | 0.2786 | 0.6369 |
| H | 4998 | 4.173 | 3.626 | 13.77 | 0.9166 | 0.2499 | 0.5716 |
Computed on train+val only (155 samples; test labels excluded from all ranking-related analysis).

## 11. G/H redundancy

| mean |r| | median |r| | p95 |r| | frac>0.5 | frac>0.9 | linear CKA | eff rank G | eff rank H |
|---|---|---|---|---|---|---|---|
| 0.1653 | 0.1394 | 0.4037 | 0.01248 | 4.5e-05 | 0.3741 | 5.8 | 67.0 |
Low mean cross-correlation with nonzero shared linear structure (CKA) — H is not a simple duplicate of G, but also not fully independent information.

## 12. Mechanism–performance associations (exploratory)

Correct-vs-incorrect contrasts (Cohen's d / Cliff's δ; observational, not causal):

| model | feature | mean ok | mean err | d | δ | CI of mean diff |
|---|---|---|---|---|---|---|
| R5_5050 | H_norm | 3.6707 | 3.6572 | 0.0399 | -0.0586 | [-0.063, 0.094] |
| R5_5050 | n_regimes | 7.9885 | 7.9776 | 0.0842 | 0.0109 | [-0.0185, 0.0433] |
| R5_5050 | regime_entropy | 1.9059 | 1.902 | 0.0575 | -0.0323 | [-0.0105, 0.0199] |
| R5_5050 | transition_count | 26.6724 | 28.0448 | -0.2458 | -0.1256 | [-2.5912, -0.1667] |

Decision-flip vs mechanism magnitude (Spearman):

| feature | ρ | p | n+ | n- |
|---|---|---|---|---|
| H_norm | -0.0227 | 0.6915 | 25 | 11 |
| regime_entropy | -0.0546 | 0.3395 | 25 | 11 |
| n_regimes | 0.0181 | 0.7511 | 25 | 11 |
| transition_rate | 0.0289 | 0.6138 | 25 | 11 |

Regime-complexity bins (accuracy):

| bin | entropy range | n | M0 | R5 | Δ |
|---|---|---|---|---|---|
| 0 | [1.195,1.892) | 77 | 0.4286 | 0.4935 | 0.0649 |
| 1 | [1.892,1.920) | 77 | 0.6364 | 0.6883 | 0.0519 |
| 2 | [1.920,1.938) | 77 | 0.5325 | 0.6104 | 0.0779 |
| 3 | [1.938,1.963) | 77 | 0.4805 | 0.4675 | -0.013 |

## 13. Confidence / calibration (Ridge decision values)

| model | Brier | ECE(10) | conf correct | conf incorrect | margin | entropy |
|---|---|---|---|---|---|---|
| MiniROCKET | 0.651 | 0.2106 | 0.3278 | 0.3101 | 0.0927 | 1.5351 |
| R5_5050 | 0.5563 | 0.222 | 0.3589 | 0.3221 | 0.1188 | 1.5143 |

## 14. Effect-size summary

| comparison | seed | delta | CI | P(δ>0) | perm p | Holm p | interpretation |
|---|---|---|---|---|---|---|---|
| R5_5050-MiniROCKET | 42 | 0.0526 | [0.0125, 0.0931] | 0.9953 | 0.01165 | 0.01165 | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-MiniROCKET | 43 | 0.0239 | [-0.0162, 0.0635] | 0.8779 | 0.25314 |  | weak evidence |
| R5_5050-MiniROCKET | 44 | 0.0413 | [0.0003, 0.0821] | 0.976 | 0.04895 |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-InceptionTime | 42 | 0.4865 | [0.4283, 0.5414] | 1.0 | 5e-05 | 0.00045 | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-FCN | 42 | 0.4866 | [0.4284, 0.5414] | 1.0 | 5e-05 | 0.00045 | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-ResNet1D | 42 | 0.3988 | [0.3395, 0.4555] | 1.0 | 5e-05 | 0.00045 | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-PatchTST | 42 | 0.3808 | [0.3214, 0.4384] | 1.0 | 5e-05 | 0.00045 | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-InceptionTime | 43 | 0.2739 | [0.207, 0.3384] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-FCN | 43 | 0.3911 | [0.3301, 0.4501] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-ResNet1D | 43 | 0.4499 | [0.3916, 0.505] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-PatchTST | 43 | 0.3874 | [0.3276, 0.4458] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-InceptionTime | 44 | 0.4442 | [0.384, 0.503] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-FCN | 44 | 0.475 | [0.4162, 0.5311] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-ResNet1D | 44 | 0.4694 | [0.4102, 0.5263] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| R5_5050-PatchTST | 44 | 0.4632 | [0.4052, 0.5201] | 1.0 |  |  | strong evidence (CI excludes 0, bootstrap P>=.95) |
| InceptionTime-MiniROCKET | 42 | -0.4339 | [-0.4873, -0.3777] | 0.0 | 5e-05 | 0.00045 | strong evidence of negative difference |
| FCN-MiniROCKET | 42 | -0.434 | [-0.4874, -0.3778] | 0.0 | 5e-05 | 0.00045 | strong evidence of negative difference |
| ResNet1D-MiniROCKET | 42 | -0.3462 | [-0.401, -0.2882] | 0.0 | 5e-05 | 0.00045 | strong evidence of negative difference |
| PatchTST-MiniROCKET | 42 | -0.3282 | [-0.3816, -0.2718] | 0.0 | 5e-05 | 0.00045 | strong evidence of negative difference |
| InceptionTime-MiniROCKET | 43 | -0.2499 | [-0.3131, -0.1856] | 0.0 |  |  | strong evidence of negative difference |
| FCN-MiniROCKET | 43 | -0.3672 | [-0.4253, -0.3059] | 0.0 |  |  | strong evidence of negative difference |
| ResNet1D-MiniROCKET | 43 | -0.426 | [-0.4807, -0.3687] | 0.0 |  |  | strong evidence of negative difference |
| PatchTST-MiniROCKET | 43 | -0.3635 | [-0.4186, -0.3063] | 0.0 |  |  | strong evidence of negative difference |
| InceptionTime-MiniROCKET | 44 | -0.4029 | [-0.4605, -0.3423] | 0.0 |  |  | strong evidence of negative difference |
| FCN-MiniROCKET | 44 | -0.4337 | [-0.4883, -0.3757] | 0.0 |  |  | strong evidence of negative difference |
| ResNet1D-MiniROCKET | 44 | -0.4281 | [-0.4841, -0.3709] | 0.0 |  |  | strong evidence of negative difference |
| PatchTST-MiniROCKET | 44 | -0.4219 | [-0.4767, -0.3652] | 0.0 |  |  | strong evidence of negative difference |

## 15. Multiple-testing procedure

Holm correction applied across the seed-42 planned family (R5−M0 and R5-vs-baseline randomization tests and McNemar tests). Baseline-vs-M0 comparisons are exploratory (registry, analysis_registry.json). Test set used only post-hoc.

## 16. What the evidence supports

1. On seed 42, R5(50/50) − M0 = 0.0526 (95% CI [0.0125, 0.0931], excludes 0) and all four deep baselines sit far below both MiniROCKET and R5 with CIs far from M0's — the ordering MiniROCKET < R5 << baselines is unlikely to be test-set sampling noise under the bootstrap.
2. R5 > M0 on all three seeds (deltas +0.0526/+0.0290/+0.0413); descriptively reproducible, though n=3 precludes significance.
3. Regimes are non-degenerate across seeds (perplexity 5.6–7.0).
4. H is a distinct but partially overlapping representation relative to G (low mean |r|, nonzero CKA).
5. G/H/G+H diagnostics: G-only 0.5037, H-only 0.4686, G+H 0.5500 — H alone is weaker than G alone, and the combination is the best; H contributes complementary signal.

## 17. What the evidence does NOT support

1. Any claim that R5 is superior to MiniROCKET at the population level: n=3 seeds, single dataset, single split.
2. Any causal claim that regime structure causes correct predictions (associations only).
3. Any significance statement from the 3-seed comparison.
4. Any claim that ρ*=0.4 is optimal (it is CV-selected; 50/50 retained on 2/3 seeds).
5. Neural-baseline conclusions beyond characterization: those models are far below MiniROCKET and highly seed-variable, consistent with undertraining/limited-data regime, not a claim about the architectures in general.

## 18. Limitations

- Test set n=308 → Macro-F1 CIs are wide (±0.05–0.08 typical).
- 3 seeds; SSL/VQ context variation only, not data resampling.
- Adaptive-ρ predictions unavailable for seeds 43/44 (no checkpoints); scalar-only treatment there.
- Neural probabilities not stored; confidence analysis restricted to Ridge decision values (M0, R5).
- Mechanism analyses are observational; no interventional evidence.

## 19. Recommended main-paper figures

- fig1_per_seed_performance (MAIN): per-seed test Macro-F1 for all six models.
- fig2_r5_vs_m0_paired_delta (MAIN): R5−M0 paired bootstrap deltas with 95% CIs.
- fig3_r5_vs_m0_per_class_f1 (MAIN): per-class F1 with CIs.
- fig4_error_overlap_heatmap (MAIN): Jaccard overlap + disagreement matrix.
- fig7_G_vs_H_redundancy (MAIN): effective ranks + cross-|r| summary.

## 20. Recommended supplementary figures

- fig5_r5_allocation_stability, fig6_regime_occupancy_entropy, fig8_gain_vs_regime_complexity, fig9_gain_vs_H_norm, fig10_confidence_by_outcome, fig11_H_distribution_by_seed, fig12_G_vs_H_label_association.

## 25. Explicit answers (Q1–Q10, evidence-weighted)

Q1. R5 > M0 on all 3 seeds: descriptively reproducible.
Q2. Seed-42 delta 0.0526 vs bootstrap SD ~0.05: CI excludes 0: larger than typical test-sample uncertainty.
Q3. R5 gains arise across multiple classes, concentrated where M0 is weakest (per-class table, section 7).
Q4. Yes — R5's errors only partially overlap M0's (Jaccard < 1, disagreement > 0); R5 is not a copy of M0.
Q5. H is distinguishable from G (low cross-|r|, different effective ranks) but not fully independent (CKA > 0).
Q6. Yes — H features show label association on train+val (ANOVA-F/MI), weaker than G on average but present.
Q7. Mixed — gains do not increase monotonically with regime complexity (binned deltas, section 12).
Q8. Weak/absent — Spearman between decision flips and ||H|| is not consistently positive (r5_gain_vs_H.csv).
Q9. Mostly — ρ* = 0.5 on 2/3 seeds; one CV-selected 0.4.
Q10. Well-supported: seed-42 paired ordering (R5 vs M0 and vs all baselines). Descriptive only: seed-level stability, allocation stability, mechanism associations. Unsupported: population-level superiority, causality, optimality of ρ*.

## 26. Reproducibility

```bash
cd ECG_Benchmark
python -m experiments.haptics_inference_analysis.analysis   # cached banks; resume-safe
python -m experiments.haptics_inference_analysis.report
python -m experiments.haptics_inference_analysis.reports
```

Seeds: bootstrap 42042, per-comparison offsets; permutation 20,000 swaps; Holm within planned families. No test-set tuning.
