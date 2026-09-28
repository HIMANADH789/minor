# FINAL INFERENCE INVENTORY — Haptics

| artifact | type | designation | data source | confirmatory? | method | seeds |
|---|---|---|---|---|---|---|
| artifact_inventory.json | JSON | INTERNAL | all stored assets | n/a | status classification | 42/43/44 |
| bootstrap_macro_f1.csv | CSV | SUPP | test predictions | confirmatory | paired bootstrap 10k, seed 42042 | 42/43/44 |
| paired_bootstrap_deltas.csv | CSV | MAIN | test predictions | confirmatory | paired bootstrap deltas | 42/43/44 |
| permutation_tests.csv | CSV | MAIN | test predictions | confirmatory | randomization 20k swaps + Holm | 42 + R5−M0 across seeds |
| per_class_metrics.csv / per_class_bootstrap.csv | CSV | SUPP | test predictions | exploratory | bootstrap CIs per class | 42/43/44 |
| error_overlap.csv / mcnemar_tests.csv | CSV | MAIN | test predictions | confirmatory | Jaccard/disagreement + exact binomial McNemar | 42 (3 for overlap) |
| model_agreement.csv | CSV | SUPP | test predictions | exploratory | agreement + Cohen's kappa | 42 |
| r5_seed_stability.csv | CSV | SUPP | stored scalars | descriptive | mean/SD/CV | 42/43/44 |
| r5_allocation_stability.csv | CSV | SUPP | stored scalars + multiseed json | descriptive | ρ* frequency | 42/43/44 |
| regime_statistics.csv | CSV | SUPP | stored VQ diagnostics | descriptive | occupancy entropy/perplexity | 42/43/44 |
| H_distribution.csv | CSV | INTERNAL | frozen banks + stored diagnostics | descriptive | distribution moments | 42/43/44 |
| G_vs_H_label_association.csv | CSV | SUPP | frozen banks (train+val only) | exploratory | ANOVA-F + MI | 42 |
| G_H_redundancy.csv | CSV | MAIN | frozen banks | exploratory | correlation + CKA + effective rank | 42 |
| sample_mechanism_effects.csv | CSV | SUPP | banks + predictions | exploratory | Cohen d, Cliff δ, bootstrap | 42 |
| r5_gain_vs_H.csv / regime_complexity_bins.csv | CSV | SUPP | banks + predictions | exploratory | Spearman, binned rates | 42 |
| confidence_analysis.csv | CSV | SUPP | Ridge decision values | exploratory | Brier/ECE/margin by outcome | 42 |
| G_H_GplusH_diagnostic.csv | CSV | MAIN | frozen banks (canonical Ridge) | confirmatory | single test eval each (already stored/derived) | 42 |
| effect_size_summary.csv | CSV | MAIN | all above | confirmatory | consolidated effect sizes + Holm | 42/43/44 |
| analysis_registry.json | JSON | INTERNAL | design | n/a | planned vs exploratory registry | n/a |

## Figures (results/haptics_inference/figures/, PDF + PNG)

| figure | designation | content |
|---|---|---|
| fig1_per_seed_performance | MAIN | per-seed test Macro-F1, six models, mean±SD diamonds |
| fig2_r5_vs_m0_paired_delta | MAIN | R5−M0 paired bootstrap deltas with 95% CIs per seed |
| fig3_r5_vs_m0_per_class_f1 | MAIN | per-class F1 bootstrap means ± CI, R5 vs M0 |
| fig4_error_overlap_heatmap | MAIN | Jaccard error overlap + disagreement heatmaps (seed 42) |
| fig5_r5_allocation_stability | SUPP | retained ρ* per seed with N_G:N_H labels |
| fig6_regime_occupancy_entropy | SUPP | VQ occupancy + entropy/perplexity by seed |
| fig7_G_vs_H_redundancy | MAIN | effective ranks + G-H cross-correlation summary |
| fig8_gain_vs_regime_complexity | SUPP | M0/R5 accuracy by regime-entropy quartile |
| fig9_gain_vs_H_norm | SUPP | decision-flip balance by ||H|| quintile |
| fig10_confidence_by_outcome | SUPP | Ridge confidence by outcome group |
| fig11_H_distribution_by_seed | INTERNAL | H mean±SD per seed |
| fig12_G_vs_H_label_association | SUPP | G vs H ANOVA-F/MI summary (log scale) |

Reproduce: `python -m experiments.haptics_inference_analysis.analysis` (resume-safe, cached banks at C:/temp/results) then `.report` and `.reports`. All statistics fixed-seed; test set (n=308) used for post-hoc inference only.
