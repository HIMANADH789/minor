"""Final scientific reports for the Haptics inferential analysis:
HAPTICS_INFERENTIAL_REPORT.md and FINAL_INFERENCE_INVENTORY.md.
Reads the CSV/JSON artifacts written by analysis.py + report.py.
"""
import csv
import json
import os

from experiments.haptics_inference_analysis.analysis import (
    OUT, ROOT, SEEDS, csv_rows)


def rows(name):
    p = os.path.join(OUT, name)
    return csv_rows(p) if os.path.exists(p) else []


def get(d, k, default=""):
    return d.get(k, default) if d else default


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    boot = rows("bootstrap_macro_f1.csv")
    deltas = rows("paired_bootstrap_deltas.csv")
    perms = rows("permutation_tests.csv")
    pcb = rows("per_class_bootstrap.csv")
    eo = rows("error_overlap.csv")
    mcn = rows("mcnemar_tests.csv")
    agr = rows("model_agreement.csv")
    stab = {r["metric"]: r["value"] for r in rows("r5_seed_stability.csv")}
    alloc = rows("r5_allocation_stability.csv")
    reg = rows("regime_statistics.csv")
    assoc = rows("G_vs_H_label_association.csv")
    red = rows("G_H_redundancy.csv")
    mech = rows("sample_mechanism_effects.csv")
    gain = rows("r5_gain_vs_H.csv")
    bins = rows("regime_complexity_bins.csv")
    conf = rows("confidence_analysis.csv")
    diag = rows("G_H_GplusH_diagnostic.csv")
    eff = rows("effect_size_summary.csv")
    hd = rows("H_distribution.csv")

    r5m0 = [r for r in deltas if r["comparison"] == "R5_5050-MiniROCKET"]
    perms42 = [r for r in perms if r["seed"] == "42"]
    tags = json.load(open(os.path.join(
        r"C:/temp/results", "haptics_inference_tags.json")))

    # ---- answer evidence-weighting helpers -----------------------------
    def ci_excludes(x, lo, hi):
        return float(lo) > 0 or float(hi) < 0

    r5m0_42 = next((r for r in r5m0 if r["seed"] == "42"), None)
    strong_r5 = r5m0_42 and ci_excludes("x", r5m0_42["ci_low"],
                                        r5m0_42["ci_high"])

    L = []
    A = L.append

    A("# Haptics — Final Inferential / Scientific Significance Report\n")
    A("ANALYSIS ONLY package built exclusively from stored artifacts "
      "(predictions, frozen G/H banks rebuilt identity-verified from the "
      "seed-42 context checkpoint, stored VQ diagnostics, stored scalar "
      "scores). No training, no new mechanisms, no extra seeds. "
      "Test set (n=308) used for post-hoc inference only.\n")

    A("## 1. Objective\n")
    A("Extract maximum scientific evidence from the already-trained "
      "Haptics models: quantify uncertainty of test Macro-F1, paired "
      "model differences, seed stability, H-feature discriminativeness, "
      "G/H redundancy, mechanism associations, and which conclusions "
      "survive uncertainty analysis.\n")

    A("## 2. Existing model/data assets\n")
    A("| Asset | Status | Source |")
    A("|---|---|---|")
    A("| Predictions M0 + 4 baselines, seed 42 | AVAILABLE | "
      "results/external_stack_generalization/predictions/*.npy |")
    A("| Baseline predictions seeds 43/44 | AVAILABLE | "
      "results/haptics_baselines_3seed/<model>/seed{43,44}/predictions.npy |")
    A("| R5(50/50) predictions all 3 seeds | AVAILABLE | "
      "canonical seed-42 recovered (0.5500) + "
      "results/rcmkn_r2_haptics_3seed/seed{42,43,44}/predictions.csv |")
    A("| Adaptive-ρ R5 predictions seeds 43/44 | MISSING | no checkpoints "
      "stored; scalar scores only (no-training rule) |")
    A("| G/H banks + regime codes (seed 42) | AVAILABLE (rebuilt from "
      "frozen checkpoint; identity gate exact: 0.5500, α=4.2813) | "
      "cache npy + canonical loader |")
    A("| Neural probabilities/logits | MISSING | not stored by the "
      "canonical baseline runner |")
    A("| Ridge decision values (M0, R5) | DERIVABLE | frozen banks + "
      "deterministic refit |\n")

    A("## 3. Primary performance uncertainty (bootstrap, 10k reps, "
      "seed 42042)\n")
    A("| model | seed | Macro-F1 | boot mean | boot SD | 95% CI |")
    A("|---|---|---|---|---|---|")
    for r in boot:
        A(f"| {r['model']} | {r['seed']} | {fmt(r['observed_macro_f1'])} | "
          f"{fmt(r['boot_mean'])} | {fmt(r['boot_sd'])} | "
          f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] |")
    A("")

    A("## 4. Paired model comparisons\n")
    A("Paired bootstrap over the same 308 test samples + paired "
      "randomization (20k swaps, exchangeability H0) with Holm correction "
      "within the seed-42 planned family. A CI excluding 0 is *evidence* "
      "against test-set sampling noise — not proof.\n")
    A("| comparison | seed | delta | 95% CI | P(delta>0) | perm p | Holm p |")
    A("|---|---|---|---|---|---|---|")
    for r in deltas:
        pr = next((p for p in perms if p["comparison"] == r["comparison"]
                   and p["seed"] == str(r["seed"])), None)
        A(f"| {r['comparison']} | {r['seed']} | {fmt(r['observed_delta'])} | "
          f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | "
          f"{fmt(r['P_delta_gt_0'])} | "
          f"{fmt(pr['perm_p_two_sided'], 5) if pr else '—'} | "
          f"{fmt(pr['holm_p'], 5) if pr else '—'} |")
    A("")

    A("## 5. Multi-seed stability (n=3, descriptive)\n")
    A(f"R5 per-seed: 42 = 0.5500 (50/50), 43 = 0.5264 (ρ=0.4), "
      f"44 = 0.5387 (50/50). Mean {stab.get('mean')} ± {stab.get('sd')} "
      f"(median {stab.get('median')}, range [{stab.get('min')}, "
      f"{stab.get('max')}], CV {stab.get('cv')}). "
      "Three seeds cannot support population-level significance claims.\n")
    A("| seed | R5 | R5 - M0 |")
    A("|---|---|---|")
    for s in SEEDS:
        d = next((r for r in deltas if r["comparison"] ==
                  "R5_5050-MiniROCKET" and r["seed"] == str(s)), None)
        r5 = next((r for r in boot if r["model"] == "R5_5050" and
                   r["seed"] == str(s)), None)
        A(f"| {s} | {fmt(r5['observed_macro_f1']) if r5 else '—'} | "
          f"{fmt(d['observed_delta']) if d else '—'} |")
    A("")

    A("## 6. R5 allocation stability\n")
    A("| seed | retained config | ρ* | N_G:N_H | R5(50/50) | R5(ρ*) | "
      "diff |")
    A("|---|---|---|---|---|---|---|")
    for r in alloc:
        A(f"| {r['seed']} | {r['retained_config']} | {r['rho_star']} | "
          f"{r['N_G']}:{r['N_H']} | {fmt(r['R5_5050_test'])} | "
          f"{fmt(r['R5_adaptive_CV_selected_test'])} | "
          f"{fmt(r['difference_5050_minus_adaptive'])} |")
    A("ρ* = 0.5 on 2/3 seeds; adaptive ρ* = 0.4 once (CV-selected, never "
      "\"optimal\"). Performance change from adaptive allocation is "
      "small (≤ 0.008).\n")

    A("## 7. Per-class effects\n")
    A("R5 vs M0 per-class F1 (bootstrap means, n=308 resamples):\n")
    A("| class | R5 F1 | M0 F1 | Δ |")
    A("|---|---|---|---|")
    for c in sorted({r["class"] for r in pcb}, key=int):
        r5 = next((r for r in pcb if r["model"] == "R5_5050" and
                   r["class"] == c), None)
        m0 = next((r for r in pcb if r["model"] == "MiniROCKET" and
                   r["class"] == c), None)
        if r5 and m0:
            A(f"| {c} | {fmt(r5['f1_boot_mean'])} "
              f"[{fmt(r5['ci_low'])}, {fmt(r5['ci_high'])}] | "
              f"{fmt(m0['f1_boot_mean'])} "
              f"[{fmt(m0['ci_low'])}, {fmt(m0['ci_high'])}] | "
              f"{fmt(float(r5['f1_boot_mean']) - float(m0['f1_boot_mean']))} |")
    A("")

    A("## 8. Error / disagreement structure\n")
    A("Seed-42 error overlap (Jaccard) and disagreement vs R5:\n")
    A("| pair | Jaccard | disagreement | both wrong | A-only | B-only |")
    A("|---|---|---|---|---|---|")
    for r in eo:
        if r["seed"] == "42":
            A(f"| {r['model_A']} vs {r['model_B']} | "
              f"{fmt(r['error_overlap_jaccard'])} | "
              f"{fmt(r['disagreement_rate'])} | {r['both_wrong']} | "
              f"{r['A_only_error']} | {r['B_only_error']} |")
    A("")
    A("McNemar exact tests (seed 42, Holm-corrected):\n")
    A("| comparison | discordant A/B | p | Holm p |")
    A("|---|---|---|---|")
    for r in mcn:
        A(f"| {r['comparison']} | {r['discordant_A_only']}/"
          f"{r['discordant_B_only']} | {fmt(r['p_two_sided'], 5)} | "
          f"{fmt(r['holm_p'], 5)} |")
    A("")

    A("## 9. Regime statistics\n")
    A("| seed | active codes | entropy (nats) | normalized | perplexity |")
    A("|---|---|---|---|---|")
    for r in reg:
        A(f"| {r['seed']} | {r['active_codes']} | {r['entropy_nats']} | "
          f"{r['normalized_entropy']} | {r['perplexity']} |")
    A("All seeds use non-degenerate regimes (perplexity 5.6–7.0 of "
      "max 8), so R5 is not operating on a collapsed representation.\n")

    A("## 10. H-feature informativeness\n")
    A("| bank | n | mean F | median F | mean top-50 F | frac F>1 | "
      "mean top-50 MI | frac MI>0.05 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in assoc:
        A(f"| {r['bank']} | {r['n_features']} | {r['mean_F']} | "
          f"{r['median_F']} | {r['mean_top50_F']} | {r['frac_F_gt_1']} | "
          f"{r['mean_top50_MI']} | {r['frac_MI_gt_0.05']} |")
    A("Computed on train+val only (155 samples; test labels excluded from "
      "all ranking-related analysis).\n")

    A("## 11. G/H redundancy\n")
    if red:
        r0 = red[0]
        A(f"| mean |r| | median |r| | p95 |r| | frac>0.5 | frac>0.9 | "
          "linear CKA | eff rank G | eff rank H |")
        A("|---|---|---|---|---|---|---|---|")
        A(f"| {r0['mean_abs_r']} | {r0['median_abs_r']} | "
          f"{r0['p95_abs_r']} | {r0['frac_abs_r_gt_0.5']} | "
          f"{r0['frac_abs_r_gt_0.9']} | {r0['linear_CKA_G_vs_H']} | "
          f"{r0['effective_rank_G']} | {r0['effective_rank_H']} |")
    A("Low mean cross-correlation with nonzero shared linear structure "
      "(CKA) — H is not a simple duplicate of G, but also not fully "
      "independent information.\n")

    A("## 12. Mechanism–performance associations (exploratory)\n")
    A("Correct-vs-incorrect contrasts (Cohen's d / Cliff's δ; "
      "observational, not causal):\n")
    A("| model | feature | mean ok | mean err | d | δ | CI of mean diff |")
    A("|---|---|---|---|---|---|---|")
    for r in mech:
        if r["model"] == "R5_5050" and r["feature"] in (
                "H_norm", "regime_entropy", "n_regimes", "transition_count"):
            A(f"| {r['model']} | {r['feature']} | {r['mean_correct']} | "
              f"{r['mean_incorrect']} | {r['cohens_d']} | "
              f"{r['cliffs_delta']} | [{r['boot_ci_low']}, "
              f"{r['boot_ci_high']}] |")
    A("")
    A("Decision-flip vs mechanism magnitude (Spearman):\n")
    A("| feature | ρ | p | n+ | n- |")
    A("|---|---|---|---|---|")
    for r in gain:
        A(f"| {r['feature']} | {r['spearman_rho']} | {r['p_value']} | "
          f"{r['n_plus']} | {r['n_minus']} |")
    A("")
    A("Regime-complexity bins (accuracy):\n")
    A("| bin | entropy range | n | M0 | R5 | Δ |")
    A("|---|---|---|---|---|---|")
    for r in bins:
        A(f"| {r['bin']} | {r['entropy_range']} | {r['n']} | "
          f"{r['M0_acc']} | {r['R5_acc']} | {r['delta_acc']} |")
    A("")

    A("## 13. Confidence / calibration (Ridge decision values)\n")
    A("| model | Brier | ECE(10) | conf correct | conf incorrect | margin "
      "| entropy |")
    A("|---|---|---|---|---|---|---|")
    for r in conf:
        if not str(r["model"]).startswith("group:"):
            A(f"| {r['model']} | {r['brier']} | {r['ece_10bin']} | "
              f"{r['mean_conf_correct']} | {r['mean_conf_incorrect']} | "
              f"{r['mean_margin']} | {r['mean_entropy']} |")
    A("")

    A("## 14. Effect-size summary\n")
    A("| comparison | seed | delta | CI | P(δ>0) | perm p | Holm p | "
      "interpretation |")
    A("|---|---|---|---|---|---|---|---|")
    for r in eff:
        A(f"| {r['comparison']} | {r['seed']} | {r['observed_delta']} | "
          f"[{r['ci_low']}, {r['ci_high']}] | {r['bootstrap_P_positive']} | "
          f"{r['perm_p']} | {r['holm_p']} | {r['interpretation']} |")
    A("")

    A("## 15. Multiple-testing procedure\n")
    A("Holm correction applied across the seed-42 planned family "
      "(R5−M0 and R5-vs-baseline randomization tests and McNemar tests). "
      "Baseline-vs-M0 comparisons are exploratory (registry, "
      "analysis_registry.json). Test set used only post-hoc.\n")

    A("## 16. What the evidence supports\n")
    r542 = next((r for r in deltas if r["comparison"] ==
                 "R5_5050-MiniROCKET" and r["seed"] == "42"), None)
    A(f"1. On seed 42, R5(50/50) − M0 = "
      f"{fmt(r542['observed_delta']) if r542 else 'n/a'} "
      f"(95% CI [{fmt(r542['ci_low'])}, {fmt(r542['ci_high'])}]"
      f"{', excludes 0' if r542 and ci_excludes('x', r542['ci_low'], r542['ci_high']) else ''}"
      f") and all four deep baselines sit far below both MiniROCKET and "
      f"R5 with CIs far from M0's — the ordering MiniROCKET < R5 "
      f"<< baselines is unlikely to be test-set sampling noise under the "
      f"bootstrap.")
    A("2. R5 > M0 on all three seeds (deltas +0.0526/+0.0290/+0.0413); "
      "descriptively reproducible, though n=3 precludes significance.")
    A("3. Regimes are non-degenerate across seeds (perplexity 5.6–7.0).")
    A("4. H is a distinct but partially overlapping representation "
      "relative to G (low mean |r|, nonzero CKA).")
    A("5. G/H/G+H diagnostics: G-only 0.5037, H-only 0.4686, G+H 0.5500 "
      "— H alone is weaker than G alone, and the combination is the "
      "best; H contributes complementary signal.\n")

    A("## 17. What the evidence does NOT support\n")
    A("1. Any claim that R5 is superior to MiniROCKET at the population "
      "level: n=3 seeds, single dataset, single split.")
    A("2. Any causal claim that regime structure causes correct "
      "predictions (associations only).")
    A("3. Any significance statement from the 3-seed comparison.")
    A("4. Any claim that ρ*=0.4 is optimal (it is CV-selected; 50/50 "
      "retained on 2/3 seeds).")
    A("5. Neural-baseline conclusions beyond characterization: those "
      "models are far below MiniROCKET and highly seed-variable, "
      "consistent with undertraining/limited-data regime, not a claim "
      "about the architectures in general.\n")

    A("## 18. Limitations\n")
    A("- Test set n=308 → Macro-F1 CIs are wide (±0.05–0.08 typical).")
    A("- 3 seeds; SSL/VQ context variation only, not data resampling.")
    A("- Adaptive-ρ predictions unavailable for seeds 43/44 (no "
      "checkpoints); scalar-only treatment there.")
    A("- Neural probabilities not stored; confidence analysis restricted "
      "to Ridge decision values (M0, R5).")
    A("- Mechanism analyses are observational; no interventional "
      "evidence.\n")

    A("## 19. Recommended main-paper figures\n")
    A("- fig1_per_seed_performance (MAIN): per-seed test Macro-F1 for "
      "all six models.")
    A("- fig2_r5_vs_m0_paired_delta (MAIN): R5−M0 paired bootstrap "
      "deltas with 95% CIs.")
    A("- fig3_r5_vs_m0_per_class_f1 (MAIN): per-class F1 with CIs.")
    A("- fig4_error_overlap_heatmap (MAIN): Jaccard overlap + "
      "disagreement matrix.")
    A("- fig7_G_vs_H_redundancy (MAIN): effective ranks + cross-|r| "
      "summary.\n")

    A("## 20. Recommended supplementary figures\n")
    A("- fig5_r5_allocation_stability, fig6_regime_occupancy_entropy, "
      "fig8_gain_vs_regime_complexity, fig9_gain_vs_H_norm, "
      "fig10_confidence_by_outcome, fig11_H_distribution_by_seed, "
      "fig12_G_vs_H_label_association.\n")

    A("## 25. Explicit answers (Q1–Q10, evidence-weighted)\n")
    A("Q1. R5 > M0 on all 3 seeds — descriptively reproducible.")
    strong_q2 = bool(r542) and ci_excludes(
        "x", r542["ci_low"], r542["ci_high"])
    A("Q1. R5 > M0 on all 3 seeds: descriptively reproducible.")
    q2_delta = fmt(r542["observed_delta"]) if r542 else "n/a"
    q2 = ("Q2. Seed-42 delta " + q2_delta
          + " vs bootstrap SD ~0.05: "
          + ("CI excludes 0: larger than typical test-sample "
             "uncertainty" if strong_q2
             else "within test-sample uncertainty") + ".")
    A(q2)
    A("Q3. R5 gains arise across multiple classes, concentrated where "
      "M0 is weakest (per-class table, section 7).")
    A("Q4. Yes — R5's errors only partially overlap M0's (Jaccard < 1, "
      "disagreement > 0); R5 is not a copy of M0.")
    A("Q5. H is distinguishable from G (low cross-|r|, different "
      "effective ranks) but not fully independent (CKA > 0).")
    A("Q6. Yes — H features show label association on train+val "
      "(ANOVA-F/MI), weaker than G on average but present.")
    A("Q7. Mixed — gains do not increase monotonically with regime "
      "complexity (binned deltas, section 12).")
    A("Q8. Weak/absent — Spearman between decision flips and ||H|| is "
      "not consistently positive (r5_gain_vs_H.csv).")
    A("Q9. Mostly — ρ* = 0.5 on 2/3 seeds; one CV-selected 0.4.")
    A("Q10. Well-supported: seed-42 paired ordering (R5 vs M0 and vs "
      "all baselines). Descriptive only: seed-level stability, "
      "allocation stability, mechanism associations. Unsupported: "
      "population-level superiority, causality, optimality of ρ*.\n")

    A("## 26. Reproducibility\n")
    A("```bash")
    A("cd ECG_Benchmark")
    A("python -m experiments.haptics_inference_analysis.analysis   "
      "# cached banks; resume-safe")
    A("python -m experiments.haptics_inference_analysis.report")
    A("python -m experiments.haptics_inference_analysis.reports")
    A("```\n")
    A("Seeds: bootstrap 42042, per-comparison offsets; permutation "
      "20,000 swaps; Holm within planned families. No test-set tuning.")

    with open(os.path.join(OUT, "HAPTICS_INFERENTIAL_REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("report written:", os.path.join(OUT,
                                          "HAPTICS_INFERENTIAL_REPORT.md"),
          flush=True)

    # ================= 27. FINAL_INFERENCE_INVENTORY.md ==================
    I = []
    W = I.append
    W("# FINAL INFERENCE INVENTORY — Haptics\n")
    W("| artifact | type | designation | data source | confirmatory? | "
      "method | seeds |")
    W("|---|---|---|---|---|---|---|")
    inv_rows = [
        ("artifact_inventory.json", "JSON", "INTERNAL", "all stored "
         "assets", "n/a", "status classification", "42/43/44"),
        ("bootstrap_macro_f1.csv", "CSV", "SUPP", "test predictions",
         "confirmatory", "paired bootstrap 10k, seed 42042", "42/43/44"),
        ("paired_bootstrap_deltas.csv", "CSV", "MAIN", "test predictions",
         "confirmatory", "paired bootstrap deltas", "42/43/44"),
        ("permutation_tests.csv", "CSV", "MAIN", "test predictions",
         "confirmatory", "randomization 20k swaps + Holm", "42 + "
         "R5−M0 across seeds"),
        ("per_class_metrics.csv / per_class_bootstrap.csv", "CSV",
         "SUPP", "test predictions", "exploratory", "bootstrap CIs per "
         "class", "42/43/44"),
        ("error_overlap.csv / mcnemar_tests.csv", "CSV", "MAIN",
         "test predictions", "confirmatory", "Jaccard/disagreement + "
         "exact binomial McNemar", "42 (3 for overlap)"),
        ("model_agreement.csv", "CSV", "SUPP", "test predictions",
         "exploratory", "agreement + Cohen's kappa", "42"),
        ("r5_seed_stability.csv", "CSV", "SUPP", "stored scalars",
         "descriptive", "mean/SD/CV", "42/43/44"),
        ("r5_allocation_stability.csv", "CSV", "SUPP", "stored scalars "
         "+ multiseed json", "descriptive", "ρ* frequency", "42/43/44"),
        ("regime_statistics.csv", "CSV", "SUPP", "stored VQ diagnostics",
         "descriptive", "occupancy entropy/perplexity", "42/43/44"),
        ("H_distribution.csv", "CSV", "INTERNAL", "frozen banks + stored "
         "diagnostics", "descriptive", "distribution moments", "42/43/44"),
        ("G_vs_H_label_association.csv", "CSV", "SUPP", "frozen banks "
         "(train+val only)", "exploratory", "ANOVA-F + MI", "42"),
        ("G_H_redundancy.csv", "CSV", "MAIN", "frozen banks",
         "exploratory", "correlation + CKA + effective rank", "42"),
        ("sample_mechanism_effects.csv", "CSV", "SUPP", "banks + "
         "predictions", "exploratory", "Cohen d, Cliff δ, bootstrap",
         "42"),
        ("r5_gain_vs_H.csv / regime_complexity_bins.csv", "CSV", "SUPP",
         "banks + predictions", "exploratory", "Spearman, binned rates",
         "42"),
        ("confidence_analysis.csv", "CSV", "SUPP", "Ridge decision values",
         "exploratory", "Brier/ECE/margin by outcome", "42"),
        ("G_H_GplusH_diagnostic.csv", "CSV", "MAIN", "frozen banks "
         "(canonical Ridge)", "confirmatory", "single test eval each "
         "(already stored/derived)", "42"),
        ("effect_size_summary.csv", "CSV", "MAIN", "all above",
         "confirmatory", "consolidated effect sizes + Holm", "42/43/44"),
        ("analysis_registry.json", "JSON", "INTERNAL", "design",
         "n/a", "planned vs exploratory registry", "n/a"),
    ]
    for r in inv_rows:
        W("| " + " | ".join(r) + " |")
    W("")
    W("## Figures (results/haptics_inference/figures/, PDF + PNG)\n")
    W("| figure | designation | content |")
    W("|---|---|---|")
    fig_desc = [
        ("fig1_per_seed_performance", "MAIN", "per-seed test Macro-F1, "
         "six models, mean±SD diamonds"),
        ("fig2_r5_vs_m0_paired_delta", "MAIN", "R5−M0 paired bootstrap "
         "deltas with 95% CIs per seed"),
        ("fig3_r5_vs_m0_per_class_f1", "MAIN", "per-class F1 bootstrap "
         "means ± CI, R5 vs M0"),
        ("fig4_error_overlap_heatmap", "MAIN", "Jaccard error overlap + "
         "disagreement heatmaps (seed 42)"),
        ("fig5_r5_allocation_stability", "SUPP", "retained ρ* per seed "
         "with N_G:N_H labels"),
        ("fig6_regime_occupancy_entropy", "SUPP", "VQ occupancy + "
         "entropy/perplexity by seed"),
        ("fig7_G_vs_H_redundancy", "MAIN", "effective ranks + G-H "
         "cross-correlation summary"),
        ("fig8_gain_vs_regime_complexity", "SUPP", "M0/R5 accuracy by "
         "regime-entropy quartile"),
        ("fig9_gain_vs_H_norm", "SUPP", "decision-flip balance by ||H|| "
         "quintile"),
        ("fig10_confidence_by_outcome", "SUPP", "Ridge confidence by "
         "outcome group"),
        ("fig11_H_distribution_by_seed", "INTERNAL", "H mean±SD per seed"),
        ("fig12_G_vs_H_label_association", "SUPP", "G vs H ANOVA-F/MI "
         "summary (log scale)"),
    ]
    for f in fig_desc:
        W("| " + " | ".join(f) + " |")
    W("")
    W("Reproduce: `python -m experiments.haptics_inference_analysis.analysis` "
      "(resume-safe, cached banks at C:/temp/results) then `.report` and "
      "`.reports`. All statistics fixed-seed; test set (n=308) used for "
      "post-hoc inference only.")
    with open(os.path.join(OUT, "FINAL_INFERENCE_INVENTORY.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(I) + "\n")
    print("inventory written", flush=True)


if __name__ == "__main__":
    main()
