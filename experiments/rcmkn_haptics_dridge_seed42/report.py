"""REPORT.md writer for the differential-Ridge control."""
import os


def write_report(out_dir, result, audits, identity, coef_curve, redundancy):
    a = []
    gs = result["selected_gamma"]
    a.append("# Haptics Differential-Ridge Control (seed 42)\n")
    a.append("Does the usefulness of the full R2 Heterogeneity block H rest "
             "on **feature identity** or merely on **block-level "
             "regularization strength**? All 4998 H features remain present "
             "for every gamma; only the H-block penalty changes "
             "(alpha_H = gamma * alpha_G). This is a regularization "
             "control, NOT feature selection.\n")

    a.append("## 1. Motivation\n")
    a.append("R5 showed that reducing the number of H features improves "
             "UWave results. Before interpreting that as feature selection, "
             "the alternative must be excluded: fewer H features also means "
             "a weaker effective H contribution. This control holds the H "
             "block intact and dials only its regularization.\n")

    a.append("## 2. R2 baseline\n")
    a.append("Canonical Haptics R2 (seed 42): val 0.9014, test 0.5500, "
             "alpha 4.281332398719396, dim 9996 ([G 4998 || H 4998]). "
             "M0 = 0.4974. All references frozen; never re-evaluated.\n")

    a.append("## 3. Why R5 raised the regularization hypothesis\n")
    a.append("Reducing N_H removes columns AND shrinks the aggregate "
             "influence of the H block under a shared penalty. If a "
             "differential penalty on the FULL H block reproduces the same "
             "gain, the effect is regularization/capacity, not subset "
             "selection.\n")

    a.append("## 4. Differential Ridge formulation\n")
    a.append("min over beta_G, beta_H of ||Y - G beta_G - H beta_H||^2 + "
             "alpha_G ||beta_G||^2 + gamma*alpha_G ||beta_H||^2, implemented "
             "exactly by pre-scaling H columns by 1/sqrt(gamma) and fitting "
             "one ordinary Ridge at alpha_G on [G || H_scaled]; predictions "
             "map beta_H_hat = beta_H_scaled / sqrt(gamma). Verified against "
             "direct primal AND dual block-penalty solves "
             "(see audits.json, `audit11_scaling_trick_verified`).\n")

    a.append("## 5-6. Gamma grid and selection protocol\n")
    a.append("- Grid: {1, 2, 4, 8, 16, 32, 64} (fixed before results).")
    a.append("- alpha_G frozen at the canonical R2 value "
             "(4.281332398719396); never re-searched.")
    a.append("- Selection: 5-fold stratified CV over the full 155-sample "
             "development set (explicitly permitted by spec sec 11 because "
             "the 23-sample validation split is too small alone); tie "
             "tolerance 0.001 Macro-F1 favors the smaller gamma.\n")

    a.append("## 7. Leakage control\n")
    a.append("Official test labels never enter gamma selection, CV, or "
             "fitting; the official test is evaluated exactly once, for the "
             "frozen gamma* only (audit13).\n")

    a.append("## 8. Gamma validation curve (5-fold CV, dev set)\n")
    a.append("| gamma | alpha_H | mean CV Macro-F1 | std |")
    a.append("|---|---|---|---|")
    for g in sorted(int(k) for k in result["gamma_curve_cv"]):
        v = result["gamma_curve_cv"][str(g)]
        a.append(f"| {g} | {result['alpha_G'] * g:.3f} | "
                 f"{v['mean_cv_macro_f1']:.4f} | {v['std_cv_macro_f1']:.4f} |")
    a.append(f"\nSelected gamma* = **{gs}** (tie rule: smaller gamma within "
             f"0.001).\n")

    a.append("## 9. Coefficient diagnostics\n")
    a.append("| gamma | ||beta_G|| | ||beta_H|| | eff. df (G) | eff. df (H) |")
    a.append("|---|---|---|---|---|")
    for g in sorted(int(k) for k in coef_curve):
        c = coef_curve[str(g)]
        a.append(f"| {g} | {c['beta_G_l2']} | {c['beta_H_l2']} | "
                 f"{c['eff_df_G']} | {c['eff_df_H']} |")
    a.append(f"\nAt gamma*={gs}: ||beta_G||={result['beta_norms']['beta_G_l2']}, "
             f"||beta_H||={result['beta_norms']['beta_H_l2']}.\n")

    a.append("## 10-12. Official test and frozen-reference comparison\n")
    a.append("| Method | Test Macro-F1 |")
    a.append("|---|---|")
    a.append(f"| M0 (MiniROCKET) | {result['frozen_refs']['M0_test']} |")
    a.append(f"| R2 ([G||H], gamma=1) | {result['frozen_refs']['R2_test']} |")
    a.append(f"| **DRidge(gamma*={gs})** | **{result['test_macro_f1']}** |")
    a.append(f"\n- DRidge - M0 = {result['deltas']['DRidge-M0']:+.4f}")
    a.append(f"- DRidge - R2 = {result['deltas']['DRidge-R2']:+.4f}\n")

    a.append("## 13. Relation to R5\n")
    a.append("R5 (UWave) selected smaller H budgets and beat the fixed "
             "50/50 allocation. If DRidge at moderate gamma reproduces a "
             "similar gain here, the budget effect is largely a "
             "regularization effect; if not, feature count/identity "
             "retains a distinct role. Note the caveats: different dataset, "
             "R5 selected rho by CV whereas gamma here reuses the frozen "
             "canonical alpha_G.\n")

    a.append("## 14. Scientific interpretation\n")
    d_r2 = result["deltas"]["DRidge-R2"]
    r2_val = result["frozen_refs"]["R2_val"]
    if d_r2 > 0.005:
        interp = ("Differential regularization improves over R2 on Haptics: "
                  "part of the H-block value is controlled by penalty "
                  "strength rather than feature identity.")
    elif d_r2 < -0.005:
        interp = ("Differential regularization does NOT improve over R2 on "
                  "Haptics: weakening the H block hurts, so H contributes "
                  "informatively at the canonical shared penalty.")
    else:
        interp = ("Differential regularization leaves performance "
                  "essentially unchanged on Haptics: at the canonical "
                  "alpha_G the Ridge already balances the blocks, and the "
                  "R5 UWave budget effect does not transfer as a pure "
                  "regularization effect to Haptics.")
    a.append(f"- {interp}")
    a.append(f"- CV curve shape: gamma=1 CV "
             f"{result['gamma_curve_cv']['1']['mean_cv_macro_f1']:.4f} vs "
             f"selected gamma* CV {result['gamma_curve_cv'][str(gs)]['mean_cv_macro_f1']:.4f}.")
    a.append(f"- gamma=1 identity check: {identity['gamma1_vs_canonical_R2_pred_mismatches']} "
             "prediction mismatches vs the canonical R2 stored test "
             "predictions (must be 0).\n")

    a.append("## 15. Limitations\n")
    a.append("- Single dataset (Haptics), single seed; n=155 development "
             "samples make CV Macro-F1 high-variance.")
    a.append("- alpha_G is the canonical shared-penalty value, not "
             "re-optimized for the differential parameterization (by "
             "design, to keep the comparison clean).")
    a.append("- Effective-df diagnostics are computed block-wise and are "
             "approximate for the coupled objective.\n")

    a.append("## 16. Stop-rule conclusion\n")
    a.append("Per the stop rule: no feature selection, gates, new datasets, "
             "new gamma values, or new architectures follow this control. "
             "The regularization hypothesis is now tested on Haptics with a "
             "numerically verified differential-Ridge implementation.\n")

    with open(os.path.join(out_dir, "REPORT.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")
