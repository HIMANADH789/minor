"""REPORT.md writer for the R5 UWave H-budget experiment."""
import json
import os

SHORT = {"UWaveGestureLibraryAll": "All", "UWaveGestureLibraryX": "X",
         "UWaveGestureLibraryY": "Y", "UWaveGestureLibraryZ": "Z"}


def write_report(out_dir, all_res, cross):
    a = []
    a.append("# R5 — Validation-Selected H-Budget Allocation on UWave (seed 42)\n")
    a.append("Ablation of the R2 feature-budget assumption on "
             "UWaveGestureLibrary{All,X,Y,Z}. The R2 representation "
             "construction and the canonical Ridge classifier are unchanged; "
             "the ONLY new degree of freedom is a single validation-selected "
             "budget fraction rho.\n")

    a.append("## 1. Motivation\n")
    a.append("On UWave the fixed 50/50 R2 allocation (4998 G + 4998 H) was "
             "flat-to-negative vs MiniROCKET M0 (deltas -0.71 .. +0.12 pp, "
             "two 'approximately unchanged', two 'degraded'). R5 asks "
             "whether that failure is caused by spending too much of the "
             "finite 9,996-feature budget on H.\n")

    a.append("## 2. Failure of the fixed 50/50 allocation\n")
    a.append("With only ~761 development rows, the 4,998 H features carry "
             "weaker per-feature signal than the PPV block; R2's validation "
             "Macro-F1 exceeded M0's on every UWave dataset while test "
             "deltas stayed flat-to-negative, consistent with budget "
             "over-allocation to H rather than harmful H content.\n")

    a.append("## 3. Why R3/R4 are not revisited\n")
    a.append("R3 (learned continuous gate) and R4 (regime gate) add "
             "learned parameters and were previously explored; R5 instead "
             "tests the simplest possible hypothesis — the allocation "
             "fraction — with a single scalar selected on internal CV. No "
             "gating, no end-to-end learning, no new architecture.\n")

    a.append("## 4-6. R5 hypothesis, rho definition, exact budget\n")
    a.append("- Hypothesis: some rho < 0.5 gives the best internal CV "
             "Macro-F1; if rho*=0.5 the original allocation was already "
             "appropriate; if rho*=0, H carries no useful information here.")
    a.append("- rho = fraction of the 9,996-feature budget allocated to H; "
             "candidates {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}, fixed in advance.")
    a.append("- Budget rule: `N_H = int(round(rho*9996))`, "
             "`N_G = 9996 - N_H`; the sum is exactly 9,996 for every "
             "candidate (0/9996, 1000/8996, 1999/7997, 2999/6997, "
             "3998/5998, 4998/4998).\n")

    a.append("## 7-9. H ranking, leakage prevention, nested CV\n")
    a.append("- H ranking: ANOVA F-statistic (`sklearn f_classif`, "
             "deterministic, cheap, auditable), fixed for all datasets and "
             "all rho.")
    a.append("- Leakage prevention: the ranking is recomputed INSIDE every "
             "CV training fold using TRAIN-FOLD labels only; validation-fold "
             "labels never touch the ranking; official test labels never "
             "enter rho selection, ranking, alpha selection, or fitting.")
    a.append("- G selection: first N_G canonical MiniROCKET features (fixed "
             "ordering, never label-ranked) — rho controls only the H "
             "budget.")
    a.append("- CV: 5-fold stratified CV over the development set "
             "(internal train + validation, identical split/indices as the "
             "R2 experiment); per fold, fit `RidgeClassifierCV` on the "
             "train fold (alpha via its internal LOO-CV on the same fold) "
             "and evaluate the held-out fold; report mean +/- std.\n")

    a.append("## 10-12. Ridge, dataset protocol, controls\n")
    a.append("- Ridge: `RidgeClassifierCV(alphas=np.logspace(-4,4,20))` — "
             "the canonical repository protocol; no new alpha tuning.")
    a.append("- Data: identical loaders/split/indices as "
             "`r2_uwave_seed42` (asserted); per-sample z-normalization; "
             "seed 42 everywhere.")
    a.append("- Frozen R2 stages: the context model is LOADED from the "
             "stored R2 checkpoints (not retrained); G_full/H_full banks are "
             "recomputed with the same extractor convention (identity "
             "audited).")
    a.append("- Controls: M0 and R2 test values are the stored R2-experiment "
             "results (no extra test evaluations); the rho=0 CV folds are "
             "the explicit in-run M0-representation control, and in-run "
             "rho=0 / rho=0.5 validation scores are recorded as "
             "control checks.\n")

    a.append("## 13-15. CV results, selected rho, final test\n")
    a.append("### Inner-CV curve (mean +/- std Macro-F1 over 5 folds)\n")
    a.append("| Dataset | " + " | ".join(f"rho={r}" for r in
                                         ("0.0", "0.1", "0.2", "0.3", "0.4",
                                          "0.5")) + " | rho* |")
    a.append("|---|" + "---|" * 7)
    for ds, r in all_res.items():
        cells = []
        for rk in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5"):
            c = r["cv_curve"].get(rk)
            cells.append(f"{c['mean_cv_macro_f1']:.4f} +/- "
                         f"{c['std_cv_macro_f1']:.4f}" if c else "-")
        a.append(f"| {ds} | " + " | ".join(cells) +
                 f" | **{r['selected_rho']}** |")
    a.append("")
    a.append("### Final test (one official evaluation per dataset)\n")
    a.append("| Dataset | M0 Test | R2 Test | rho* | R5 Test | D R5-M0 | "
             "D R5-R2 |")
    a.append("|---|---|---|---|---|---|---|")
    for ds, r in all_res.items():
        a.append(f"| {ds} | {r['m0_test']:.4f} | {r['r2_test']:.4f} | "
                 f"{r['selected_rho']} | {r['r5']['test_macro_f1']:.4f} | "
                 f"{r['deltas']['R5-M0']:+.4f} | {r['deltas']['R5-R2']:+.4f} |")
    a.append("")

    a.append("## 16. Feature-selection stability\n")
    a.append("Pairwise Jaccard overlap of the fold-wise top-N_H H sets "
             "(diagnostic only; does not affect rho selection):\n")
    a.append("| Dataset | rho* | mean Jaccard | min Jaccard | mean F (selected) |")
    a.append("|---|---|---|---|---|")
    for ds, r in all_res.items():
        d = r["selected_H_diag"]
        if d.get("fold_ranking_jaccard_mean") is None:
            continue
        a.append(f"| {ds} | {r['selected_rho']} | "
                 f"{d['fold_ranking_jaccard_mean']} | "
                 f"{d['fold_ranking_jaccard_min']} | "
                 f"{d['mean_F_selected']} |")
    a.append("")

    a.append("## 17. Cross-dataset interpretation\n")
    b = cross["buckets"]
    a.append(f"- rho* < 0.5: {', '.join(b['rho_star_lt_0.5']) or '(none)'}")
    a.append(f"- rho* = 0.5: {', '.join(b['rho_star_eq_0.5']) or '(none)'}")
    a.append(f"- rho* = 0: {', '.join(b['rho_star_eq_0']) or '(none)'}")
    a.append(f"- R5 > R2: {', '.join(b['r5_improves_over_r2']) or '(none)'}")
    a.append(f"- R5 > M0: {', '.join(b['r5_improves_over_m0']) or '(none)'}")
    a.append("")

    a.append("## 18. Limitations\n")
    a.append("- Single seed; 5-fold CV std is small-sample; the F-statistic "
             "ranking is univariate and ignores feature interactions.")
    a.append("- Tie tolerance 0.001 favors smaller rho by predeclared rule.")
    a.append("- M0/R2 controls reuse stored test results rather than "
             "re-evaluating (no extra test passes).\n")

    a.append("## 19. Final conclusion\n")
    n_lt = len(b["rho_star_lt_0.5"])
    n_imp_r2 = len(b["r5_improves_over_r2"])
    n_imp_m0 = len(b["r5_improves_over_m0"])
    if n_imp_r2 > 0 and n_lt > 0:
        concl = ("A smaller H budget beats the fixed 50/50 allocation on "
                 "some UWave datasets: the R2 failure there is partly a "
                 "budget-allocation problem.")
    elif all(r["selected_rho"] == 0.5 for r in all_res.values()):
        concl = ("The validation selected the original 50/50 allocation on "
                 "every dataset: the R2 budget was already appropriate, and "
                 "the UWave shortfall is not an allocation artifact.")
    elif all(r["selected_rho"] == 0.0 for r in all_res.values()) and n_imp_m0 == 0:
        concl = ("The validation selected rho=0 (G-only) everywhere: H "
                 "carries no useful discriminative information on UWave "
                 "under this protocol.")
    else:
        concl = ("Mixed outcome: rho selection and test deltas do not "
                 "support a single clean interpretation; see the per-dataset "
                 "tables.")
    a.append(f"- {concl}")
    a.append("- Per spec: success is judged only on untouched-test "
             "performance; a nonzero rho* alone is not a success claim.\n")

    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(a) + "\n")
    return path
