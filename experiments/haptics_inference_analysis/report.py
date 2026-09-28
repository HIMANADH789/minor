"""Figures, effect-size summary, analysis registry, and final reports for
the Haptics inferential analysis. Consumes the CSVs produced by
analysis.py — no model evaluation happens here.

ANALYSIS ONLY (post-hoc inference on stored predictions/features).
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.haptics_inference_analysis.analysis import (
    BASE_MODELS, CACHE, FIG, OUT, ROOT, SEEDS, csv_rows, holm)

C = os.path.join(CACHE, "haptics_inference_arrays.npz")
ARRAYS = np.load(C) if os.path.exists(C) else None


def rows(name):
    p = os.path.join(OUT, name)
    return csv_rows(p) if os.path.exists(p) else []


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200,
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig:", name, flush=True)


def main():
    os.makedirs(FIG, exist_ok=True)
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
    hd_rows = rows("H_distribution.csv")

    # ================= FIGURE 1: per-seed performance ====================
    models = BASE_MODELS + ["MiniROCKET", "R5"]
    colors = {"InceptionTime": "#8c8c8c", "FCN": "#c9c9c9",
              "ResNet1D": "#b3b3b3", "PatchTST": "#a6a6a6",
              "MiniROCKET": "#1f77b4", "R5": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.13
    for i, m in enumerate(models):
        for j, s in enumerate(SEEDS):
            v = [r for r in boot if r["model"] == m and r["seed"] == str(s)]
            if v:
                obs = float(v[0]["observed_macro_f1"])
                ax.bar(i + (j - 1) * w, obs, w, color=colors[m],
                       edgecolor="white", linewidth=.3)
            else:
                ax.bar(i + (j - 1) * w, 0, w, color="none")
    for i, m in enumerate(models):
        vals = [float(r["observed_macro_f1"]) for r in boot
                if r["model"] == m]
        if len(vals) == 3:
            ax.errorbar(i, np.mean(vals), yerr=np.std(vals, ddof=1),
                        fmt="D", color="black", ms=4, capsize=3, zorder=5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=12)
    ax.set_ylabel("test Macro-F1")
    ax.set_title("Haptics per-seed test Macro-F1 "
                 "(diamond = mean$\\pm$SD over seeds)")
    ax.axhline(0.4974, color="#1f77b4", ls=":", lw=1)
    ax.text(4.9, 0.51, "M0 = 0.4974 (deterministic)", fontsize=7,
            color="#1f77b4")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[m])
                       for m in models] +
                      [plt.Line2D([], [], marker="s", color="w",
                                  markerfacecolor="lightgray",
                                  markersize=9, label="seed 42/43/44")],
              fontsize=7, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=.3)
    savefig(fig, "fig1_per_seed_performance")
    TAGS = {"fig1_per_seed_performance": "MAIN"}

    # ================= FIGURE 2: R5-M0 paired bootstrap delta ============
    dr5 = [r for r in deltas if r["comparison"] == "R5_5050-MiniROCKET"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ys = []
    for k, r in enumerate(dr5):
        y = 2 - k
        ys.append(y)
        ax.errorbar(float(r["observed_delta"]), y,
                    xerr=[[float(r["observed_delta"]) - float(r["ci_low"])],
                          [float(r["ci_high"]) - float(r["observed_delta"])]],
                    fmt="o", color="#d62728", capsize=4,
                    label="R5(50/50) - M0 (paired bootstrap 95% CI)")
    ax.axvline(0, color="k", lw=.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"seed {r['seed']}" for r in dr5])
    ax.set_xlabel("Macro-F1 delta (R5 - M0)")
    ax.set_title("R5(50/50) vs M0: paired bootstrap deltas on 308 test "
                 "samples")
    ax.grid(axis="x", alpha=.3)
    if dr5:
        ax.legend(fontsize=7, loc="lower right")
    savefig(fig, "fig2_r5_vs_m0_paired_delta")
    TAGS["fig2_r5_vs_m0_paired_delta"] = "MAIN"

    # ================= FIGURE 3: per-class F1 with CIs ===================
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    classes = sorted({r["class"] for r in pcb}, key=int)
    x = np.arange(len(classes))
    for k, m in enumerate(["R5_5050", "MiniROCKET"]):
        f1m = [float([r for r in pcb if r["model"] == m and
                      r["class"] == c][0]["f1_boot_mean"]) for c in classes]
        lo = [float([r for r in pcb if r["model"] == m and
                     r["class"] == c][0]["ci_low"]) for c in classes]
        hi = [float([r for r in pcb if r["model"] == m and
                     r["class"] == c][0]["ci_high"]) for c in classes]
        ax.errorbar(x + (k - .5) * .18, f1m,
                    yerr=[np.array(f1m) - lo, np.array(hi) - f1m],
                    fmt="o", ms=5, capsize=3,
                    color="#d62728" if k == 0 else "#1f77b4",
                    label={"R5_5050": "R5 (50/50)",
                           "MiniROCKET": "MiniROCKET M0"}[m])
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("class"); ax.set_ylabel("per-class F1 (bootstrap mean, "
                                          "95% CI)")
    ax.set_title("Per-class F1: R5 vs M0 (n=308 bootstrap)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig(fig, "fig3_r5_vs_m0_per_class_f1")
    TAGS["fig3_r5_vs_m0_per_class_f1"] = "MAIN"

    # ================= FIGURE 4: error overlap heatmap ===================
    tag42 = [r for r in eo if r["seed"] == "42"]
    labels = ["MiniROCKET", "R5_5050"] + BASE_MODELS
    M = len(labels)
    jac = np.full((M, M), np.nan)
    dis = np.full((M, M), np.nan)
    for r in tag42:
        ia, ib = labels.index(r["model_A"]), labels.index(r["model_B"])
        jac[ia, ib] = jac[ib, ia] = float(r["error_overlap_jaccard"])
        dis[ia, ib] = dis[ib, ia] = float(r["disagreement_rate"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, Mat, t, fmt in ((axes[0], jac, "Jaccard error overlap",
                             "{:.2f}"),
                            (axes[1], dis, "prediction disagreement rate",
                             "{:.2f}")):
        im = ax.imshow(Mat, cmap="viridis", vmin=0)
        ax.set_xticks(range(M)); ax.set_yticks(range(M))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(M):
            for j in range(M):
                if not np.isnan(Mat[i, j]):
                    ax.text(j, i, fmt.format(Mat[i, j]), ha="center",
                            va="center", fontsize=6,
                            color="w" if Mat[i, j] < .6 else "k")
        ax.set_title(t, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=.8)
    savefig(fig, "fig4_error_overlap_heatmap")
    TAGS["fig4_error_overlap_heatmap"] = "MAIN"

    # ================= FIGURE 5: allocation stability ====================
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    seeds = [str(r["seed"]) for r in alloc]
    rho_star = [float(r["rho_star"]) for r in alloc]
    ax.bar(seeds, rho_star, color=["#d62728" if r["retained_config"] ==
                                   "50/50" else "#ff9896" for r in alloc],
           width=.55)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.text(2.35, 0.505, "rho=0.5 (50/50, R2 point)", fontsize=7)
    for i, r in enumerate(alloc):
        ax.text(i, float(r["rho_star"]) + .01,
                f"{r['N_G']}:{r['N_H']}", ha="center", fontsize=7)
    ax.set_ylim(0, .58); ax.set_ylabel("retained rho*")
    ax.set_xlabel("seed")
    ax.set_title("R5 budget allocation retained per seed "
                 "(CV-selected; labels = N_G:N_H)")
    ax.grid(axis="y", alpha=.3)
    savefig(fig, "fig5_r5_allocation_stability")
    TAGS["fig5_r5_allocation_stability"] = "SUPPLEMENTARY"

    # ================= FIGURE 6: regime occupancy/entropy ================
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for j, s in enumerate(SEEDS):
        rr = [r for r in reg if r["seed"] == str(s)]
        if not rr:
            continue
        occ = np.array(json.loads(rr[0]["occupancy"]))
        axes[0].bar(np.arange(len(occ)) + (j - 1) * .25, occ, .25,
                    label=f"seed {s}")
        axes[1].bar(j, float(rr[0]["entropy_nats"]), .5,
                    color=f"C{j}")
        axes[1].text(j, float(rr[0]["entropy_nats"]) + .02,
                     f"{rr[0]['perplexity']}", ha="center", fontsize=7)
    axes[0].set_xlabel("VQ code k"); axes[0].set_ylabel("occupancy q_k")
    axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=.3)
    axes[1].set_xticks(range(len(SEEDS)))
    axes[1].set_xticklabels([str(s) for s in SEEDS])
    axes[1].set_ylabel("regime occupancy entropy (nats)")
    axes[1].set_title("perplexity annotated")
    axes[1].grid(axis="y", alpha=.3)
    fig.suptitle("Regime occupancy and entropy across seeds "
                 "(stored VQ diagnostics)", fontsize=10)
    savefig(fig, "fig6_regime_occupancy_entropy")
    TAGS["fig6_regime_occupancy_entropy"] = "SUPPLEMENTARY"

    # ================= FIGURE 7: G vs H redundancy =======================
    if red:
        r0 = red[0]
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
        axes[0].bar(["G", "H"], [float(r0["effective_rank_G"]),
                                 float(r0["effective_rank_H"])],
                    color=["#1f77b4", "#d62728"], width=.5)
        for i, k in enumerate(["effective_rank_G", "effective_rank_H"]):
            axes[0].text(i, float(r0[k]) * 1.02, r0[k], ha="center",
                         fontsize=8)
        axes[0].set_ylabel("effective rank (subsampled)")
        axes[0].set_title("Effective rank")
        stats = [("mean_abs_r", "|r| mean"), ("median_abs_r", "|r| median"),
                 ("p95_abs_r", "|r| p95"),
                 ("frac_abs_r_gt_0.5", "frac |r|>0.5"),
                 ("frac_abs_r_gt_0.9", "frac |r|>0.9")]
        axes[1].bar([s[1] for s in stats],
                    [float(r0[s[0]]) for s in stats],
                    color="#7f7f7f", width=.55)
        axes[1].set_xticklabels([s[1] for s in stats], rotation=25,
                                ha="right", fontsize=7)
        axes[1].set_title(f"G-H cross-correlation (linear CKA = "
                          f"{r0['linear_CKA_G_vs_H']})")
        axes[1].grid(axis="y", alpha=.3)
        fig.suptitle("G vs H redundancy (500-feature subsample, train+val)",
                     fontsize=10)
        savefig(fig, "fig7_G_vs_H_redundancy")
        TAGS["fig7_G_vs_H_redundancy"] = "MAIN"

    # ================= FIGURE 8: gain vs regime complexity ===============
    if bins:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        xs = [int(r["bin"]) for r in bins]
        ax.bar([x - .18 for x in xs], [float(r["M0_acc"]) for r in bins],
               .34, label="M0 acc", color="#1f77b4")
        ax.bar([x + .18 for x in xs], [float(r["R5_acc"]) for r in bins],
               .34, label="R5 acc", color="#d62728")
        for r in bins:
            ax.text(int(r["bin"]), .02,
                    f"n={r['n']}\n{r['entropy_range']}", ha="center",
                    fontsize=6, color="k")
        ax.set_xticks(xs)
        ax.set_xlabel("regime-entropy quartile bin (low -> high)")
        ax.set_ylabel("test accuracy")
        dmax = max(float(r["delta_acc"]) for r in bins) if bins else 0
        ax.set_title(f"R5 vs M0 accuracy by regime complexity "
                     f"(delta per bin: "
                     f"{['%+.3f' % float(r['delta_acc']) for r in bins]})")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        savefig(fig, "fig8_gain_vs_regime_complexity")
        TAGS["fig8_gain_vs_regime_complexity"] = "SUPPLEMENTARY"

    # ================= FIGURE 9: gain vs H magnitude =====================
    if ARRAYS is not None:
        delta_c = ARRAYS["delta_c"]
        h_norm = ARRAYS["h_norm"]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        qs = np.quantile(h_norm, np.linspace(0, 1, 6))
        for b in range(5):
            m_ = (h_norm >= qs[b]) & (h_norm <= qs[b + 1] if b == 4
                                     else h_norm < qs[b + 1])
            if m_.sum() == 0:
                continue
            win = (delta_c[m_] == 1).sum()
            lose = (delta_c[m_] == -1).sum()
            ax.bar(b, win - lose, .6,
                   color="#2ca02c" if win > lose else "#d62728", alpha=.8)
            ax.text(b, (win - lose) + (0.5 if win >= lose else -1.0),
                    f"+{win}/-{lose}", ha="center", fontsize=7)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(range(5))
        ax.set_xlabel("test-sample ||H|| quintile (low -> high)")
        ax.set_ylabel("R5-only wins - M0-only wins")
        for g in gain:
            if g["feature"] == "H_norm":
                ax.set_title(f"Decision flips vs ||H|| "
                             f"(Spearman rho={g['spearman_rho']}, "
                             f"p={g['p_value']}, "
                             f"n+={g['n_plus']}, n-={g['n_minus']})")
        ax.grid(axis="y", alpha=.3)
        savefig(fig, "fig9_gain_vs_H_norm")
        TAGS["fig9_gain_vs_H_norm"] = "SUPPLEMENTARY"

    # ================= FIGURE 10: confidence by outcome ==================
    conf_g = [r for r in conf if str(r["model"]).startswith("group:")]
    if conf_g:
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        names = [r["model"].replace("group:", "") for r in conf_g]
        c_r5 = [float(r["mean_conf_correct"]) for r in conf_g]
        c_m0 = [float(r["mean_conf_incorrect"]) for r in conf_g]
        ax.bar(np.arange(len(names)) - .18, c_r5, .34,
               label="R5(50/50) Ridge confidence", color="#d62728")
        ax.bar(np.arange(len(names)) + .18, c_m0, .34,
               label="M0 Ridge confidence", color="#1f77b4")
        for i, r in enumerate(conf_g):
            ax.text(i - .18, c_r5[i] + .005, str(r["n"]), ha="center",
                    fontsize=6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=10, fontsize=8)
        ax.set_ylabel("mean softmax confidence (Ridge decision values)")
        ax.set_title("Confidence by outcome group (n annotated)")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
        savefig(fig, "fig10_confidence_by_outcome")
        TAGS["fig10_confidence_by_outcome"] = "SUPPLEMENTARY"

    # ================= additional requested figures ======================
    # H distribution by seed
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sd_rows = [r for r in hd_rows if "seed" in r["split"]]
    if sd_rows:
        seeds_l = [r["split"].split("seed")[1].split("_")[0]
                   for r in sd_rows]
        ax.bar(seeds_l, [float(r["mean"]) for r in sd_rows], .5,
               yerr=[float(r["sd"]) for r in sd_rows], capsize=4,
               color="#d62728", alpha=.85)
        ax.set_xlabel("seed"); ax.set_ylabel("H mean $\\pm$ SD")
        ax.set_title("H feature distribution by seed (stored diagnostics)")
        ax.grid(axis="y", alpha=.3)
        savefig(fig, "fig11_H_distribution_by_seed")
        TAGS["fig11_H_distribution_by_seed"] = "INTERNAL"
    # G vs H label association
    if assoc:
        a0 = assoc[0]
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        keys = [("mean_top50_F", "mean top-50 F"),
                ("frac_F_gt_1", "frac F>1"),
                ("mean_top50_MI", "mean top-50 MI"),
                ("frac_MI_gt_0.05", "frac MI>0.05")]
        gv = [float([r for r in assoc if r["bank"] == "G"][0][k])
              for k, _ in keys]
        hv = [float([r for r in assoc if r["bank"] == "H"][0][k])
              for k, _ in keys]
        ax.bar(np.arange(4) - .18, gv, .34, label="G bank", color="#1f77b4")
        ax.bar(np.arange(4) + .18, hv, .34, label="H bank", color="#d62728")
        ax.set_yscale("log")
        ax.set_xticks(range(4))
        ax.set_xticklabels([l for _, l in keys], fontsize=8)
        ax.set_title("Label association: G vs H (ANOVA-F / MI, "
                     "train+val only)")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        savefig(fig, "fig12_G_vs_H_label_association")
        TAGS["fig12_G_vs_H_label_association"] = "SUPPLEMENTARY"

    # ================= 20. effect-size summary ===========================
    def interp(lo, hi, side_p):
        if side_p is None or side_p == "":
            return "inconclusive"
        if lo > 0 and float(side_p) >= .95:
            return "strong evidence (CI excludes 0, bootstrap P>=.95)"
        if lo > 0:
            return "moderate evidence (CI excludes 0)"
        if hi < 0:
            return "strong evidence of negative difference"
        if float(side_p) >= .7:
            return "weak evidence"
        return "inconclusive"

    fam = [(r["comparison"], r["seed"]) for r in perms
           if r["seed"] == "42"]
    holm_map = {r["comparison"] + "|" + r["seed"]: r["holm_p"]
                for r in perms}
    eff_rows = []
    for r in deltas:
        key = r["comparison"] + "|" + str(r["seed"])
        pr = next((p for p in perms if p["comparison"] == r["comparison"]
                   and p["seed"] == str(r["seed"])), None)
        eff_rows.append({
            "comparison": r["comparison"], "seed": r["seed"],
            "observed_delta": r["observed_delta"],
            "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "bootstrap_P_positive": r["P_delta_gt_0"],
            "perm_p": pr["perm_p_two_sided"] if pr else "",
            "holm_p": pr["holm_p"] if pr else "",
            "interpretation": interp(float(r["ci_low"]),
                                     float(r["ci_high"]),
                                     float(r["P_delta_gt_0"]))})
    with open(os.path.join(OUT, "effect_size_summary.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eff_rows[0].keys()))
        w.writeheader()
        w.writerows(eff_rows)

    # ================= 21. analysis registry =============================
    registry = {
        "planned_confirmatory_family": [
            {"comparison": "R5_5050-MiniROCKET", "test": "paired bootstrap "
             "delta (10k) + randomization (20k swaps) + McNemar",
             "reason": "primary scientific question: is R5 above M0 beyond "
                       "test-set noise", "exploratory": False,
             "correction": "Holm across the 5 seed-42 planned pairs + "
                           "R5-M0 across seeds"},
        ] + [
            {"comparison": f"R5_5050-{m}", "test": "paired bootstrap + "
             "randomization", "reason": "R5 vs conventional baselines",
             "exploratory": False, "correction": "Holm (same family)"}
            for m in BASE_MODELS],
        "exploratory_family": [
            {"comparison": f"{m}-MiniROCKET", "test": "paired bootstrap",
             "reason": "baseline characterization", "exploratory": True,
             "correction": "none (descriptive)"},
            {"analysis": "mechanism/association sections 11-18",
             "test": "effect sizes (Cohen d, Cliff delta), Spearman, "
                     "binned rates", "reason": "observational association, "
                     "not causal", "exploratory": True, "correction": "none"}],
        "seed_level": {"n_seeds": 3, "rule": "descriptive only; no "
                                              "significance claims"},
        "test_set_use": "post-hoc inference only; no tuning"}

    with open(os.path.join(OUT, "analysis_registry.json"), "w") as f:
        json.dump(registry, f, indent=2)

    # stash for the report writer
    ctx = dict(boot=boot, deltas=deltas, perms=perms, pcb=pcb, eo=eo,
               mcn=mcn, agr=agr, stab=stab, alloc=alloc, reg=reg,
               assoc=assoc, red=red, mech=mech, gain=gain, bins=bins,
               conf=conf, diag=diag, eff_rows=eff_rows, tags=TAGS,
               registry=registry)
    with open(os.path.join(CACHE, "haptics_inference_ctx.json"), "w") as f:
        json.dump({k: v for k, v in ctx.items() if k != "tags"}, f,
                  default=str)
    with open(os.path.join(CACHE, "haptics_inference_tags.json"), "w") as f:
        json.dump(TAGS, f)
    print("figures + summaries complete", flush=True)


if __name__ == "__main__":
    main()
