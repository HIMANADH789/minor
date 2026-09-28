"""Figures for the Haptics differential-Ridge control."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def make_figures(result, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    gammas = sorted(int(g) for g in result["gamma_curve_cv"])
    cv = [result["gamma_curve_cv"][str(g)]["mean_cv_macro_f1"]
          for g in gammas]
    std = [result["gamma_curve_cv"][str(g)]["std_cv_macro_f1"]
           for g in gammas]
    gs = result["selected_gamma"]

    # FIG 1: validation (CV) Macro-F1 vs gamma
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(gammas, cv, yerr=std, fmt="o-", color="tab:blue",
                capsize=4, label="5-fold CV Macro-F1 (dev set)")
    ax.axhline(result["frozen_refs"]["R2_val"], color="tab:red", ls="--",
               label=f"canonical R2 val ({result['frozen_refs']['R2_val']})")
    ax.scatter([gs], [result["val_macro_f1"]], marker="*", s=180,
               color="tab:green", zorder=5,
               label=f"selected gamma*={gs} (final-fit val)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("gamma = alpha_H / alpha_G")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Validation Macro-F1 vs H-block regularization")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_val_vs_gamma.png"), dpi=150)
    plt.close(fig)

    # FIG 2: test result M0 / R2 / DRidge(gamma*)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    names = ["M0\nMiniROCKET", "R2\n[G||H] 50/50",
             f"DRidge(gamma*={gs})"]
    vals = [result["frozen_refs"]["M0_test"], result["frozen_refs"]["R2_test"],
            result["test_macro_f1"]]
    bars = ax.bar(names, vals,
                  color=["tab:gray", "tab:blue", "tab:green"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.4, 0.65)
    ax.set_ylabel("official test Macro-F1")
    ax.set_title("Haptics test: M0 vs R2 vs Differential Ridge")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_test_comparison.png"), dpi=150)
    plt.close(fig)

    # FIG 3: ||beta_G||, ||beta_H|| vs gamma
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cc = result.get("coef_curve", {})
    if cc:
        keys = sorted(int(g) for g in cc)
        bG = [cc[str(g)]["beta_G_l2"] for g in keys]
        bH = [cc[str(g)]["beta_H_l2"] for g in keys]
        ax.plot(keys, bG, "o-", label="||beta_G||")
        ax.plot(keys, bH, "s-", label="||beta_H||")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("gamma")
        ax.set_ylabel("coefficient L2 norm (train+val fit)")
        ax.set_title("Block coefficient magnitude vs H regularization")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_beta_norms.png"), dpi=150)
    plt.close(fig)

    # FIG 4: alpha_H / alpha_G across gamma
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.plot(gammas, gammas, "o-", color="tab:purple")
    ax.scatter([gs], [gs], marker="*", s=180, color="tab:red", zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("gamma")
    ax.set_ylabel("alpha_H / alpha_G")
    ax.set_title("Penalty ratio across the gamma grid")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_alpha_ratio.png"), dpi=150)
    plt.close(fig)
