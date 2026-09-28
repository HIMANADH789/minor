"""Figures for the R5 UWave H-budget experiment (spec 32)."""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHORT = {"UWaveGestureLibraryAll": "All", "UWaveGestureLibraryX": "X",
         "UWaveGestureLibraryY": "Y", "UWaveGestureLibraryZ": "Z"}
RHO_KEYS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]


def make_figures(all_res, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- FIGURES 1-4: CV Macro-F1 vs rho per dataset ----
    for i, ds in enumerate(all_res):
        r = all_res[ds]
        rhos = [float(k) for k in r["cv_curve"]]
        means = [r["cv_curve"][k]["mean_cv_macro_f1"] for k in r["cv_curve"]]
        stds = [r["cv_curve"][k]["std_cv_macro_f1"] for k in r["cv_curve"]]
        fig, ax = plt.subplots(figsize=(5.6, 4))
        ax.errorbar(rhos, means, yerr=stds, marker="o", capsize=3,
                    color="#3b6fb6", label="mean CV Macro-F1 +/- std")
        ax.axvline(r["selected_rho"], color="#c0392b", linestyle="--",
                   label=f"selected rho* = {r['selected_rho']}")
        ax.axhline(r["m0_test"], color="#888", linestyle=":",
                   label=f"M0 test = {r['m0_test']:.4f}")
        ax.axhline(r["r2_test"], color="#2e8b57", linestyle=":",
                   label=f"R2 test = {r['r2_test']:.4f}")
        ax.set_xlabel("rho (fraction of the 9996-feature budget given to H)")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"R5 CV curve — {SHORT.get(ds, ds)} (5-fold, train+val)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig{i + 1}_cv_vs_rho_"
                                         f"{SHORT.get(ds, ds)}.png"), dpi=160)
        plt.close(fig)

    # ---- FIGURE 5: final test M0 vs R2 vs R5 ----
    datasets = list(all_res)
    x = np.arange(len(datasets))
    w = 0.26
    m0 = [all_res[d]["m0_test"] for d in datasets]
    r2 = [all_res[d]["r2_test"] for d in datasets]
    r5 = [all_res[d]["r5"]["test_macro_f1"] for d in datasets]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w, m0, width=w, label="M0 MiniROCKET", color="#888888")
    ax.bar(x, r2, width=w, label="R2 (rho=0.5)", color="#c9a227")
    ax.bar(x + w, r5, width=w, label="R5 (selected rho*)", color="#3b6fb6")
    for xi, (a, b, c) in enumerate(zip(m0, r2, r5)):
        for off, v in ((-w, a), (0, b), (w, c)):
            ax.text(xi + off, v + 0.004, f"{v:.4f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(d, d) for d in datasets])
    lo = min(min(m0), min(r2), min(r5)) - 0.04
    ax.set_ylim(max(0, lo), 1.0)
    ax.set_ylabel("Official test Macro-F1 (seed 42)")
    ax.set_title("R5 vs M0 vs R2 on UWave")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5_final_test.png"), dpi=160)
    plt.close(fig)

    # ---- FIGURE 6: selected rho per dataset ----
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sel = [all_res[d]["selected_rho"] for d in datasets]
    ax.bar([SHORT.get(d, d) for d in datasets], sel, color="#3b6fb6",
           width=0.5)
    ax.axhline(0.5, color="#c9a227", linestyle="--", label="R2 budget (0.5)")
    ax.axhline(0.0, color="#888", linewidth=0.8, label="M0 (0.0)")
    for i, s in enumerate(sel):
        ax.text(i, s + 0.01, f"{s:.1f}", ha="center", fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Selected rho*")
    ax.set_title("Validation-selected H budget per dataset")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_selected_rho.png"), dpi=160)
    plt.close(fig)
    return fig_dir
