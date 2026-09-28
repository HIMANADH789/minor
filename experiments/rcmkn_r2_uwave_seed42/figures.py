"""Figures for the R2 UWave Motion generalization experiment (spec 34)."""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHORT = {"UWaveGestureLibraryAll": "All (T=945)",
         "UWaveGestureLibraryX": "X (T=315)",
         "UWaveGestureLibraryY": "Y (T=315)",
         "UWaveGestureLibraryZ": "Z (T=315)"}


def make_figures(all_res, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    datasets = list(all_res.keys())
    short = [SHORT.get(d, d) for d in datasets]
    m0 = [all_res[d]["results"]["M0"]["test_macro_f1"] for d in datasets]
    r2 = [all_res[d]["results"]["R2"]["test_macro_f1"] for d in datasets]
    deltas = [r - m for r, m in zip(r2, m0)]

    # ---- FIGURE 1: M0 vs R2 test Macro-F1 ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(datasets))
    w = 0.36
    ax.bar(x - w / 2, m0, width=w, label="M0 MiniROCKET", color="#888888")
    ax.bar(x + w / 2, r2, width=w, label="R2 [G || H]", color="#3b6fb6")
    for xi, (a, b) in enumerate(zip(m0, r2)):
        ax.text(xi - w / 2, a + 0.005, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.005, f"{b:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(short)
    lo = min(min(m0), min(r2)) - 0.04
    ax.set_ylim(max(0.0, lo), 1.0)
    ax.set_ylabel("Test Macro-F1 (seed 42)")
    ax.set_title("UWave Motion generalization: MiniROCKET vs R2")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_m0_vs_r2.png"), dpi=160)
    plt.close(fig)

    # ---- FIGURE 2: R2 - M0 improvement ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#2e8b57" if d > 0.005 else
              ("#c0392b" if d < -0.005 else "#999999") for d in deltas]
    ax.bar(short, deltas, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.0015 if d >= 0 else -0.004), f"{d:+.4f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Test Macro-F1 delta (R2 - M0)")
    ax.set_title("R2 improvement over MiniROCKET per dataset")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_delta.png"), dpi=160)
    plt.close(fig)

    # ---- FIGURE 3: R2 VQ regime occupancy for All ----
    ds_all = "UWaveGestureLibraryAll"
    if ds_all in all_res:
        occ = all_res[ds_all]["results"]["R2"]["regime_diagnostics"]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(8)
        w = 0.38
        ax.bar(x - w / 2, occ["train"]["usage"], width=w, label="train+val",
               color="#3b6fb6")
        ax.bar(x + w / 2, occ["test"]["usage"], width=w, label="official test",
               color="#c9a227")
        ax.set_xticks(x)
        ax.set_xticklabels([f"code {k}" for k in range(8)])
        ax.set_ylabel("Occupancy fraction")
        ax.set_title(f"R2 hard-VQ regime occupancy — {SHORT[ds_all]}\n"
                     "(code identities are learned; not semantic classes)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig3_vq_occupancy_all.png"),
                    dpi=160)
        plt.close(fig)

    # ---- FIGURE 4: H-feature distribution across datasets ----
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.1 * len(datasets), 3.4),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, d in zip(axes, datasets):
        h = all_res[d]["mechanistic_h"]["R2"]
        keys = ["mean_H", "std_H", "max_H"]
        vals = [h[k] for k in keys]
        ax.bar(["mean", "std", "max"], vals, color="#3b6fb6", width=0.5)
        ax.set_title(SHORT[d], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("H statistic (R2 regimes, train+val)")
    fig.suptitle("Regime-conditioned heterogeneity magnitude per dataset",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_H_distribution.png"), dpi=160)
    plt.close(fig)
    return fig_dir
