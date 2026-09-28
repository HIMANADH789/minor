"""Figures for the R2 Haptics 3-seed robustness experiment (spec section 20)."""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.rcmkn_r2_haptics_3seed.config import M0_REF


def make_figures(seed_results, summary, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    seeds = sorted(seed_results.keys())
    tests = [seed_results[s]["results"]["test_macro_f1"] for s in seeds]
    deltas = [seed_results[s]["delta_vs_M0"] for s in seeds]
    m0 = M0_REF

    # ---- FIGURE 1: per-seed test Macro-F1 with M0 reference ----
    fig, ax = plt.subplots(figsize=(6, 4))
    mean, std = summary["test_mean"], summary["test_std"]
    ax.axhline(m0, color="gray", linestyle="--", linewidth=1,
               label=f"Canonical M0 = {m0:.4f}")
    ax.bar([str(s) for s in seeds], tests, color="#3b6fb6", width=0.55,
           label="R2 per-seed test Macro-F1")
    ax.axhline(mean, color="#3b6fb6", linestyle=":", linewidth=1.2,
               label=f"R2 mean = {mean:.4f} +/- {std:.4f}")
    for s, t in zip(seeds, tests):
        ax.text(str(s), t + 0.004, f"{t:.4f}", ha="center", fontsize=9)
    ax.set_ylim(min(m0, min(tests)) - 0.05, max(max(tests), m0) + 0.05)
    ax.set_ylabel("Test Macro-F1")
    ax.set_xlabel("Seed (learned R2 context; MiniROCKET fixed at 42)")
    ax.set_title("R2 Haptics 3-seed robustness: test Macro-F1")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_test_mf1_per_seed.png"), dpi=160)
    plt.close(fig)

    # ---- FIGURE 2: per-seed improvement over M0 ----
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2e8b57" if d > 0 else "#c0392b" for d in deltas]
    ax.bar([str(s) for s in seeds], deltas, color=colors, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(np.mean(deltas), color="#555", linestyle=":",
               label=f"mean delta = {np.mean(deltas):+.4f}")
    for s, d in zip(seeds, deltas):
        ax.text(str(s), d + (0.002 if d >= 0 else -0.006), f"{d:+.4f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Test Macro-F1 delta vs M0")
    ax.set_xlabel("Seed")
    ax.set_title("R2 improvement over canonical MiniROCKET (M0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_delta_vs_m0.png"), dpi=160)
    plt.close(fig)

    # ---- FIGURE 3: VQ occupancy across seeds ----
    fig, ax = plt.subplots(figsize=(7.5, 4))
    K = 8
    width = 0.8 / len(seeds)
    xs = np.arange(K)
    cmap = plt.get_cmap("viridis")
    for j, s in enumerate(seeds):
        occ = seed_results[s]["vq_diagnostics"]["occupancy"]
        ax.bar(xs + j * width - 0.4 + width / 2, occ, width=width,
               color=cmap(j / max(len(seeds) - 1, 1)), label=f"seed {s}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"code {k}" for k in range(K)])
    ax.set_ylabel("Test-set occupancy fraction")
    ax.set_title("Hard-VQ regime occupancy across seeds\n"
                 "(code identities are NOT semantically aligned across seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_vq_occupancy.png"), dpi=160)
    plt.close(fig)
    return fig_dir
