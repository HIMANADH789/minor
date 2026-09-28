"""Figures for the context-dependence screen (seed 42)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(ROOT, "results", "drtn_conditioned_minirocket_context3_seed42")
FIG_DIR = os.path.join(OUT_DIR, "figures")

VARIANTS = ["M0", "M1", "M2", "M3", "A_SOFT"]
COLORS = {"M0": "#888888", "M1": "#1f77b4", "M2": "#ff7f0e", "M3": "#2ca02c",
          "A_SOFT": "#9467bd"}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    report = json.load(open(os.path.join(OUT_DIR, "report.json")))
    results = {r["dataset"]: r for r in report["results"]}
    datasets = [r["dataset"] for r in report["results"]]

    # 1. grouped Macro-F1 comparison
    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.16
    for j, v in enumerate(VARIANTS):
        xs = [i + (j - 2) * width for i in range(len(datasets))]
        vals = [results[d]["results"][v]["test_macro_f1"] for d in datasets]
        ax.bar(xs, vals, width=width, label=v, color=COLORS[v])
        for x, val in zip(xs, vals):
            ax.text(x, val + 0.004, f"{val:.4f}", ha="center", va="bottom",
                    fontsize=7)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test Macro-F1 (seed 42)")
    ax.set_title("DRTN-conditioned MiniROCKET context-dependence screen")
    ax.set_ylim(0.9, 1.03)
    ax.legend(ncol=5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "macro_f1_comparison.png"), dpi=160)
    plt.close(fig)

    # 2. M1 - M0 per dataset
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    vals = [round(results[d]["deltas"]["M1_minus_M0"], 4) for d in datasets]
    bars = ax.bar(datasets, vals,
                  color=["#2ca02c" if v > 0 else "#d62728" for v in vals])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.0006 if v >= 0 else -0.0015), f"{v:+.4f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_ylabel("Test Macro-F1 delta (M1 - M0)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "delta_M1_minus_M0.png"), dpi=160)
    plt.close(fig)

    # 3. M1 - M2 and M1 - M3 per dataset (grouped)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.2
    for j, (a, b) in enumerate([("M1", "M2"), ("M1", "M3"), ("M1", "A_SOFT")]):
        xs = [i + (j - 1) * width for i in range(len(datasets))]
        vals = [round(results[d]["deltas"][f"{a}_minus_{b}"], 4) for d in datasets]
        ax.bar(xs, vals, width=width, label=f"{a} - {b}")
        for x, val in zip(xs, vals):
            ax.text(x, val + (0.0006 if val >= 0 else -0.0016), f"{val:+.4f}",
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test Macro-F1 delta")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "delta_M1_minus_controls.png"), dpi=160)
    plt.close(fig)

    # 4. context-dependence diagnostic: mean H scale by variant
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.2
    for j, v in enumerate(["M1", "M2", "M3", "A_SOFT"]):
        xs = [i + (j - 1.5) * width for i in range(len(datasets))]
        vals = [results[d]["heterogeneity_stats"][v]["mean"] for d in datasets]
        ax.bar(xs, vals, width=width, label=f"mean H ({v})", color=COLORS[v])
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean heterogeneity H (test split)")
    ax.set_title("Context-dependence scale: learned vs destroyed alignment")
    ax.set_yscale("log")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "context_dependence_scale.png"), dpi=160)
    plt.close(fig)

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
