"""
Figures for the Cell-C margin diagnostic (Haptics, seed 42).

FIG margin_distribution.png    : box+strip of Ridge margins for MR-correct,
                                 MR-wrong&DRTN-wrong, and Cell C (highlighted).
FIG margin_bucket_analysis.png : Cell-C count by margin quartile/decile vs
                                 uniform expectation + per-bucket accuracies.

Reads results/haptics_margin_diagnostic_seed42/{report.json, per_sample_margin.csv}.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "haptics_margin_diagnostic_seed42")


def fig_margin_distribution():
    df = pd.read_csv(os.path.join(RES, "per_sample_margin.csv"))
    m = df["mr_margin"].values
    mr_c = df["mr_correct"].values.astype(bool)
    dr_c = df["drtn_correct"].values.astype(bool)
    cell_c = df["cell_C"].values.astype(bool)

    groups = [
        ("MiniROCKET correct (n=160)", m[mr_c], "#2563eb", 0.35),
        ("MR wrong & DRTN wrong (n=129)", m[~mr_c & ~dr_c], "#9ca3af", 0.35),
        ("CELL C: MR wrong, DRTN correct (n=19)", m[cell_c], "#dc2626", 0.9),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    pos = np.arange(len(groups))
    for p, (name, v, color, alpha) in zip(pos, groups):
        bp = ax.boxplot(v, positions=[p], widths=0.5, vert=False,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        bp["boxes"][0].set(facecolor=color, alpha=0.25)
        rng = np.random.default_rng(42)
        ax.scatter(v + rng.uniform(-0.035, 0.035, len(v)),
                   np.full(len(v), p) + rng.uniform(-0.14, 0.14, len(v)),
                   s=12, color=color, alpha=alpha, edgecolors="none", zorder=3)
    ax.set_yticks(pos, [g[0] for g in groups], fontsize=9)
    ax.set_xlabel("MiniROCKET Ridge decision margin (top1 − top2)")
    ax.set_title("FIGURE — Ridge-margin distribution: Cell C sits in the low-margin region\n"
                 f"medians: Cell-C {np.median(m[cell_c]):.3f} < all-MR-errors "
                 f"{np.median(m[~mr_c]):.3f} < MR-correct {np.median(m[mr_c]):.3f}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "margin_distribution.png"), dpi=150)
    plt.close(fig)


def fig_bucket_analysis():
    report = json.load(open(os.path.join(RES, "report.json")))
    quart = report["quartile_analysis"]["rows"]
    dec = report["decile_analysis"]["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # --- left: quartiles ---
    ax = axes[0]
    labs = [r["bucket"].split(" ")[0] for r in quart]
    cc = [r["cell_c_count"] for r in quart]
    exp = [r["uniform_expected_cell_c"] for r in quart]
    cb = [r["cell_b_count"] for r in quart]
    xs = np.arange(4)
    ax.bar(xs - 0.18, cc, 0.36, color="#dc2626", label="Cell C (DRTN-only correct)")
    ax.bar(xs + 0.18, cb, 0.36, color="#2563eb", alpha=0.55,
           label="Cell B (MR-only correct)")
    ax.plot(xs, exp, "k_", ms=18, lw=0, label="uniform expectation (4.75)")
    for x, c in zip(xs, cc):
        ax.text(x - 0.18, c + 0.3, str(c), ha="center", fontsize=9, color="#dc2626")
    ax.set_xticks(xs, labs)
    ax.set_ylabel("sample count")
    ax.set_title("Cell C / Cell B by MiniROCKET-margin quartile", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- right: deciles, Cell-C rate + MR/DRTN accuracy ---
    ax = axes[1]
    xs = np.arange(10)
    rate = [r["cell_c_rate_in_bucket"] for r in dec]
    acc = [r["mr_accuracy"] for r in dec]
    dacc = [r["drtn_accuracy"] for r in dec]
    ax.bar(xs, rate, 0.6, color="#dc2626", alpha=0.8,
           label="Cell-C rate in decile")
    ax.axhline(report["quartile_analysis"]["overall_cell_c_rate"], color="black",
               ls=":", lw=1.2, label="overall Cell-C rate (0.0617)")
    ax.plot(xs, acc, "o-", color="#2563eb", ms=4, label="MiniROCKET accuracy")
    ax.plot(xs, dacc, "s--", color="#059669", ms=4, label="DRTN accuracy")
    ax.set_xticks(xs, [r["bucket"] for r in dec], fontsize=8)
    ax.set_xlabel("MiniROCKET-margin decile (D1 = lowest margin)")
    ax.set_ylabel("rate / accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cell-C localization across margin deciles", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("FIGURE — Cell-C concentration: 10/5/4/0 across quartiles "
                 "(enrichment Q1 = 2.11×, Q4 = 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "margin_bucket_analysis.png"), dpi=150)
    plt.close(fig)


def main():
    fig_margin_distribution()
    fig_bucket_analysis()
    for f in sorted(os.listdir(RES)):
        if f.endswith(".png"):
            print(f)


if __name__ == "__main__":
    main()
