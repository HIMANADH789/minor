"""Figures for the seed-42 DRTN-conditioned MiniROCKET transfer screen.

All figures are generated from the frozen result artifacts; no test result
influences axis choices or dataset filtering (they are plotted as-is).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "results",
                    "drtn_conditioned_minirocket_transfer_seed42")

ALL_DS = ["EpilepticSeizures", "Phoneme", "ECG5000_UNBAL", "ECG5000_BAL",
          "CWRU_UNBAL", "CWRU_BAL"]
COLORS = {"M0": "#888888", "M1": "#1f77b4", "M2": "#ff7f0e", "M3": "#2ca02c"}


def load_results():
    out = []
    for ds in ALL_DS:
        with open(os.path.join(BASE, ds, "result.json")) as f:
            out.append(json.load(f))
    return out


def fig_performance(results):
    ds_names = [r["dataset"] for r in results]
    x = np.arange(len(ds_names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, v in enumerate(["M0", "M1", "M2", "M3"]):
        vals = [r["results"][v]["test_macro_f1"] for r in results]
        ax.bar(x + (i - 1.5) * w, vals, w, label=v, color=COLORS[v])
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=20, ha="right")
    ax.set_ylabel("Test Macro-F1 (seed 42)")
    ax.set_title("DRTN-conditioned MiniROCKET transfer screen — all variants")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", "performance_comparison.png"),
                dpi=150)
    plt.close(fig)


def fig_delta(results, key, fname, title):
    ds_names = [r["dataset"] for r in results]
    vals = [r["deltas"][key] for r in results]
    colors = ["#2ca02c" if v > 0.005 else "#d62728" if v < -0.005
              else "#999999" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(ds_names, vals, color=colors)
    ax.axhline(0, color="k", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.001 if v >= 0 else -0.001), f"{v:+.4f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylabel("Δ test Macro-F1 (seed 42)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", fname), dpi=150)
    plt.close(fig)


def fig_heterogeneity(results):
    ds_names = [r["dataset"] for r in results]
    m1 = [r["heterogeneity_diagnostics"]["M1_test"]["mean"] for r in results]
    m3 = [r["heterogeneity_diagnostics"]["M3_test"]["mean"] for r in results]
    x = np.arange(len(ds_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, m1, w, label="M1 (DRTN regimes)", color="#1f77b4")
    ax.bar(x + w / 2, m3, w, label="M3 (shuffled regimes)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=20, ha="right")
    ax.set_ylabel("Mean heterogeneity (test)")
    ax.set_title("Regime-heterogeneity magnitude: learned vs shuffled")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", "heterogeneity_diagnostic.png"),
                dpi=150)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)
    results = load_results()
    fig_performance(results)
    fig_delta(results, "M1_M0", "delta_M1_M0.png",
              "Δ(M1−M0): DRTN-conditioned vs canonical MiniROCKET")
    fig_delta(results, "M1_M3", "delta_M1_M3.png",
              "Δ(M1−M3): learned regimes vs temporally shuffled regimes")
    fig_heterogeneity(results)
    print("figures written to", os.path.join(BASE, "figures"))


if __name__ == "__main__":
    main()
