"""Figures for the corrected negative-dataset re-test (seed 42)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))),
    "results", "drtn_conditioned_minirocket_corrected_negative_retest")
FIG_DIR = os.path.join(OUT_DIR, "figures")

VARIANTS = ["M0", "M1", "M2", "M3"]
COLORS = {"M0": "#888888", "M1": "#1f77b4", "M2": "#ff7f0e", "M3": "#2ca02c"}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    report = json.load(open(os.path.join(OUT_DIR, "report.json")))
    results = report["results"]
    datasets = [r["dataset"] for r in results]
    scores = {r["dataset"]: {v: r["results"][v]["test_macro_f1"] for v in VARIANTS}
              for r in results}

    # 1. grouped Macro-F1 comparison
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.2
    for j, v in enumerate(VARIANTS):
        xs = [i + (j - 1.5) * width for i in range(len(datasets))]
        ax.bar(xs, [scores[d][v] for d in datasets], width=width, label=v,
               color=COLORS[v])
        for x, d in zip(xs, datasets):
            ax.text(x, scores[d][v] + 0.004, f"{scores[d][v]:.4f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test Macro-F1 (seed 42)")
    ax.set_title("Corrected DRTN-conditioned MiniROCKET: previously-NEGATIVE datasets")
    lo = min(min(scores[d].values()) for d in datasets)
    ax.set_ylim(max(0.0, lo - 0.05), 1.01)
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "macro_f1_comparison.png"), dpi=160)
    plt.close(fig)

    # 2/3. per-dataset deltas vs the two controls
    for delta_key, fname in [(("M1", "M0"), "delta_M1_minus_M0.png"),
                             (("M1", "M3"), "delta_M1_minus_M3.png")]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        vals = [round(scores[d][delta_key[0]] - scores[d][delta_key[1]], 4)
                for d in datasets]
        bars = ax.bar(datasets, vals,
                      color=["#2ca02c" if v > 0 else "#d62728" for v in vals])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.0015 if v >= 0 else -0.0035), f"{v:+.4f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_ylabel(f"Test Macro-F1 delta ({delta_key[0]} - {delta_key[1]})")
        ax.set_title("seed 42, corrected implementation")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=160)
        plt.close(fig)

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
