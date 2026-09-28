"""
3-seed replication figures (results/drtn_haptics_3seed_replication/figures/):
  fig1_by_seed.png   : test Macro-F1 by model and seed (no significance marks)
  fig2_mean_sd.png   : mean +/- SD per model
  fig3_paired.png    : paired seed-wise DTC-CTC and R5K16-DTC differences
  fig4_codebook.png  : codebook usage histograms per seed (DTC / R5-K16)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "results", "drtn_haptics_3seed_replication")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

MODELS = ["ctc", "dtc", "r5_k16"]
LABELS = {"ctc": "CTC (continuous)", "dtc": "DTC (VQ K=8)",
          "r5_k16": "R5-K16 (VQ+div)"}
COLORS = {"ctc": "#6baed6", "dtc": "#fd8d3c", "r5_k16": "#74c476"}
SEEDS = [42, 43, 44]
MR = 0.4974


def main():
    rep = json.load(open(os.path.join(BASE, "report.json")))
    vals = {m: [rep["aggregate_table"][m]["values"][str(s)] for s in SEEDS]
            for m in MODELS}
    x = np.arange(len(SEEDS))
    w = 0.26

    # FIG 1: by seed
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * w, vals[m], w, label=LABELS[m],
               color=COLORS[m])
        for xi, v in zip(x + (i - 1) * w, vals[m]):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(MR, color="#de2d26", ls="--", lw=1.2,
               label="MiniROCKET 0.4974 (external reference)")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("test Macro-F1"); ax.set_ylim(0, 0.55)
    ax.set_title("3-seed replication, frozen 100/15 budget — "
                 "no significance markers (n=3, exploratory)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_by_seed.png"), dpi=160)
    plt.close(fig)

    # FIG 2: mean +/- SD
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, m in enumerate(MODELS):
        a = rep["aggregate_table"][m]
        ax.errorbar(i, a["mean"], yerr=a["sd"], fmt="o", capsize=5,
                    color=COLORS[m], ms=8)
        ax.text(i + 0.06, a["mean"], f"{a['mean']:.4f} ± {a['sd']:.4f}",
                fontsize=9, va="center")
    ax.axhline(MR, color="#de2d26", ls="--", lw=1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel("test Macro-F1 (mean ± SD over seeds)")
    ax.set_ylim(0.25, 0.55)
    ax.set_title("Aggregate across seeds 42/43/44")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_mean_sd.png"), dpi=160)
    plt.close(fig)

    # FIG 3: paired differences
    p1 = rep["paired_differences"]["dtc_minus_ctc"]["per_seed"]
    p2 = rep["paired_differences"]["r5k16_minus_dtc"]["per_seed"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(0, color="gray", lw=1)
    for i, s in enumerate(SEEDS):
        ax.plot(i - 0.12, p1[str(s)], "o", color="#fd8d3c", ms=9,
                label="DTC − CTC" if i == 0 else None)
        ax.plot(i + 0.12, p2[str(s)], "s", color="#74c476", ms=9,
                label="R5-K16 − DTC" if i == 0 else None)
        ax.text(i - 0.12, p1[str(s)] + 0.004, f"{p1[str(s)]:+.3f}",
                ha="center", fontsize=8)
        ax.text(i + 0.12, p2[str(s)] + 0.004, f"{p2[str(s)]:+.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("paired Δ test Macro-F1")
    ax.set_title("Paired seed-wise effects (primary: DTC − CTC)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_paired.png"), dpi=160)
    plt.close(fig)

    # FIG 4: codebook usage per seed
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2))
    for row, m in enumerate(("dtc", "r5_k16")):
        for col, s in enumerate(SEEDS):
            r = json.load(open(os.path.join(
                BASE, f"seed{s}", m, "result.json")))
            u = r.get("codebook_final_usage_val")
            ax = axes[row, col]
            if u:
                ax.bar(range(len(u)), u, color=COLORS[m])
                ax.set_ylim(0, 0.55)
            fdv = r["final_diag_val"]
            ax.set_title(f"{m} seed{s}  act={fdv['active_codes']} "
                         f"Hn={fdv['normalized_entropy']:.2f}", fontsize=8)
            if col == 0:
                ax.set_ylabel("usage fraction")
    fig.suptitle("Codebook usage histograms (validation), per seed")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_codebook.png"), dpi=160)
    plt.close(fig)

    print("figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
