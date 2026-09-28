"""Publication-quality Demšar critical-difference figures (v2 redesign).

Regenerates, with a clean paper-ready layout:
  1) results/haptics_final_5seed/CRITICAL_DIFFERENCE_HAPTICS.png/.pdf
     (6 models x 5 seeds, classic Demšar layout)
  2) results/cd_diagrams/CRITICAL_DIFFERENCE_4DATASETS.png/.pdf
     (MR vs HERAMBA R5, 4 datasets: rank axis + per-dataset delta panel;
   dataset set per user: Phoneme, EpilepticSeizures, ElectricDevices,
   ItalyPowerDemand — ElectricDevices replaces ECG5000 per request)

Design: serif typography, staggered non-overlapping labels, stems to a
single rank axis, staggered clique bars, colour-blind-safe palette.
Data sources (read-only): results/haptics_final_5seed/PER_SEED_RESULTS.csv
and the per-dataset per-seed result JSONs.
"""
import csv
import itertools
import json
import os

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

plt = None  # set after matplotlib import in main


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.linewidth": 0.9,
        "savefig.dpi": 300,
    })
    return plt


# --------------------------------------------------------------------------
# Figure 1: Haptics, 6 models x 5 seeds
# --------------------------------------------------------------------------
def haptics():
    SRC = os.path.join(ROOT, "results", "haptics_final_5seed",
                       "PER_SEED_RESULTS.csv")
    OUTD = os.path.join(ROOT, "results", "haptics_final_5seed")
    SEEDS = [42, 43, 44, 45, 46]
    MODELS = ["HERAMBA", "MiniRocket", "InceptionTime", "ResNet1D", "FCN",
              "PatchTST"]
    Q, K = 2.850, 6
    rows = list(csv.DictReader(open(SRC)))
    sc = {m: {int(r["seed"]): float(r["test_macro_f1"]) for r in rows
              if r["method"] == m} for m in MODELS}
    M = np.array([[sc[m][s] for s in SEEDS] for m in MODELS])
    n = len(SEEDS)
    ranks = np.array([rankdata(-M[:, j], method="average")
                      for j in range(n)]).T
    mr = ranks.mean(axis=1)
    cd = Q * np.sqrt(K * (K + 1) / (6.0 * n))

    # maximal cliques of the "not significantly different" graph
    adj = np.abs(mr[:, None] - mr[None, :]) < cd
    np.fill_diagonal(adj, False)
    cliques = []
    for size in range(K, 1, -1):
        for combo in itertools.combinations(range(K), size):
            if all(adj[a, b] for a, b in itertools.combinations(combo, 2)):
                if not any(set(combo) <= set(c) for c in cliques):
                    cliques.append(sorted(combo, key=lambda t: mr[t]))

    order = list(np.argsort(mr))          # best (rank1) first
    ncli = len(cliques)

    fig_h = 3.6 + 0.42 * ncli
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    lo, hi = 0.55, K + 0.05
    ax.set_xlim(lo, hi)                    # rank 1 at LEFT of numeric axis
    axis_y = 0.0
    cli_y0 = -(0.55 + 0.52 * (ncli - 1))
    lab_y0 = 1.05

    # axis line + ticks
    ax.axhline(axis_y, color="0.2", lw=1.1, zorder=1)
    for r in range(1, K + 1):
        ax.plot([r, r], [axis_y - 0.07, axis_y + 0.07], color="0.2", lw=1.1)
        ax.text(r, axis_y + 0.16, str(r), ha="center", va="bottom",
                fontsize=10.5)
    ax.text(K + 0.42, axis_y, "rank", ha="left", va="center", fontsize=10.5,
            style="italic", color="0.3")

    # clique bars BELOW the axis (staggered)
    for i, cl in enumerate(cliques):
        xs = [mr[m] for m in cl]
        y = axis_y - 0.55 - 0.52 * i
        ax.plot([min(xs), max(xs)], [y, y], lw=2.6, color="0.15",
                solid_capstyle="round", zorder=3)
        for x in (min(xs), max(xs)):
            ax.plot([x, x], [y, y + 0.13], lw=1.4, color="0.15", zorder=3)
        mid = (min(xs) + max(xs)) / 2
        lab = "–".join(MODELS[m].replace("MiniRocket", "MR")
                       .replace("InceptionTime", "IT")
                       .replace("ResNet1D", "RN")
                       .replace("PatchTST", "PT") for m in cl)
        ax.text(mid, y - 0.30, lab, ha="center", va="top", fontsize=8.8,
                color="0.25")

    # stems + labels: classic two-level stagger above the axis
    # (even rows high level, odd rows low level -> no collisions)
    levels = {}
    for row, m_idx in enumerate(order):
        x = mr[m_idx]
        # level: alternate 0/1, but push to level 1 if the previous label
        # on the same level is within 1.15 rank units
        lvl = 0 if row % 2 == 0 else 1
        if lvl == 0 and levels.get(0) is not None and                 abs(x - levels[0]) < 1.15:
            lvl = 1
        levels[lvl] = x
        y = lab_y0 + 0.78 * lvl
        ax.plot([x, x], [axis_y + 0.05, y - 0.14], lw=0.9, color="0.7",
                zorder=2)
        ax.plot(x, axis_y, "o", ms=7.5, color="white", mec="0.15", mew=1.6,
                zorder=4)
        name = MODELS[m_idx]
        ax.text(x, y, f"{name} ({x:.2f})", ha="center", va="bottom",
                fontsize=11)

    ax.set_xlim(lo, hi)
    ax.set_ylim(cli_y0 - 0.75, lab_y0 + 1.75)
    ax.axis("off")
    ax.set_title(
        "Haptics — critical difference (Demšar), 5 seeds × 6 models\n"
        f"Nemenyi CD$_{{0.05}}$ = {cd:.2f} · mean test Macro-F1 ranks "
        "(1 = best); bars join ranks not separated at α = 0.05\n"
        "(seeds as repeats — descriptive)", fontsize=11.5, pad=10)
    fig.tight_layout()
    png = os.path.join(OUTD, "CRITICAL_DIFFERENCE_HAPTICS.png")
    pdf = os.path.join(OUTD, "CRITICAL_DIFFERENCE_HAPTICS.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf, "mean_ranks": {
        MODELS[i]: round(float(mr[i]), 3) for i in range(K)}, "cd": cd}


# --------------------------------------------------------------------------
# Figure 2: 4 datasets, MR vs HERAMBA R5 (rank axis + delta panel)
# --------------------------------------------------------------------------
def four_datasets():
    OUTD = os.path.join(ROOT, "results", "cd_diagrams")
    DATA = {  # dataset -> (MR per-seed, HERAMBA final per-seed)
        "Phoneme": ([0.0808] * 3, [0.1238, 0.1128, 0.1171]),
        "EpilepticSeizures": ([0.9194] * 3, [0.9461, 0.9404, 0.9444]),
        "ElectricDevices": ([0.6537] * 3, [0.6669, 0.6688, 0.6669]),
        "ItalyPowerDemand": ([0.9650] * 3, [0.9592, 0.9572, 0.9611]),
    }
    ORDER = ["Phoneme", "EpilepticSeizures", "ElectricDevices",
             "ItalyPowerDemand"]
    deltas = {d: float(np.mean(DATA[d][1]) - np.mean(DATA[d][0]))
              for d in ORDER}
    r_her = [1 if deltas[d] > 0 else 2 for d in ORDER]
    r_mr = [3 - r for r in r_her]   # k=2: ranks are {1,2} -> complement
    mean_her = float(np.mean(r_her))
    mean_mr = float(np.mean(r_mr))
    cd = 1.960 * np.sqrt(2 * 3 / (6 * len(ORDER)))

    fig = plt.figure(figsize=(8.6, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.52)

    # ---- panel A: CD rank axis ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0.55, 2.45)
    ax.axhline(0, color="0.2", lw=1.1)
    for r in (1, 2):
        ax.plot([r, r], [-0.06, 0.06], color="0.2", lw=1.1)
        ax.text(r, 0.10, str(r), ha="center", fontsize=10.5)
    ax.text(2.28, 0, "rank", ha="left", va="center", fontsize=10.5,
            style="italic", color="0.3")
    # HERAMBA point at mean_her, label above; MR at mean_mr, label below
    ax.plot(mean_her, 0, "o", ms=10, color="#c0392b", mec="0.15", mew=1.4,
            zorder=4)
    ax.annotate(f"HERAMBA R5  ({mean_her:.2f})",
                xy=(mean_her, 0), xytext=(0.62, 0.55),
                ha="left", va="center", fontsize=11.5, color="#c0392b",
                arrowprops=dict(arrowstyle="-", lw=0.9, color="0.7"))
    ax.plot(mean_mr, 0, "o", ms=10, color="#2c5f8a", mec="0.15", mew=1.4,
            zorder=4)
    ax.annotate(f"MiniRocket  ({mean_mr:.2f})",
                xy=(mean_mr, 0), xytext=(2.38, -0.55),
                ha="right", va="center", fontsize=11.5, color="#2c5f8a",
                arrowprops=dict(arrowstyle="-", lw=0.9, color="0.7"))
    ax.set_ylim(-1.0, 0.95)
    ax.axis("off")
    ax.set_title(
        "A · Critical difference — HERAMBA R5 vs MiniRocket "
        f"(4 datasets, 3 seeds)\nNemenyi CD = {cd:.2f} (k=2) — exceeds "
        "the max possible rank gap; ordering is descriptive",
        fontsize=11, pad=8)

    # ---- panel B: per-dataset delta with per-seed dots ----
    ax2 = fig.add_subplot(gs[1])
    xs = np.arange(len(ORDER))
    for i, d in enumerate(ORDER):
        per = np.array(DATA[d][1]) - np.array(DATA[d][0])
        ax2.scatter(np.full(3, xs[i]) + np.linspace(-0.05, 0.05, 3), per,
                    s=26, color="#c0392b", alpha=0.55, zorder=3,
                    edgecolor="none")
        ax2.hlines(deltas[d], xs[i] - 0.16, xs[i] + 0.16, color="#c0392b",
                   lw=2.6, zorder=4)
        ax2.text(xs[i] - 0.22, deltas[d], f"{deltas[d]:+.4f}",
                 va="center", ha="right", fontsize=9.5, color="#7c2d22")
    ax2.axhline(0, color="0.3", lw=1.0)
    ax2.text(-0.38, 0.0035, "HERAMBA better ↑", fontsize=9,
             color="#7c2d22", ha="left")
    ax2.text(-0.38, -0.0035, "MR better ↓", fontsize=9,
             color="#2c5f8a", ha="left")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([d.replace("\\_", " ") for d in ORDER],
                        fontsize=10)
    ax2.set_ylabel("Δ test Macro-F1  (HERAMBA − MR)", fontsize=10.5)
    ax2.set_title("B · Per-dataset Δ (dots = seeds, line = mean)",
                  fontsize=11, pad=6)
    ax2.grid(alpha=0.3, axis="y")
    lo = min(min(np.array(DATA[d][1]) - np.array(DATA[d][0]))
             for d in ORDER) - 0.008
    hi = max(max(np.array(DATA[d][1]) - np.array(DATA[d][0]))
             for d in ORDER) + 0.008
    ax2.set_ylim(lo, hi)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.suptitle("HERAMBA R5 vs MiniRocket — critical-difference summary "
                 "across 4 datasets", fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(OUTD, "CRITICAL_DIFFERENCE_4DATASETS.png")
    pdf = os.path.join(OUTD, "CRITICAL_DIFFERENCE_4DATASETS.pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf, "mean_ranks": {
        "HERAMBA R5": round(mean_her, 2), "MiniRocket": round(mean_mr, 2)},
        "deltas": {d: round(v, 4) for d, v in deltas.items()}, "cd": cd}


if __name__ == "__main__":
    plt = style()
    r1 = haptics()
    print("HAPTICS:", json.dumps(r1, default=str))
    r2 = four_datasets()
    print("4DATASETS:", json.dumps(r2, default=str))
