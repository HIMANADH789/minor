"""Haptics 5-seed test Macro-F1 across all methods — per-seed graph.

Data: results/haptics_final_5seed/AGGREGATED_RESULTS.csv (5 valid seeds
42-46 per method; MR deterministic canonical control).

Panels:
  A) per-seed lines (seed on x, test Macro-F1 on y) — shows seed
     stability and the seed-43 InceptionTime spike honestly
  B) grouped bars per method (5 seed bars + mean marker) — same data

Outputs:
  results/haptics_final_5seed/PER_SEED_MACROF1_5SEED.png/.pdf
"""
import csv
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results", "haptics_final_5seed",
                   "AGGREGATED_RESULTS.csv")
OUT = os.path.join(ROOT, "results", "haptics_final_5seed")

SEEDS = [42, 43, 44, 45, 46]
ORDER = ["HERAMBA", "MiniRocket", "InceptionTime", "ResNet1D", "FCN",
         "PatchTST"]
COLORS = {"HERAMBA": "#d62728", "MiniRocket": "#1f77b4",
          "InceptionTime": "#2ca02c", "ResNet1D": "#9467bd",
          "FCN": "#ff7f0e", "PatchTST": "#8c564b"}

rows = {r["method"]: r for r in csv.DictReader(open(SRC))}
assert set(ORDER) <= set(rows), "missing methods in AGGREGATED_RESULTS"
M = np.array([[float(rows[m][f"test_f1_seed{s}"]) for s in SEEDS]
              for m in ORDER])
assert not np.isnan(M).any()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

# ---- Panel A: per-seed lines ----
for i, m in enumerate(ORDER):
    axA.plot(SEEDS, M[i], "-o", color=COLORS[m], label=m, lw=1.8, ms=6,
             alpha=0.95)
axA.set_xticks(SEEDS)
axA.set_xlabel("Seed", fontsize=11)
axA.set_ylabel("Test Macro-F1", fontsize=11)
axA.set_title("A · Per-seed test Macro-F1 (5 seeds)", fontsize=12)
axA.grid(alpha=0.3)
axA.legend(fontsize=9, ncol=2, loc="lower left", framealpha=0.9)

# ---- Panel B: grouped bars (5 seeds each) + mean marker ----
w = 0.15
xc = np.arange(len(ORDER))
for i, m in enumerate(ORDER):
    axB.bar(xc[i] + (np.arange(5) - 2) * w, M[i], width=w * 0.92,
            color=COLORS[m], alpha=0.55, edgecolor=COLORS[m], lw=0.8)
    mu = M[i].mean()
    axB.hlines(mu, xc[i] - 2.6 * w, xc[i] + 2.6 * w, color=COLORS[m],
               lw=2.4)
    axB.text(xc[i], mu + 0.008, f"{mu:.3f}", ha="center", fontsize=8.5,
             color=COLORS[m], fontweight="bold")
axB.set_xticks(xc)
axB.set_xticklabels(ORDER, rotation=20, ha="right", fontsize=9.5)
axB.set_ylabel("Test Macro-F1", fontsize=11)
axB.set_title("B · Seed bars (light) + mean (line), 5 seeds", fontsize=12)
axB.grid(alpha=0.3, axis="y")
axB.set_ylim(0, 0.62)

fig.suptitle(
    "Haptics — test Macro-F1 across methods, 5 seeds (42–46) · "
    "MiniRocket = deterministic canonical control (identical across seeds)",
    fontsize=12.5, y=1.02)
plt.tight_layout()
png = os.path.join(OUT, "PER_SEED_MACROF1_5SEED.png")
pdf = os.path.join(OUT, "PER_SEED_MACROF1_5SEED.pdf")
plt.savefig(png, dpi=200, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
plt.close()
print("saved:", png)
print("saved:", pdf)
for i, m in enumerate(ORDER):
    print(f"{m:14s} seeds={M[i].tolist()} mean={M[i].mean():.4f} "
          f"std={M[i].std(ddof=1):.4f}")
