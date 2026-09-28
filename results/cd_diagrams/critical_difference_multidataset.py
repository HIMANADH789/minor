"""Combined Demšar-style critical-difference diagram — 4 datasets x 2 models.

Datasets chosen per request:
  2 best   : EpilepticSeizures (mean Δ +0.0242), Phoneme (mean Δ +0.0371)
  moderate : ECG5000_UNBAL (mean Δ −0.0067; one validation-fallback
             cell at seed 44, included in the final values)
  negative : ItalyPowerDemand (mean Δ −0.0058; NOT the worst — FordA and
             GunPoint ties/CWRU negatives are excluded per request)

Each dataset is one "block" (Demšar's rank axis): per dataset, rank the
2 models by the test Macro-F1 actually recorded for that dataset
(3-seed mean; MR = deterministic canonical control where applicable —
HERAMBA = final benchmark values as reported, no-fallback/raw kept
distinct via the stats JSON). k=2 -> Nemenyi q_0.05 = 1.960,
CD = 1.960 * sqrt(2*3/(6*4)) = 0.990 — with k=2 the CD always exceeds
the max rank gap (1.0), so no bar is drawn; the diagram then shows the
rank order with the dataset axis (standard Demšar presentation).

Outputs:
  results/cd_diagrams/CRITICAL_DIFFERENCE_4DATASETS.png/.pdf
  results/cd_diagrams/CD_MULTIDATASET_STATS.json
Also records the Haptics diagram location in the stats JSON.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "cd_diagrams")
os.makedirs(OUT, exist_ok=True)

# dataset -> (MR per-seed or canonical, HERAMBA final per-seed)
DATA = {
    "EpilepticSeizures": {
        "mr": [0.9194] * 3,             # canonical deterministic control
        "her": [0.9461, 0.9404, 0.9444],
    },
    "Phoneme": {
        "mr": [0.0808] * 3,
        "her": [0.1238, 0.1128, 0.1171],
    },
    "ECG5000_UNBAL": {
        "mr": [0.5938] * 3,
        "her": [0.5846, 0.5828, 0.5938],   # final (incl. fallback cell)
    },
    "ItalyPowerDemand": {
        "mr": [0.9650] * 3,
        "her": [0.9592, 0.9572, 0.9611],
    },
}
MODELS = ["HERAMBA R5", "MiniRocket"]
DATASETS = list(DATA.keys())

# mean scores per dataset -> rank per dataset (1 = best)
mean_mr = np.array([np.mean(DATA[d]["mr"]) for d in DATASETS])
mean_her = np.array([np.mean(DATA[d]["her"]) for d in DATASETS])
score_mat = np.vstack([mean_her, mean_mr])            # k=2 x n=4
ranks = np.array([rankdata(-score_mat[:, j], method="average")
                  for j in range(len(DATASETS))]).T   # k x n
mean_ranks = ranks.mean(axis=1)

K, n = 2, len(DATASETS)
Q = 1.960
cd = Q * np.sqrt(K * (K + 1) / (6.0 * n))
deltas = {d: round(float(np.mean(DATA[d]["her"]) - np.mean(DATA[d]["mr"])), 4)
          for d in DATASETS}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.6, 3.4))
lo = float(mean_ranks.min()) - 0.55
hi = float(mean_ranks.max()) + 0.55
ax.set_xlim(hi, lo)                    # rank 1 at the RIGHT
ax.set_ylim(-0.75, K + 0.45)
# dataset axis marks (which dataset produced each rank position)
y_bar0 = K + 0.16
for row, m_idx in enumerate([0, 1]):   # HERAMBA row 0, MR row 1
    x = mean_ranks[m_idx]
    ax.plot(x, row, "o", ms=10, color="0.12", zorder=3)
    ax.text(x + 0.06, row, f"{MODELS[m_idx]}  ({x:.2f})",
            va="center", ha="left", fontsize=12)
# dataset ticks: best-vs-worst rank pairs per dataset
for j, d in enumerate(DATASETS):
    r_her, r_mr = ranks[0, j], ranks[1, j]
    y = -0.45
    ax.plot([r_her, r_mr], [y, y], lw=1.4, color="0.55")
    ax.plot([r_her], [y], "|", ms=8, color="0.55")
    ax.plot([r_mr], [y], "|", ms=8, color="0.55")
    ax.text((r_her + r_mr) / 2, y - 0.22, d, ha="center", va="top",
            fontsize=9.5, color="0.25")
ax.axvline(1.0, color="0.85", lw=0.8, zorder=0)
ax.axvline(2.0, color="0.85", lw=0.8, zorder=0)
ax.set_yticks([])
ax.set_xticks([1, 2])
ax.set_xlabel("Average rank (1 = best, 2 = worst) across 4 datasets",
              fontsize=11)
ax.set_title(
    "MR vs HERAMBA R5 — critical-difference diagram (Demšar)\n"
    f"4 datasets × 3 seeds · Nemenyi CD = {cd:.2f} (k=2, n=4; exceeds the "
    f"k=2 max gap 1.0 → no significance bar) · test Macro-F1",
    fontsize=11)
for sp in ("left", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
png = os.path.join(OUT, "CRITICAL_DIFFERENCE_4DATASETS.png")
pdf = os.path.join(OUT, "CRITICAL_DIFFERENCE_4DATASETS.pdf")
plt.savefig(png, dpi=200)
plt.savefig(pdf)
plt.close()

stats = {
    "method": "Demšar (2006) CD diagram; each dataset = one block ranked "
              "by mean test Macro-F1 over 3 seeds",
    "models": MODELS,
    "datasets": DATASETS,
    "dataset_mean_scores": {
        d: {"MR": round(float(np.mean(DATA[d]["mr"])), 4),
            "HERAMBA_R5": round(float(np.mean(DATA[d]["her"])), 4)}
        for d in DATASETS},
    "per_seed": DATA,
    "ranks_per_dataset": {d: {"HERAMBA R5": float(ranks[0, j]),
                              "MiniRocket": float(ranks[1, j])}
                          for j, d in enumerate(DATASETS)},
    "mean_ranks": {"HERAMBA R5": round(float(mean_ranks[0]), 3),
                   "MiniRocket": round(float(mean_ranks[1]), 3)},
    "per_dataset_delta": deltas,
    "cd_nemenyi": round(float(cd), 4),
    "note": "k=2: CD (0.99) always exceeds the maximum possible rank gap "
            "(1.0), so no pair can be statistically separated — the "
            "diagram is descriptive ordering only",
    "selection_note": "ECG5000_UNBAL HERAMBA = final benchmark values "
                      "(includes one validation-fallback cell at seed 44); "
                      "all other cells are raw HERAMBA = final (no "
                      "fallback fired)",
    "haptics_diagram": {
        "png": "results/haptics_final_5seed/CRITICAL_DIFFERENCE_HAPTICS.png",
        "pdf": "results/haptics_final_5seed/CRITICAL_DIFFERENCE_HAPTICS.pdf",
        "viewer": "results/haptics_final_5seed/CRITICAL_DIFFERENCE_VIEWER.html",
        "stats": "results/haptics_final_5seed/CRITICAL_DIFFERENCE_STATS.json",
    },
    "files": {"png": os.path.relpath(png, ROOT),
              "pdf": os.path.relpath(pdf, ROOT)},
}
with open(os.path.join(OUT, "CD_MULTIDATASET_STATS.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("mean ranks:", stats["mean_ranks"])
print("deltas:", deltas)
print("CD:", stats["cd_nemenyi"])
print("saved:", png)
print("saved:", pdf)
