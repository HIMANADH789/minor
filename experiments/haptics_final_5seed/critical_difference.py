"""Demšar-style critical-difference diagram for the Haptics 5-seed benchmark.

Data: results/haptics_final_5seed/PER_SEED_RESULTS.csv (test Macro-F1,
seeds 42-46; MR is the deterministic canonical control shown in every
seed cell per the repo convention).

Method (Demšar 2006, adapted to ONE dataset with seeds as repeats):
  - per seed, rank the k=6 models by test Macro-F1 (1 = best, average
    ranks for ties; no ties occur in these data)
  - mean rank per model
  - Friedman statistic; p-value by EXACT permutation over the 5 seed
    blocks (the chi2 / Iman-Davenport approximations are unreliable at
    n = 5: chi2 > n*(k-1) makes the F correction negative)
  - Nemenyi critical difference CD = q_0.05 * sqrt(k(k+1)/(6n)),
    q_0.05(k=6) = 2.850; models whose mean ranks differ by < CD are
    joined by a bar (maximal cliques of the "not significantly
    different" graph)
  - HONESTY NOTE (stated on the figure): one dataset, 5 seeds -> the
    diagram is a descriptive summary of seed-wise rank stability, NOT
    the multi-dataset significance claim of the original Demšar setup.
    With k=6 and n=5 the Nemenyi CD (3.37) is large; only rank
    differences above it are marked.

Output: results/haptics_final_5seed/CRITICAL_DIFFERENCE_HAPTICS.png/.pdf
        results/haptics_final_5seed/CRITICAL_DIFFERENCE_STATS.json
"""
import csv
import itertools
import json
import os

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results", "haptics_final_5seed",
                   "PER_SEED_RESULTS.csv")
OUT = os.path.join(ROOT, "results", "haptics_final_5seed")

SEEDS = [42, 43, 44, 45, 46]
MODELS = ["HERAMBA", "MiniRocket", "InceptionTime", "ResNet1D", "FCN",
          "PatchTST"]
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}
K = len(MODELS)
Q = NEMENYI_Q05[K]

rows = list(csv.DictReader(open(SRC)))
scores = {m: {int(r["seed"]): float(r["test_macro_f1"]) for r in rows
              if r["method"] == m} for m in MODELS}
assert all(s in scores[m] for m in MODELS for s in SEEDS)
M = np.array([[scores[m][s] for s in SEEDS] for m in MODELS])  # k x n
K, n = M.shape

# seed-wise ranks (1 = best, average ties)
ranks = np.array([rankdata(-M[:, j], method="average")
                  for j in range(n)]).T
mean_ranks = ranks.mean(axis=1)

chi2 = (12 * n / (K * (K + 1))) * ((ranks.sum(axis=1) ** 2).sum()
                                   - K * n * (K + 1) ** 2 / 4)

# exact permutation p for Friedman under H0 (5! = 120 blocks -> exact)
def friedman_stat(rank_cols):
    rk = np.array(rank_cols)
    return (12 * rk.shape[0] / (K * (K + 1))) * (
        (rk.sum(axis=1) ** 2).sum() - K * rk.shape[0] * (K + 1) ** 2 / 4)

block_ranks = ranks.T  # n x k
obs = friedman_stat(block_ranks)
perm_stats = []
for perm in itertools.permutations(range(K)):
    pm = block_ranks[:, list(perm)]
    perm_stats.append(friedman_stat(pm))
perm_stats = np.array(perm_stats)
p_perm = float((perm_stats >= obs - 1e-12).mean())

cd = Q * np.sqrt(K * (K + 1) / (6.0 * n))

# "not significantly different" pairs -> maximal cliques
sig_diff = np.abs(mean_ranks[:, None] - mean_ranks[None, :]) >= cd
np.fill_diagonal(sig_diff, False)
adj = ~sig_diff
cliques = []
for size in range(K, 1, -1):
    for combo in itertools.combinations(range(K), size):
        if all(adj[a, b] for a, b in itertools.combinations(combo, 2)):
            if not any(set(combo) <= set(c) for c in cliques):
                cliques.append(sorted(combo, key=lambda t: mean_ranks[t]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_w = 9.2
fig_h = 0.62 * K + 1.9
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
lo = float(mean_ranks.min()) - 0.9
hi = float(mean_ranks.max()) + 0.9
ax.set_xlim(hi, lo)                     # rank 1 at the RIGHT
ax.set_ylim(-0.55, K + 0.5)

y_bar0 = K + 0.18
for c_idx, clique in enumerate(cliques):
    xs = [mean_ranks[m] for m in clique]
    y = y_bar0 - 0.16 * c_idx
    ax.plot([min(xs), max(xs)], [y, y], lw=2.4, color="0.2",
            solid_capstyle="round")
    ax.plot([min(xs)] * 2, [y, y - 0.07], lw=1.0, color="0.2")
    ax.plot([max(xs)] * 2, [y, y - 0.07], lw=1.0, color="0.2")

order = np.argsort(mean_ranks)
for row, m_idx in enumerate(order):
    x = mean_ranks[m_idx]
    y = row
    ax.plot([x, x], [y, y_bar0 - 0.16 * len(cliques) + 0.02], lw=0.8,
            color="0.78", zorder=1)
    ax.plot(x, y, "o", ms=9, color="0.12", zorder=3)
    name = MODELS[m_idx]
    if x >= (lo + hi) / 2:   # point is right of centre -> label to right
        ax.text(x + 0.07, y, f"{name}  ({x:.2f})", va="center", ha="left",
                fontsize=11)
    else:
        ax.text(x - 0.07, y, f"{name}  ({x:.2f})", va="center", ha="right",
                fontsize=11)

ax.set_yticks([])
ax.set_xticks(list(range(1, K + 1)))
ax.set_xlabel("Average rank (1 = best, 6 = worst)", fontsize=11)
ax.set_title(
    "Haptics — critical-difference diagram (Demšar), 5 seeds × 6 models\n"
    f"Nemenyi CD$_{{0.05}}$ = {cd:.2f} (k={K}, n={n}) · Friedman exact "
    f"p = {p_perm:.3g} · seeds as repeats (descriptive)",
    fontsize=11)
for sp in ("left", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
png = os.path.join(OUT, "CRITICAL_DIFFERENCE_HAPTICS.png")
pdf = os.path.join(OUT, "CRITICAL_DIFFERENCE_HAPTICS.pdf")
plt.savefig(png, dpi=200)
plt.savefig(pdf)
plt.close()

stats = {
    "method": "Demšar (2006) CD diagram, seeds-as-repeats on one dataset; "
              "descriptive rank-stability summary, not a multi-dataset "
              "significance claim",
    "metric": "test Macro-F1",
    "seeds": SEEDS,
    "models": MODELS,
    "mean_test_macro_f1": {MODELS[i]: round(float(M[i].mean()), 4)
                           for i in range(K)},
    "per_seed_scores": {MODELS[i]: {str(s): scores[MODELS[i]][s]
                                    for s in SEEDS} for i in range(K)},
    "seed_wise_ranks": {MODELS[i]: [round(float(ranks[i, j]), 2)
                                    for j in range(n)] for i in range(K)},
    "mean_ranks": {MODELS[i]: round(float(mean_ranks[i]), 3)
                   for i in range(K)},
    "friedman_chi2": round(float(chi2), 4),
    "friedman_p_exact_permutation": round(p_perm, 5),
    "note_on_iman_davenport": "chi2 > n*(k-1): the Iman-Davenport F "
                              "correction is invalid at n=5; exact "
                              "permutation p reported instead",
    "cd_nemenyi_005": round(float(cd), 4),
    "q_nemenyi_k6": Q,
    "cliques_insignificant": [[MODELS[m] for m in c] for c in cliques],
    "files": {"png": os.path.relpath(png, ROOT),
              "pdf": os.path.relpath(pdf, ROOT)},
}
with open(os.path.join(OUT, "CRITICAL_DIFFERENCE_STATS.json"), "w") as f:
    json.dump(stats, f, indent=2)

print("mean ranks:", stats["mean_ranks"])
print("Friedman chi2:", stats["friedman_chi2"],
      "| exact permutation p:", stats["friedman_p_exact_permutation"])
print("CD(Nemenyi):", stats["cd_nemenyi_005"])
print("cliques:", stats["cliques_insignificant"])
print("saved:", png)
print("saved:", pdf)
