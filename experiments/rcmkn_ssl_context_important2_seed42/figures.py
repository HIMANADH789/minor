"""Figures for the SSL-context transfer part 2 (ECG5000_BAL / CWRU_BAL)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results",
                   "rcmkn_ssl_context_important2_seed42")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

report = json.load(open(os.path.join(OUT, "report.json")))
datasets = report["datasets"]
variants = ["R0", "R2", "C1", "C2"]
colors = {"R0": "#4c72b0", "R2": "#55a868", "C1": "#dd8452", "C2": "#c44e52"}

# ---------- fig 1: ablation bars ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, ds in zip(axes, datasets):
    vals = [report["per_dataset_test_macro_f1"][ds][v] for v in variants]
    bars = ax.bar(variants, vals, color=[colors[v] for v in variants])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=8)
    m0 = report["canonical_M0_references"][ds]
    ax.axhline(m0, color="k", ls="--", lw=0.8, label=f"canonical M0={m0:.4f}")
    ax.set_ylim(min(vals) - 0.02, 1.005)
    ax.set_title(ds, fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("test Macro-F1")
fig.suptitle("SSL-context transfer part 2: R0/R2/C1/C2 (seed 42)", y=1.02)
fig.savefig(os.path.join(FIG, "fig1_ablation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- fig 2: mean H ------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, ds in zip(axes, datasets):
    mech = report["mechanistic_h"][ds]
    means = [mech[v]["mean_H"] for v in variants]
    bars = ax.bar(variants, means, color=[colors[v] for v in variants])
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.4f}",
                ha="center", fontsize=8)
    ax.set_title(ds, fontsize=11)
axes[0].set_ylabel("mean H (trainva)")
fig.suptitle("Heterogeneity magnitude: learned (R0 DRTN / R2 SSL) vs controls",
             y=1.02)
fig.savefig(os.path.join(FIG, "fig2_heterogeneity.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ---------- fig 3: R2-R0 delta across all 5 transferred datasets ---------------
fig, ax = plt.subplots(figsize=(9, 4.8))
all_ds = ["Phoneme", "ECG5000_UNBAL", "CWRU_UNBAL", "ECG5000_BAL", "CWRU_BAL"]
p1 = json.load(open(os.path.join(
    os.path.dirname(OUT), "rcmkn_ssl_context_transfer_seed42", "report.json")))
deltas = []
for ds in all_ds:
    if ds in p1["deltas"]:
        deltas.append(p1["deltas"][ds]["R2-R0"])
    else:
        deltas.append(report["deltas"][ds]["R2-R0"])
bars = ax.bar(all_ds, deltas,
              color=["#55a868" if v > 0 else "#c44e52" for v in deltas])
ax.axhline(0, color="k", lw=0.8)
for b, v in zip(bars, deltas):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0012 * (1 if v >= 0 else -2.5),
            f"{v:+.4f}", ha="center", fontsize=9)
ax.set_ylabel("R2 - R0 (test Macro-F1)")
ax.set_title("SSL-context vs DRTN reference across the 5-dataset transfer "
             "sweep (seed 42, descriptive)")
fig.savefig(os.path.join(FIG, "fig3_r2_r0_sweep.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

print(f"saved 3 figures to {FIG}")
