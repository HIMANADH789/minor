"""Figures for the SSL-context transfer experiment (seed 42)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "rcmkn_ssl_context_transfer_seed42")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

report = json.load(open(os.path.join(OUT, "report.json")))
datasets = report["datasets"]
variants = ["R0", "R2", "C1", "C2"]

# ---------- fig 1: ablation bars per dataset ---------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = {"R0": "#4c72b0", "R2": "#55a868", "C1": "#dd8452", "C2": "#c44e52"}
for ax, ds in zip(axes, datasets):
    vals = [report["per_dataset_test_macro_f1"][ds][v] for v in variants]
    bars = ax.bar(variants, vals, color=[colors[v] for v in variants])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}",
                ha="center", fontsize=8)
    m0_ref = report["canonical_M0_references"][ds]
    ax.axhline(m0_ref, color="k", ls="--", lw=0.8, label=f"M0={m0_ref:.4f}")
    ax.set_title(ds, fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
axes[0].set_ylabel("test Macro-F1")
fig.suptitle("SSL-context transfer: R0/R2/C1/C2 (seed 42)", y=1.02)
fig.savefig(os.path.join(FIG, "fig1_ablation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- fig 2: mean H comparison per dataset ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, ds in zip(axes, datasets):
    mech = report["mechanistic_h"][ds]
    means = [mech[v]["mean_H"] for v in variants]
    bars = ax.bar(variants, means, color=[colors[v] for v in variants])
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
                ha="center", fontsize=8)
    ax.set_title(ds, fontsize=11)
    ax.set_ylabel("mean H (trainva)")
fig.suptitle("Heterogeneity magnitude: learned vs random/shuffled", y=1.02)
fig.savefig(os.path.join(FIG, "fig2_heterogeneity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------- fig 3: R2 delta context-dependent vs R0 ---------------------------
fig, ax = plt.subplots(figsize=(8, 5))
r2_r0 = [report["deltas"][ds]["R2-R0"] for ds in datasets]
ax.bar(datasets, r2_r0, color=["#55a868" if v > 0 else "#c44e52" for v in r2_r0])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("R2 - R0 (test Macro-F1 delta)")
ax.set_title("SSL-context improvement over DRTN reference")
for i, v in enumerate(r2_r0):
    ax.text(i, v + 0.001 * (1 if v >= 0 else -3), f"{v:+.4f}", ha="center", fontsize=9)
fig.savefig(os.path.join(FIG, "fig3_delta_r2_r0.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"saved 3 figures to {FIG}")
