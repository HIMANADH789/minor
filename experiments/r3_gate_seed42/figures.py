"""Four required R3 figures (from saved artifacts; test labels only via
already-saved aggregate metrics, no example selection by test)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "r3_gate_seed42")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

DS = ["Haptics", "ECG5000_BAL"]
res = {ds: {
    "val": json.load(open(os.path.join(RES, ds, "validation_results.json"))),
    "test": json.load(open(os.path.join(RES, ds, "test_results.json"))),
    "gate": json.load(open(os.path.join(RES, ds, "gate_trajectories.json"))),
} for ds in DS}
refs = {"Haptics": {"M0": 0.4974, "R2": 0.5500},
        "ECG5000_BAL": {"M0": 0.6553, "R2": 0.6089}}

# 1. validation Macro-F1: R3-G vs R3-GH vs R3 ----------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(DS))
w_ = 0.26
for i, v in enumerate(("R3-G", "R3-GH", "R3")):
    vals = [res[ds]["val"][v]["val_macro_f1"] for ds in DS]
    ax.bar(x + (i - 1) * w_, vals, w_,
           label=v, color=["#7f7f7f", "#9ecae1", "#1f77b4"][i])
    for j, val in enumerate(vals):
        ax.text(j + (i - 1) * w_, val + 0.008, f"{val:.3f}",
                ha="center", fontsize=8)
ax.set_xticks(x, DS)
ax.set_ylim(0.4, 0.8)
ax.set_ylabel("Validation Macro-F1 (train/val only)")
ax.set_title("R3 three-way validation comparison")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f1_val_comparison.png"), dpi=150)
plt.close(fig)

# 2. w trajectory across epochs ------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.0))
for ds, col in zip(DS, ("#1f77b4", "#d62728")):
    wt = res[ds]["gate"]["trajectory_w"]
    ax.plot(range(len(wt)), wt, color=col, lw=1.6,
            label=f"{ds} (final w={res[ds]['gate']['w_final']:.3f})")
    bep = json.load(open(os.path.join(
        RES, ds, "validation_results.json")))["R3"]["best_epoch"]
    ax.axvline(bep, color=col, ls="--", lw=0.8, alpha=0.6)
ax.axhline(0.5, color="k", lw=0.7, ls=":")
ax.set_xlabel("epoch")
ax.set_ylabel("learned gate  w = sigmoid(theta)")
ax.set_title("Gate trajectory (dashed = best-validation epoch)")
ax.set_ylim(0, 1.02)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f2_w_trajectories.png"), dpi=150)
plt.close(fig)

# 3. final learned w per dataset ------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.6))
wf = [res[ds]["gate"]["w_final"] for ds in DS]
ax.bar(DS, wf, color=["#1f77b4", "#d62728"], width=0.5)
for i, v in enumerate(wf):
    ax.text(i, v + 0.015, f"{v:.4f}", ha="center", fontsize=9)
ax.axhline(0.5, color="k", lw=0.7, ls=":")
ax.set_ylim(0, 1.05)
ax.set_ylabel("final learned w")
ax.set_title("Final dataset-level gate value")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f3_final_w.png"), dpi=150)
plt.close(fig)

# 4. test Macro-F1 vs M0 and R2 -------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.4))
models = ["M0", "R2", "R3-G", "R3-GH", "R3"]
cols = ["#bbbbbb", "#7f7f7f", "#c6dbef", "#9ecae1", "#1f77b4"]
w_ = 0.16
for i, m in enumerate(models):
    vals = []
    for ds in DS:
        vals.append(refs[ds][m] if m in ("M0", "R2")
                    else res[ds]["test"][m]["test_macro_f1"])
    ax.bar(x + (i - 2) * w_, vals, w_, label=m, color=cols[i])
    for j, val in enumerate(vals):
        ax.text(j + (i - 2) * w_, val + 0.008, f"{val:.3f}",
                ha="center", fontsize=7.5)
ax.set_xticks(x, DS)
ax.set_ylim(0.2, 0.75)
ax.set_ylabel("Test Macro-F1 (seed 42)")
ax.set_title("Official test results vs frozen references")
ax.legend(ncol=5, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f4_test_vs_refs.png"), dpi=150)
plt.close(fig)

print("saved 4 figures to", FIG)
