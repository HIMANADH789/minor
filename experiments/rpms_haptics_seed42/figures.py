"""Nine required RPMS figures (from saved artifacts; test-label-free)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "rpms_haptics_seed42")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

info = json.load(open(os.path.join(RES, "information_diagnostics.json")))
branch = json.load(open(os.path.join(RES, "branch_metrics.json")))
report = json.load(open(os.path.join(RES, "report.json")))

# 1. architecture diagram -------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis("off")
boxes = [
    (0.5, 0.93, "z-normed Haptics series (train 132 / val 23 / test 308, T=1092)"),
    (0.22, 0.72, "Fixed MiniROCKET\nkernel bank (9996)"),
    (0.5, 0.72, "Frozen R2 SSL encoder\n-> HardVQ K=8 -> regimes k(t)"),
    (0.78, 0.72, "Hydra bank (_HydraInternal,\nk=8, g=64) winner counts"),
    (0.22, 0.45, "G: global PPV\n(kernels 0..3331)"),
    (0.5, 0.45, "H: regime heterogeneity\nH_m = sum_k q_k (PPV_m,k - PPV_m)^2\n(features 0..3331)"),
    (0.78, 0.45, "HydraH: sum_r q_r (W_m,r - W_m)^2\n(units 0..3331)"),
    (0.5, 0.2, "RidgeClassifierCV on [G | H | HydraH] (9996), train+val fit"),
]
for x, y, txt in boxes:
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef2f7", ec="#4472c4"))
arrows = [((0.35, 0.87), (0.22, 0.79)), ((0.5, 0.87), (0.5, 0.79)),
          ((0.65, 0.87), (0.78, 0.79)), ((0.22, 0.65), (0.22, 0.52)),
          ((0.42, 0.65), (0.5, 0.52)), ((0.78, 0.65), (0.78, 0.52)),
          ((0.22, 0.38), (0.42, 0.26)), ((0.5, 0.38), (0.5, 0.26)),
          ((0.78, 0.38), (0.58, 0.26))]
for (x0, y0), (x1, y1) in arrows:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="#4472c4"))
ax.set_title("RPMS architecture (equal budget 3332/3332/3332)", fontsize=11)
fig.savefig(os.path.join(FIG, "f1_architecture.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# 2. branch validation Macro-F1 ------------------------------------------
inc = info["track3_incremental"]
names = list(inc.keys())
vals = [inc[n]["val_macro_f1"] for n in names]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
colors = ["#7f7f7f", "#7f7f7f", "#7f7f7f", "#9ecae1", "#9ecae1",
          "#9ecae1", "#1f77b4"]
ax.bar(names, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Validation Macro-F1 (train/val only)")
ax.set_title("Branch decomposition (Track 3) -- never used for selection")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f2_branch_val_f1.png"), dpi=150)
plt.close(fig)

# 3. incremental contribution chart --------------------------------------
full = inc["G+H+HydraH"]["val_macro_f1"]
pairs = {"d(HydraH | G+H)": full - inc["G+H"]["val_macro_f1"],
         "d(H | G+HydraH)": full - inc["G+HydraH"]["val_macro_f1"],
         "d(G | H+HydraH)": full - inc["H+HydraH"]["val_macro_f1"]}
fig, ax = plt.subplots(figsize=(7, 3.8))
ks = list(pairs.keys())
vs = list(pairs.values())
ax.bar(ks, vs, color=["#1f77b4" if v >= 0 else "#d62728" for v in vs])
for i, v in enumerate(vs):
    ax.text(i, v + (2e-4 if v >= 0 else -6e-4), f"{v:+.4f}",
            ha="center", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Incremental validation gain")
ax.set_title("Leave-one-branch-out decrement of the full combination")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f3_incremental_gains.png"), dpi=150)
plt.close(fig)

# 4. branch feature-correlation heatmap ----------------------------------
red = info["track2_redundancy"]
keys = list(red.keys())
metrics = ["matrix_pearson", "profile_pearson_mean", "linear_cka"]
M = np.array([[red[k][m] for m in metrics] for k in keys])
fig, ax = plt.subplots(figsize=(6.5, 3.6))
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(metrics)), metrics, rotation=15)
ax.set_yticks(range(len(keys)), keys)
for i in range(len(keys)):
    for j in range(len(metrics)):
        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                fontsize=9)
fig.colorbar(im, ax=ax, label="similarity")
ax.set_title("Between-branch redundancy (Track 2)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f4_branch_correlation.png"), dpi=150)
plt.close(fig)

# 5. H vs HydraH relationship --------------------------------------------
al = info["track8_alignment"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
hist = np.array(al["corr_hist"], dtype=float)
centers = np.linspace(-0.95, 0.95, len(hist))
ax.bar(centers, hist / hist.sum(), width=0.09,
       color="#1f77b4", alpha=0.85)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("per-feature corr(H_m, HydraH_m)")
ax.set_ylabel("fraction of features")
ax.set_title(f"H vs HydraH alignment: mean={al['mean_corr']:.3f}, "
             f"median={al['median_corr']:.3f}, "
             f"strong={al['fraction_strong_corr']:.3f}")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f5_h_vs_hydrah.png"), dpi=150)
plt.close(fig)

# 6. VQ regime occupancy -------------------------------------------------
vq = branch["track7_vq"]["codes_trva"]
counts = np.array(vq["counts"], dtype=float)
occ = counts / max(counts.sum(), 1)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar(range(len(occ)), occ, color="#2ca02c")
ax.set_xlabel("VQ code")
ax.set_ylabel("occupancy fraction (train+val)")
ax.set_title(f"VQ regime occupancy (entropy="
             f"{vq['normalized_entropy']:.3f}, perplexity="
             f"{vq['perplexity']:.2f}, active={vq['active_codes']}/8)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f6_vq_occupancy.png"), dpi=150)
plt.close(fig)

# 7/8. mean regime contribution per branch -------------------------------
for tag, key, fname in (("H", "track6_H", "f7_mean_regime_contrib_H.png"),
                        ("HydraH", "track6_HydraH",
                         "f8_mean_regime_contrib_HydraH.png")):
    c = np.array(branch[key]["mean_contribution_per_regime"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(range(len(c)), c, color="#ff7f0e" if tag == "HydraH" else "#1f77b4")
    ax.set_xlabel("regime k")
    ax.set_ylabel(f"mean contribution q_k*(dev)^2 ({tag})")
    ax.set_title(f"Per-regime contribution to {tag} "
                 f"(dominant: k={branch[key]['dominant_regime']})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)

# 9. RPMS vs historical references ---------------------------------------
off = report["official_rpms"]["test_macro_f1"]
refs = report["references"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
names = ["M0 (canonical)", "R2 (audited)", "RPMS (official)"]
vals = [refs["M0"], refs["R2"], off]
cols = ["#bbbbbb", "#7f7f7f", "#1f77b4"]
ax.bar(names, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylim(0.4, 0.65)
ax.set_ylabel("Test Macro-F1 (seed 42)")
ax.set_title("RPMS vs canonical M0 and audited R2 (equal 9996 budget)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f9_rpms_vs_refs.png"), dpi=150)
plt.close(fig)

print("saved 9 figures to", FIG)
