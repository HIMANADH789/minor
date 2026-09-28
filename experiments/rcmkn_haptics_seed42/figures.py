"""Figures + mechanistic heterogeneity statistics for the RCMKN ablation.

Figures (no test labels are used anywhere):
  1. fig1_architecture.png        architecture diagram
  2. fig2_ablation.png            test Macro-F1 per variant (+ M0 reference)
  3. fig3_code_usage.png          code-usage histograms: DRTN vs SSL-context
  4. fig4_regimes_timeline.png    representative test sequence with regime bands
  5. fig5_regime_conditioning.png per-regime PPV vs global PPV for sample kernels
  6. fig6_hydra_summary.png       Hydra feature block summary

Also writes diagnostics/mechanistic_h_stats.json (mean/median/max/fraction
nonzero of the trainva H block for every variant, recomputed identically to
the runner).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.rcmkn_haptics_seed42.config import (
    SEED, N_GLOBAL, K_CODES, HYDRA, ENCODER, VQ, JOINT)
from experiments.rcmkn_haptics_seed42 import kernel_features as kf
from experiments.rcmkn_haptics_seed42.runner import (
    load_dataset, load_official_drtn, extract_drtn_regimes,
    extract_context_regimes, set_seed, OUT_DIR)
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    create_random_regime_control, create_shuffled_regime_control)

FIG_DIR = os.path.join(OUT_DIR, "figures")
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

set_seed(SEED)
data = load_dataset("Haptics")
Xtr, Xte = data["Xtr"], data["Xte"]
ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
report = json.load(open(os.path.join(OUT_DIR, "report.json")))


def znorm(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


Xtr_z, Xte_z = znorm(Xtr), znorm(Xte)
Xtrva_z = np.vstack([Xtr_z, znorm(data["Xva"])])

# ---------------- regime sources (all deterministic, frozen) -----------------
drtn, _ = load_official_drtn(torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reg0_trva = extract_drtn_regimes(drtn, Xtrva_z, device=dev)
reg0_te = extract_drtn_regimes(drtn, Xte_z, device=dev)

ck = torch.load(os.path.join(OUT_DIR, "context_model_seed42.pt"),
                map_location="cpu", weights_only=False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RCMKNContextModel(n_classes=data["n_classes"]).to(dev)
model.load_state_dict(ck["model_state"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
reg2_trva = extract_context_regimes(model, Xtrva_z, dev)
reg2_te = np.load(os.path.join(OUT_DIR, "_regimes_R2_te.npy"))
reg2_te_full = extract_context_regimes(model, Xte_z, dev)
assert np.array_equal(reg2_te, reg2_te_full[:20])

c1_trva = create_random_regime_control(reg2_trva, seed=SEED)
c2_trva = create_shuffled_regime_control(reg2_trva, seed=SEED)

# ---------------- shared blocks ----------------------------------------------
from aeon.transformations.collection.convolution_based import MiniRocket

extractor = MiniRocket(random_state=SEED, n_jobs=-1)
extractor.fit(Xtr_z[:, None, :].astype(np.float32))
_, valid = kf.compute_raw_activations(extractor, Xtrva_z[:4])
valid_het = valid[N_GLOBAL:]

SOURCES = {
    "R0": (reg0_trva, None),
    "R2": (reg2_trva, None),
    "C1": (c1_trva, None),
    "C2": (c2_trva, None),
    "R1": (reg0_trva, None),   # R1/R3 het = R0/R2 sources respectively
    "R3": (reg2_trva, None),
}
h_stats = {}
H_blocks = {}
for name, (r_trva, _) in SOURCES.items():
    H = kf.compute_heterogeneity_chunked(extractor, Xtrva_z, r_trva, valid_het)
    H_blocks[name] = H
    h_stats[name] = {
        "mean_H": float(H.mean()), "median_H": float(np.median(H)),
        "max_H": float(H.max()),
        "fraction_nonzero": float((H > 0).mean()),
    }
    print(f"  [H] {name}: {h_stats[name]}")
json.dump(h_stats, open(os.path.join(DIAG_DIR, "mechanistic_h_stats.json"), "w"),
          indent=2)

# ---------------- fig 1: architecture ----------------------------------------
fig, ax = plt.subplots(figsize=(11, 7))
ax.axis("off")
boxes = {
    "raw":        (0.03, 0.42, 0.13, 0.16, "RAW SERIES\n(per-sample z-norm)"),
    "mr":         (0.24, 0.70, 0.20, 0.16, "FIXED MINIROCKET BANK\nraw responses a_{m,t} (frozen)"),
    "enc":        (0.24, 0.06, 0.20, 0.22, "SSL CAUSAL ENCODER\n4 blocks, d=32\nmasked-span recon"),
    "vq":         (0.52, 0.06, 0.17, 0.22, "HARD VQ  K=8\nEMA + revival\nk_t per timestep"),
    "g":          (0.52, 0.72, 0.20, 0.13, "global PPV\n4998"),
    "het":        (0.52, 0.42, 0.20, 0.16, "H_m = sum_k q_k (PPV_{m,k}-PPV_m)^2\n4998"),
    "hydra":      (0.24, 0.34, 0.20, 0.10, "HYDRA competition\ncount_max/min 2048"),
    "ridge":      (0.80, 0.40, 0.17, 0.20, "RidgeClassifierCV\nconcat -> train+val"),
}
for k, (x, y, w, h, lab) in boxes.items():
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#eef2ff",
                               edgecolor="k", lw=1.2))
    ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=8)
def arrow(a, b):
    xa, ya, wa, ha_ = boxes[a][:4]
    xb, yb, wb, hb = boxes[b][:4]
    ax.annotate("", xy=(xb + wb / 2, yb + hb / 2), xytext=(xa + wa / 2, ya + ha_ / 2),
                arrowprops=dict(arrowstyle="->", lw=1.2))
for a, b in [("raw", "mr"), ("raw", "enc"), ("raw", "hydra"), ("enc", "vq"),
             ("mr", "g"), ("mr", "het"), ("vq", "het"), ("hydra", "ridge"),
             ("g", "ridge"), ("het", "ridge")]:
    arrow(a, b)
ax.set_title("RCMKN architecture (Haptics, seed 42)", fontsize=12)
fig.savefig(os.path.join(FIG_DIR, "fig1_architecture.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ---------------- fig 2: ablation bars ---------------------------------------
order = ["R0", "R1", "R2", "R3", "C1", "C2"]
labels = {"R0": "R0\nDRTN cond.", "R1": "R1\n+Hydra", "R2": "R2\nSSL cond.",
          "R3": "R3\nfull", "C1": "C1\nrandom", "C2": "C2\nshuffled"}
vals = [report["results"][v]["test_macro_f1"] for v in order]
fig, ax = plt.subplots(figsize=(8, 4.6))
colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
bars = ax.bar([labels[v] for v in order], vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
            ha="center", fontsize=9)
ax.axhline(0.4974, color="k", ls="--", lw=1,
           label="canonical M0 (seed 42) = 0.4974")
ax.set_ylabel("test Macro-F1")
ax.set_ylim(0.45, 0.60)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("RCMKN ablation ladder — test Macro-F1 (seed 42)")
fig.savefig(os.path.join(FIG_DIR, "fig2_ablation.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ---------------- fig 3: code usage ------------------------------------------
u0 = np.array(report["results"]["R0"]["regime_diagnostics"]["train"]["usage"])
u2 = np.array(report["results"]["R2"]["regime_diagnostics"]["train"]["usage"])
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axx, u, t in [(axes[0], u0, f"R0 official-DRTN regimes (ent={report['results']['R0']['regime_diagnostics']['train']['normalized_entropy']:.3f})"),
                  (axes[1], u2, f"R2 SSL-context regimes (ent={report['results']['R2']['regime_diagnostics']['train']['normalized_entropy']:.3f})")]:
    axx.bar(np.arange(1, K_CODES + 1), u, color="#4c72b0")
    axx.set_xlabel("code k"); axx.set_xticks(range(1, K_CODES + 1))
    axx.set_title(t, fontsize=9)
axes[0].set_ylabel("usage fraction")
fig.suptitle("Code-usage distribution (train+val)", y=1.02)
fig.savefig(os.path.join(FIG_DIR, "fig3_code_usage.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ---------------- fig 4: regime timeline -------------------------------------
s = 0
x = Xte_z[s]
fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
cmap = plt.get_cmap("tab10")
for axx, reg, t in [(axes[0], reg0_te[s], "R0 official-DRTN regimes"),
                    (axes[1], reg2_te_full[s], "R2 SSL-context regimes")]:
    axx.plot(np.arange(len(x)), x, color="k", lw=0.7, zorder=3)
    for k in range(K_CODES):
        axx.fill_between(np.arange(len(x)), x.min() - 0.5, x.max() + 0.5,
                         where=(reg == k), color=cmap(k), alpha=0.18, lw=0)
    axx.set_ylim(x.min() - 0.5, x.max() + 0.5)
    axx.set_ylabel("z-norm value")
    axx.set_title(t, fontsize=10, loc="left")
axes[1].set_xlabel("time step")
fig.suptitle(f"Test sample {s}: learned regime assignments (visualization only)",
             y=1.0)
fig.savefig(os.path.join(FIG_DIR, "fig4_regimes_timeline.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ---------------- fig 5: per-regime PPV conditioning --------------------------
act_s, _ = kf.compute_raw_activations(extractor, Xte_z[s:s + 1])
act_het = act_s[:, N_GLOBAL:, :][0]
reg_s = reg2_te_full[s]
ks = np.linspace(0, N_GLOBAL - 1, 8).astype(int)
fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharex=True)
for axx, m in zip(axes.ravel(), ks):
    vm = valid_het[m]
    ppv_mk, qk = [], []
    for k in range(K_CODES):
        sel = vm & (reg_s == k)
        n = int(sel.sum())
        ppv_mk.append(float(act_het[m, sel].mean()) if n else 0.0)
        qk.append(n / max(int(vm.sum()), 1))
    ppv_m = float(act_het[m, vm].mean())
    axx.bar(np.arange(K_CODES), ppv_mk, color=[cmap(k) for k in range(K_CODES)],
            alpha=0.8)
    axx.axhline(ppv_m, color="k", ls="--", lw=1, label="global PPV_m")
    axx.set_title(f"kernel {m}", fontsize=9)
    axx.set_ylim(0, 1)
axes[0, 0].set_ylabel("PPV_{m,k}"); axes[1, 0].set_ylabel("PPV_{m,k}")
for axx in axes[1]:
    axx.set_xlabel("regime k")
fig.suptitle("R2 regimes: per-regime MiniROCKET activation rates vs global "
             "(test sample 0, 8 kernels)")
fig.savefig(os.path.join(FIG_DIR, "fig5_regime_conditioning.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ---------------- fig 6: Hydra summary ---------------------------------------
Hd_trva, _ = kf.compute_hydra_features(Xtrva_z, data["L"], k=HYDRA["k"],
                                       g=HYDRA["g"])
nzc = (Hd_trva != 0).mean(axis=0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(nzc, bins=40, color="#4c72b0")
axes[0].set_xlabel("nonzero fraction (train+val)")
axes[0].set_ylabel("# features")
axes[0].set_title(f"Hydra sparsity (2048 features, overall "
                  f"{(Hd_trva != 0).mean():.3f} nonzero)", fontsize=10)
mag = np.sqrt(np.clip(Hd_trva, 0, None)).mean(axis=0)
order_idx = np.argsort(nzc)
axes[1].scatter(nzc[order_idx], mag[order_idx], s=4, color="#dd8452", alpha=0.5)
axes[1].set_xlabel("nonzero fraction")
axes[1].set_ylabel("mean sqrt-magnitude")
axes[1].set_title("magnitude vs sparsity", fontsize=10)
fig.suptitle("Hydra competitive/count block (after canonical _SparseScaler in "
             "the classifier)", y=1.03)
fig.savefig(os.path.join(FIG_DIR, "fig6_hydra_summary.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

print(f"saved 6 figures to {FIG_DIR}")
