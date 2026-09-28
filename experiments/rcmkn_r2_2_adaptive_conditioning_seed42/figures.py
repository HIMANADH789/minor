"""Figures for the R2.2 adaptive-conditioning screening (seed 42)."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results",
                   "rcmkn_r2_2_adaptive_conditioning_seed42")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

VARIANT_ORDER = ["B0", "B1", "B2", "B3A", "B3B"]
LABELS = {"B0": "B0 (R2 reference)", "B1": "B1 (zero-mod)",
          "B2": "B2 (R2.2)", "B3A": "B3A (random codes)",
          "B3B": "B3B (shuffled codes)"}
COLORS = {"B0": "#444444", "B1": "#999999", "B2": "#1f77b4",
          "B3A": "#d62728", "B3B": "#ff7f0e"}

report = json.load(open(os.path.join(RES, "report.json")))
per_ds = json.load(open(os.path.join(RES, "per_dataset_results.json")))
datasets = report["datasets"]
DS_COLORS = {"Haptics": "#1f77b4", "CWRU_BAL": "#2ca02c"}


def _load_mod(ds):
    ck = torch.load(os.path.join(RES, ds, "modulation_seed42.pt"),
                    map_location="cpu", weights_only=False)
    from experiments.rcmkn_r2_2_adaptive_conditioning_seed42 import (
        conditioning as cond)
    mod = cond.CodeModulation(K=8, M=cond.N_HET)
    mod.load_state_dict(ck["mod_state"])
    with torch.no_grad():
        s = torch.tanh(mod.a).numpy()
        beta = float(mod.beta_max * torch.sigmoid(mod.b).item())
    return s, beta


# 1. Macro-F1 by variant --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(VARIANT_ORDER))
w = 0.36
for j, ds in enumerate(datasets):
    vals = [per_ds[ds]["results"][v]["test_macro_f1"] for v in VARIANT_ORDER]
    ax.bar(x + (j - 0.5) * w, vals, w, label=ds, color=DS_COLORS[ds])
    for xi, v in zip(x + (j - 0.5) * w, vals):
        ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", fontsize=7)
ax.set_xticks(x, [LABELS[v] for v in VARIANT_ORDER], fontsize=8)
ax.set_ylim(0.4, 1.05)
ax.set_ylabel("Test Macro-F1 (seed 42)")
ax.set_title("R2.2 ablation ladder")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f1_macro_f1_by_variant.png"), dpi=150)
plt.close(fig)

# 2. Per-code modulation norms -------------------------------------------
fig, axes = plt.subplots(1, len(datasets), figsize=(9, 3.6), sharey=False)
for ax, ds in zip(axes, datasets):
    s, beta = _load_mod(ds)
    norms = np.linalg.norm(s, axis=1)
    ax.bar(np.arange(8), norms, color=DS_COLORS[ds])
    ax.set_title(f"{ds} (beta={beta:.3f})", fontsize=9)
    ax.set_xlabel("VQ code")
    ax.set_ylabel("||s_k||_2")
    ax.set_ylim(0, max(0.3, norms.max() * 1.15))
fig.suptitle("Per-code modulation norms (code-specific specialization)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f2_per_code_modulation_norms.png"), dpi=150)
plt.close(fig)

# 3. Distribution of s_k[m] ----------------------------------------------
fig, axes = plt.subplots(1, len(datasets), figsize=(9, 3.6))
for ax, ds in zip(axes, datasets):
    s, _ = _load_mod(ds)
    ax.hist(s.ravel(), bins=80, color=DS_COLORS[ds], alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{ds}: mean|s|={np.abs(s).mean():.4f}, "
                 f"max|s|={np.abs(s).max():.4f}", fontsize=9)
    ax.set_xlabel("s_k[m] = tanh(a)")
    ax.set_ylabel("count")
fig.suptitle("Distribution of modulation values (bounded, identity init)",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f3_modulation_distribution.png"), dpi=150)
plt.close(fig)

# 4. Example temporal sequence: activation / code / modulation factor -----
ds = "CWRU_BAL"
sample_i = 0
Xz, codes_te = None, None
fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
try:
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner \
        as transfer
    data = transfer.load_any_dataset(ds)
    xz = transfer.znorm(data["Xte"][sample_i:sample_i + 1]).ravel()
    preds = np.load(os.path.join(RES, "predictions",
                                 f"{ds}_B2_pred_te.npy"))
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    from experiments.rcmkn_haptics_seed42.runner import (
        extract_context_regimes, set_seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RCMKNContextModel(n_classes=data["n_classes"])
    ck = torch.load(os.path.join(ROOT, "results",
                                 "rcmkn_ssl_context_important2_seed42",
                                 ds, "context_model_seed42.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model = model.to(dev).eval()
    set_seed(42)
    codes = extract_context_regimes(model,
                                    transfer.znorm(data["Xte"][:1]),
                                    dev, batch=1)[0]
    bias_sign = np.load(os.path.join(RES, ds, "bias_sign.npy"))
    tau_signed = np.load(os.path.join(RES, ds, "tau_table.npy"))
    s, beta = _load_mod(ds)
    scale = 1.0 + beta * s[codes]                    # (T, 4998)
    # a strongly-modulated, active kernel: max |delta| * usage
    m_star = int(np.argmax(np.abs(s).max(0) *
                           (np.abs(s) > 0.05).mean(0)))
    m_flat = int(np.argmin(np.abs(s).max(0)))
    axes[0].plot(xz, lw=0.7, color="#444444")
    axes[0].set_ylabel("z-normed signal")
    for code in range(8):
        m = codes == code
        if m.any():
            axes[1].fill_between(np.arange(len(codes)), 0, 1,
                                 where=m, step="mid", alpha=0.6)
    axes[1].set_ylabel(f"VQ code (k_t), sample {sample_i}")
    axes[2].step(np.arange(len(codes)), scale[:, m_star], where="mid",
                 color="#1f77b4", lw=0.8)
    axes[2].set_ylabel(f"1+beta*s_{{k_t}}[{m_star}]")
    axes[2].set_title(f"kernel {m_star} (most-modulated)", fontsize=9)
    axes[3].step(np.arange(len(codes)), scale[:, m_flat], where="mid",
                 color="#7f7f7f", lw=0.8)
    axes[3].set_ylabel(f"1+beta*s_{{k_t}}[{m_flat}]")
    axes[3].set_title(f"kernel {m_flat} (least-modulated)", fontsize=9)
    axes[3].set_xlabel("t")
    fig.suptitle(f"{ds} test sample {sample_i}: signal, learned code, "
                 "modulation factor (B2)", fontsize=10)
except Exception as e:  # noqa: BLE001
    fig.clf()
    axes = fig.subplots(1, 1)
    axes.text(0.5, 0.5, f"context replay failed: {e}", ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f4_temporal_example.png"), dpi=150)
plt.close(fig)

# 5. B2 vs B3 controls ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.0))
x = np.arange(len(datasets))
for j, (v, ls) in enumerate((("B2", "-"), ("B3A", "--"), ("B3B", ":"))):
    vals = [per_ds[ds]["results"][v]["test_macro_f1"] for ds in datasets]
    ax.plot(x, vals, ls, marker="o", label=LABELS[v], color=COLORS[v])
for i, ds in enumerate(datasets):
    for v in ("B0", "B1"):
        ax.scatter([i], [per_ds[ds]["results"][v]["test_macro_f1"]],
                   marker="_", s=400, color="#444444", zorder=3)
ax.set_xticks(x, datasets)
ax.set_ylabel("Test Macro-F1")
ax.set_title("B2 vs occupancy/shuffled-code controls (grey = B0=B1)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f5_b2_vs_controls.png"), dpi=150)
plt.close(fig)

# 6. Cross-dataset B2-B0 deltas ------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.8))
deltas = [per_ds[ds]["deltas"]["B2-B0"] for ds in datasets]
ax.bar(datasets, deltas,
       color=["#2ca02c" if d > 0 else "#d62728" for d in deltas])
for i, d in enumerate(deltas):
    ax.text(i, d + (1e-4 if d >= 0 else -3e-4), f"{d:+.4f}",
            ha="center", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("B2 - B0 (test Macro-F1)")
ax.set_title("Cross-dataset R2.2 effect (seed 42)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f6_cross_dataset_deltas.png"), dpi=150)
plt.close(fig)

print("figures saved to", FIG)
