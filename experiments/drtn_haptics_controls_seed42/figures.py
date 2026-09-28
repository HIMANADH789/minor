"""
Generate the controlled-experiment figures from results/drtn_haptics_controls_seed42/:
  fig1_master.png        : val/test Macro-F1 per model + MiniROCKET reference
  fig2_params.png        : parameter counts by module
  fig3_k_effect.png      : K vs test MF1 / entropy / perplexity
  fig4_usage.png         : code usage histograms per run
  fig5_traj_compare.png  : continuous (CTC) vs discrete (DTC) latent trajectory on one val example
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "results", "drtn_haptics_controls_seed42")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

ORDER = ["continuous_control", "discrete_control", "r5_k8", "r5_k16", "r5_k32"]
LABELS = {"continuous_control": "CTC\n(continuous)",
          "discrete_control": "DTC\n(VQ, no div)",
          "r5_k8": "R5 K=8\n(official)",
          "r5_k16": "R5 K=16",
          "r5_k32": "R5 K=32"}


def load(name):
    with open(os.path.join(BASE, name, "result.json")) as f:
        return json.load(f)


def main():
    R = {n: load(n) for n in ORDER}
    report = json.load(open(os.path.join(BASE, "report.json")))

    # ---- FIG 1: master comparison ----
    val = [R[n]["best_val_mf1"] for n in ORDER]
    test = [R[n]["test"]["macro_f1"] for n in ORDER]
    mr = report["minirocket_reference"]["test_macro_f1"]
    x = np.arange(len(ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, val, w, label="val Macro-F1", color="#9ecae1")
    ax.bar(x + w / 2, test, w, label="test Macro-F1", color="#3182bd")
    ax.axhline(mr, color="#de2d26", ls="--", lw=1.4,
               label=f"MiniROCKET 0.4974 (non-neural baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in ORDER], fontsize=9)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Parameter-matched trajectory comparison (Haptics, seed 42)")
    for i, v in enumerate(test):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_master.png"), dpi=160)
    plt.close(fig)

    # ---- FIG 2: parameter audit ----
    mods = ["encoder", "trajectory", "pool", "classifier", "codebook_buffers"]
    mod_labels = ["encoder", "trajectory", "attention pool", "classifier",
                  "EMA codebook (buffer)"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(ORDER))
    colors = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#969696"]
    for m, lab, c in zip(mods, mod_labels, colors):
        vals = [R[n]["param_audit"][m] for n in ORDER]
        ax.bar(range(len(ORDER)), vals, bottom=bottom, label=lab, color=c)
        bottom += np.array(vals)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABELS[n] for n in ORDER], fontsize=9)
    ax.set_ylabel("parameters (log scale)")
    ax.set_title("Parameter audit — identical 599,413 trainable everywhere;\n"
                 "only EMA codebook buffer size differs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_params.png"), dpi=160)
    plt.close(fig)

    # ---- FIG 3: K effect ----
    ks = [8, 16, 32]
    kk = ["r5_k8", "r5_k16", "r5_k32"]
    test_k = [R[n]["test"]["macro_f1"] for n in kk]
    val_k = [R[n]["best_val_mf1"] for n in kk]
    Hn = [R[n]["final_diag_val"].get("normalized_entropy") for n in kk]
    ppl = [R[n]["final_diag_val"].get("perplexity") for n in kk]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(ks, val_k, "o-", label="val")
    axes[0].plot(ks, test_k, "s-", label="test")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("K vs performance"); axes[0].legend(fontsize=8)
    axes[1].plot(ks, Hn, "o-", color="#fd8d3c")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("normalized usage entropy")
    axes[1].set_title("K vs codebook entropy")
    axes[2].plot(ks, ppl, "o-", color="#6baed6")
    axes[2].set_xlabel("K"); axes[2].set_ylabel("perplexity")
    axes[2].set_title("K vs perplexity")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_k_effect.png"), dpi=160)
    plt.close(fig)

    # ---- FIG 4: usage histograms ----
    vq_runs = ["discrete_control", "r5_k8", "r5_k16", "r5_k32"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, n in zip(axes, vq_runs):
        u = R[n].get("codebook_final_usage_val")
        if u is None:
            continue
        ax.bar(range(len(u)), u, color="#2ca02c")
        fdv = R[n]["final_diag_val"]
        ax.set_title(f"{n}\nact={fdv['active_codes']} "
                     f"Hn={fdv['normalized_entropy']:.2f} "
                     f"ppl={fdv['perplexity']:.1f}", fontsize=9)
        ax.set_xlabel("code"); ax.set_ylim(0, max(u) * 1.25)
    axes[0].set_ylabel("usage fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_usage.png"), dpi=160)
    plt.close(fig)

    # ---- FIG 5: continuous vs discrete latent trajectory (1 val example) ----
    x_in = np.load(os.path.join(BASE, "discrete_control", "inputs_val.npy"))[0, 0]
    k_t = np.load(os.path.join(BASE, "discrete_control", "trajectories_val.npy"))[0]
    attn_dtc = np.load(os.path.join(BASE, "discrete_control", "attn_val.npy"))[0]
    attn_ctc = np.load(os.path.join(BASE, "continuous_control", "attn_val.npy"))[0]
    T = len(x_in)
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(np.arange(T), x_in, lw=0.6, color="#1f77b4")
    axes[0].set_ylabel("z-norm signal")
    axes[0].set_title("Continuous vs discrete latent trajectory (val example 0)",
                      loc="left", fontsize=10)
    axes[1].step(np.arange(T), k_t, where="post", lw=0.9, color="#d62728")
    axes[1].set_ylabel("VQ code k(t)\n(DTC)")
    axes[2].plot(np.arange(T), attn_ctc, lw=0.7, color="#31a354")
    axes[2].set_ylabel("attention a(t)\nCTC (continuous)")
    axes[3].plot(np.arange(T), attn_dtc, lw=0.7, color="#756bb1")
    axes[3].set_ylabel("attention a(t)\nDTC (discrete)")
    axes[3].set_xlabel("time step")
    for b in np.nonzero(np.diff(k_t))[0]:
        for ax in axes:
            ax.axvline(b, color="gray", alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_traj_compare.png"), dpi=160)
    plt.close(fig)

    print("figures written:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
