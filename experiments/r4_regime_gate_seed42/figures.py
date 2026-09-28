"""R4 (differentiable-Ridge) figures, spec section 26 (8 figures).

Reads only the saved result JSONs; no test labels are used for any
figure-design decision.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.r4_regime_gate_seed42.core import DATASETS, OUT_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "r4_differentiable_ridge_"
                                        "regime_gate_seed42")


def _load(ds):
    d = os.path.join(RESULTS, ds)
    with open(os.path.join(d, "validation_results.json")) as f:
        val = json.load(f)
    with open(os.path.join(d, "test_results.json")) as f:
        te = json.load(f)
    with open(os.path.join(d, "gate_metrics.json")) as f:
        gm = json.load(f)
    with open(os.path.join(d, "regime_statistics.json")) as f:
        rs = json.load(f)
    with open(os.path.join(d, "training_logs", "gate_trajectory.json")) as f:
        traj = json.load(f)
    return val, te, gm, rs, traj


def fig1_validation(all_results):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dss = list(all_results)
    x = np.arange(len(dss))
    w = 0.27
    for j, (key, label) in enumerate((("G", "G (M0 Ridge)"),
                                      ("GH", "G+H (v=1, R2 Ridge)"),
                                      ("R4", "R4 learned gates"))):
        vals = [all_results[d]["validation"][key]["val_macro_f1"]
                for d in dss]
        ax.bar(x + (j - 1) * w, vals, w, label=label)
    ax.set_xticks(x, dss)
    ax.set_ylabel("Validation Macro-F1 (trainva fit)")
    ax.set_title("R4 validation: G vs G+H (v=1) vs learned gates")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fig1_validation.png"),
                dpi=150)
    plt.close(fig)


def _traj_figs(ds, fig_id, title):
    val, te, gm, rs, traj = _load(ds)
    steps = [t["step"] for t in traj]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("tab10")
    for k in range(8):
        ax.plot(steps, [t["v"][k] for t in traj], color=cmap(k),
                label=f"v{k}", lw=1.4)
    ax.set_xlabel("outer step")
    ax.set_ylabel("gate value v_k = sigmoid(theta_k)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures",
                             f"fig{fig_id}_trajectories_{ds}.png"), dpi=150)
    plt.close(fig)


def fig4_5_final_gates(all_results):
    fig, axes = plt.subplots(1, len(all_results),
                             figsize=(5.2 * len(all_results), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (ds, r) in zip(axes, all_results.items()):
        vb = r["gate_metrics_v_best"] if "gate_metrics_v_best" in r \
            else json.load(open(os.path.join(
                RESULTS, ds, "gate_metrics.json")))["v_best"]
        ax.bar(np.arange(8), vb, color="steelblue")
        ax.axhline(0.5, color="k", ls="--", lw=0.8, label="0.5")
        ax.set_ylim(0, 1)
        ax.set_xlabel("regime k")
        ax.set_ylabel("final v_k (best-val)")
        ax.set_title(f"{ds}: final gates")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fig4_5_final_gates.png"),
                dpi=150)
    plt.close(fig)


def fig6_test(all_results):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    dss = list(all_results)
    x = np.arange(len(dss))
    w = 0.27
    for j, (key, label) in enumerate((("G", "M0 Ridge (G)"),
                                      ("GH", "R2 Ridge (G+H)"),
                                      ("R4", "R4 learned gates"))):
        vals = [all_results[d]["test"][key]["test_macro_f1"] for d in dss]
        ax.bar(x + (j - 1) * w, vals, w, label=label)
        for xi, v in zip(x + (j - 1) * w, vals):
            ax.text(xi, v + 0.005, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, dss)
    ax.set_ylabel("Official test Macro-F1")
    ax.set_title("R4 official test: M0 vs R2 vs R4")
    ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fig6_test.png"), dpi=150)
    plt.close(fig)


def fig7_8_gate_vs_regime_props(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ds, r in all_results.items():
        rs = json.load(open(os.path.join(RESULTS, ds,
                                         "regime_statistics.json")))
        vb = json.load(open(os.path.join(RESULTS, ds,
                                         "gate_metrics.json")))["v_best"]
        occ = [s["occupancy_fraction"] for s in rs]
        contrib = [s["fraction_of_H_contribution"] for s in rs]
        axes[0].scatter(occ, vb, s=42, label=ds)
        axes[1].scatter(contrib, vb, s=42, label=ds)
        for s, v in zip(rs, vb):
            axes[0].annotate(f"k{s['regime']}", (s["occupancy_fraction"], v),
                             fontsize=7, xytext=(3, 2),
                             textcoords="offset points")
            axes[1].annotate(f"k{s['regime']}",
                             (s["fraction_of_H_contribution"], v),
                             fontsize=7, xytext=(3, 2),
                             textcoords="offset points")
    axes[0].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("regime occupancy fraction")
    axes[0].set_ylabel("final gate v_k")
    axes[0].set_title("gate vs occupancy")
    axes[1].set_xlabel("fraction of total H contributed")
    axes[1].set_ylabel("final gate v_k")
    axes[1].set_title("gate vs raw H contribution")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures",
                             "fig7_8_gate_vs_regime_props.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    with open(os.path.join(RESULTS, "all_results.json")) as f:
        all_results = json.load(f)
    fig1_validation(all_results)
    _traj_figs("Haptics", 2, "Haptics: eight regime-gate trajectories")
    _traj_figs("ECG5000_BAL", 3, "ECG5000_BAL: eight regime-gate "
                                 "trajectories")
    fig4_5_final_gates(all_results)
    fig6_test(all_results)
    fig7_8_gate_vs_regime_props(all_results)
    print("figures written:", os.listdir(os.path.join(RESULTS, "figures")))


if __name__ == "__main__":
    main()
