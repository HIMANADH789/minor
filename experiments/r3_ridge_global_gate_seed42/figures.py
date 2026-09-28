"""R3 (differentiable-Ridge global gate) figures, spec section 29.

Reads only saved result JSONs; no test labels inform figure design.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results",
                       "r3_differentiable_ridge_global_gate_seed42")


def _load(ds):
    d = os.path.join(RESULTS, ds)
    with open(os.path.join(d, "validation_results.json")) as f:
        val = json.load(f)
    with open(os.path.join(d, "test_results.json")) as f:
        te = json.load(f)
    with open(os.path.join(d, "training_logs", "gate_trajectory.json")) as f:
        traj = json.load(f)
    return val, te, traj


def fig1_validation(all_results):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dss = list(all_results)
    x = np.arange(len(dss))
    w = 0.27
    for j, (key, label) in enumerate((("G", "M0 Ridge (G)"),
                                      ("GH", "R2 Ridge (G+H)"),
                                      ("R3", "R3 learned gate"))):
        vals = [all_results[d]["validation"][key]["val_macro_f1"]
                for d in dss]
        ax.bar(x + (j - 1) * w, vals, w, label=label)
    ax.set_xticks(x, dss)
    ax.set_ylabel("Validation Macro-F1 (trainva fit)")
    ax.set_title("R3 validation: M0 vs R2 vs learned global gate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fig1_validation.png"),
                dpi=150)
    plt.close(fig)


def _traj_fig(ds, fig_id, title):
    val, te, traj = _load(ds)
    steps = [t["step"] for t in traj]
    ws = [t["w"] for t in traj]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(steps, ws, color="steelblue", lw=1.8)
    best_step = val["R3"]["best_step_mf1"]
    best_w = val["R3"]["w_best"]
    ax.scatter([best_step], [best_w], color="crimson", zorder=3,
               label=f"best-val w={best_w:.4f} @ step {best_step}")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.text(steps[0], 1.01, "w=1 (R2)", fontsize=8, color="gray")
    ax.set_xlabel("outer step")
    ax.set_ylabel("gate value w = sigmoid(theta)")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures",
                             f"fig{fig_id}_trajectory_{ds}.png"), dpi=150)
    plt.close(fig)


def fig4_final_w(all_results):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    dss = list(all_results)
    x = np.arange(len(dss))
    wv = 0.27
    for j, (key, label) in enumerate((("w_init", "w init"),
                                      ("w_best", "w best-val"),
                                      ("w_final", "w final"))):
        vals = [all_results[d]["gate"][key] for d in dss]
        ax.bar(x + (j - 1) * wv, vals, wv, label=label)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="0.5")
    ax.set_xticks(x, dss)
    ax.set_ylim(0, 1)
    ax.set_ylabel("gate value w")
    ax.set_title("Final learned global gate per dataset")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fig4_final_w.png"),
                dpi=150)
    plt.close(fig)


def fig5_valf1_vs_w(all_results):
    fig, axes = plt.subplots(1, len(all_results),
                             figsize=(6.2 * len(all_results), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (ds, r) in zip(axes, all_results.items()):
        val, te, traj = _load(ds)
        ws = [t["w"] for t in traj]
        f1 = [t["val_macro_f1"] for t in traj]
        ax.scatter(ws, f1, s=14, alpha=0.65, color="steelblue")
        bw = r["gate"]["w_best"]
        ax.axvline(bw, color="crimson", ls="--", lw=1.0,
                   label=f"best-val w={bw:.4f}")
        ax.axvline(1.0, color="gray", ls=":", lw=0.8, label="w=1 (R2)")
        ax.set_xlabel("gate value w")
        ax.set_ylabel("validation Macro-F1")
        ax.set_title(f"{ds}: validation F1 vs gate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fig5_valf1_vs_w.png"),
                dpi=150)
    plt.close(fig)


def fig6_test(all_results):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    dss = list(all_results)
    x = np.arange(len(dss))
    wv = 0.27
    for j, (key, label) in enumerate((("G", "M0 Ridge (G)"),
                                      ("GH", "R2 Ridge (G+H)"),
                                      ("R3", "R3 learned gate"))):
        vals = [all_results[d]["test"][key]["test_macro_f1"] for d in dss]
        ax.bar(x + (j - 1) * wv, vals, wv, label=label)
        for xi, v in zip(x + (j - 1) * wv, vals):
            ax.text(xi, v + 0.005, f"{v:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, dss)
    ax.set_ylabel("Official test Macro-F1")
    ax.set_title("R3 official test: M0 vs R2 vs learned gate")
    ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "fig6_test.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    with open(os.path.join(RESULTS, "all_results.json")) as f:
        all_results = json.load(f)
    fig1_validation(all_results)
    _traj_fig("Haptics", 2, "Haptics: global gate trajectory")
    _traj_fig("ECG5000_BAL", 3, "ECG5000_BAL: global gate trajectory")
    fig4_final_w(all_results)
    fig5_valf1_vs_w(all_results)
    fig6_test(all_results)
    print("figures written:",
          sorted(os.listdir(os.path.join(RESULTS, "figures"))))


if __name__ == "__main__":
    main()
