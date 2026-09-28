"""
Figures: Haptics Fixed-Feature + Learned-Regime Ensemble (spec sec. 16).

Reads results/haptics_ensemble_seed42/{report.json, diagnostics/fusion_search.json,
predictions/test_per_sample.csv} and writes four PNGs into figures/.

No significance markers, no confidence bands (single-seed exploratory evidence).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "haptics_ensemble_seed42")


def load():
    report = json.load(open(os.path.join(RES, "report.json")))
    fusion = json.load(open(os.path.join(RES, "diagnostics", "fusion_search.json")))
    return report, fusion


def fig1_fusion_alpha(fusion):
    xs = [d["alpha"] for d in fusion["search"]]
    ys = [d["val_macro_f1"] for d in fusion["search"]]
    sel = fusion["selected_alpha"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, ys, "o-", ms=4, lw=1.5, color="#2563eb")
    ax.axvline(sel, color="#dc2626", ls="--", lw=1.2,
               label=f"selected alpha={sel:.2f} (val MF1={max(ys):.4f})")
    ax.set_xlabel("fusion weight  alpha  (1.0 = MiniROCKET only)")
    ax.set_ylabel("validation Macro-F1")
    ax.set_title("FIGURE 1 — Validation-only fusion search "
                 "(P_fused = a·softmax(MR/T) + (1-a)·softmax(DRTN/T))")
    ax.set_ylim(0.45, 0.70)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "fusion_alpha.png"), dpi=150)
    plt.close(fig)


def fig2_error_overlap(report):
    tv = report["validation_stage"]["complementarity_val"]
    tt = report["complementarity_test"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    for ax, tbl, title in (
        (axes[0], tv, "VALIDATION (n=23) — selection stage"),
        (axes[1], tt, "TEST (n=308) — reported after freeze"),
    ):
        m = np.array([[tbl["A_mr_and_drtn_correct"], tbl["B_mr_only_correct"]],
                      [tbl["C_drtn_only_correct"], tbl["D_both_wrong"]]])
        im = ax.imshow(m, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks([0, 1], ["DRTN correct", "DRTN wrong"])
        ax.set_yticks([0, 1], ["MR correct", "MR wrong"])
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("FIGURE 2 — Correctness matrix  (C = DRTN-only correct is the "
                 "complementary-information cell)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "error_overlap.png"), dpi=150)
    plt.close(fig)


def fig3_per_class_f1(report):
    ft = report["final_test"]
    order = [
        ("MiniROCKET (train+val refit)", "#2563eb"),
        ("DRTN R5 (frozen checkpoint)", "#dc2626"),
        ("Fusion alpha=0.95", "#f59e0b"),
        ("Stacked MR+DRTN", "#7c3aed"),
        ("Hydra (train+val refit)", "#059669"),
        ("MultiRocketHydra (train+val refit)", "#0d9488"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.13
    xs = np.arange(5)
    for i, (key, color) in enumerate(order):
        f1s = ft[key]["class_f1s"]
        ax.bar(xs + (i - 2.5) * w, f1s, w, label=key.split(" (")[0], color=color)
    ax.set_xticks(xs, [f"class {c}" for c in range(5)])
    ax.set_ylabel("test per-class F1")
    ax.set_title("FIGURE 3 — Test per-class F1 (class 0 = largest Haptics minority pain point)")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "per_class_f1.png"), dpi=150)
    plt.close(fig)


def fig4_final_comparison(report):
    ft = report["final_test"]
    labels = ["MiniROCKET\n(canonical)", "DRTN R5", "Fusion\n(a=0.95)",
              "Stacked\nMR+DRTN", "Hydra", "MultiRocket\nHydra"]
    vals = [ft["MiniROCKET (train+val refit)"]["test_macro_f1"],
            ft["DRTN R5 (frozen checkpoint)"]["test_macro_f1"],
            ft["Fusion alpha=0.95"]["test_macro_f1"],
            ft["Stacked MR+DRTN"]["test_macro_f1"],
            ft["Hydra (train+val refit)"]["test_macro_f1"],
            ft["MultiRocketHydra (train+val refit)"]["test_macro_f1"]]
    colors = ["#2563eb", "#dc2626", "#f59e0b", "#7c3aed", "#059669", "#0d9488"]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(vals[0], color="#2563eb", ls=":", lw=1.2,
               label="canonical MiniROCKET reference (0.4974)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", fontsize=8.5)
    ax.set_ylabel("test Macro-F1")
    ax.set_ylim(0, 0.62)
    ax.set_title("FIGURE 4 — Final frozen test Macro-F1, one evaluation per system\n"
                 "(single seed, exploratory — no significance markers)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "final_comparison.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    report, fusion = load()
    fig1_fusion_alpha(fusion)
    fig2_error_overlap(report)
    fig3_per_class_f1(report)
    fig4_final_comparison(report)
    for f in sorted(os.listdir(os.path.join(RES, "figures"))):
        print("figures/" + f)


if __name__ == "__main__":
    main()
