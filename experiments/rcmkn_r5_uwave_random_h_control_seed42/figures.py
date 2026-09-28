"""Figures for the random-H control."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SHORT = {"UWaveGestureLibraryAll": "All", "UWaveGestureLibraryX": "X",
         "UWaveGestureLibraryY": "Y", "UWaveGestureLibraryZ": "Z"}


def _dataset_panel(r, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rank, rm, rs = r["ranked_test"], r["random_mean"], r["random_std"]
    ax.axhspan(rm - rs, rm + rs, color="tab:blue", alpha=0.15,
               label=f"random mean ± std ({rm:.4f} ± {rs:.4f})")
    ax.axhline(rank, color="tab:red", lw=2,
               label=f"R5 ranked ({rank:.4f})")
    for rs_seed, t in zip([420001, 420002, 420003, 420004, 420005],
                          r["per_subset_test"]):
        ax.plot(rs_seed, t, "o", color="tab:blue")
    ax.set_xlabel("random subset seed")
    ax.set_ylabel("test Macro-F1")
    ax.set_title(f"{SHORT[r['dataset']]}: ranked vs random-H subsets "
                 f"({r['outcome']})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(all_res, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    # Figures 1-4: per-dataset ranked vs 5 random subsets
    for i, (ds, r) in enumerate(all_res.items(), start=1):
        _dataset_panel(r, os.path.join(fig_dir, f"fig{i}_ranked_vs_random_"
                                       f"{SHORT[ds]}.png"))
    # Figure 5: ranked vs random mean ± std across datasets
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [SHORT[ds] for ds in all_res]
    x = np.arange(len(names))
    ranked = [r["ranked_test"] for r in all_res.values()]
    means = [r["random_mean"] for r in all_res.values()]
    stds = [r["random_std"] for r in all_res.values()]
    ax.errorbar(x - 0.08, means, yerr=stds, fmt="o", color="tab:blue",
                capsize=4, label="random-H mean ± std (5 subsets)")
    ax.plot(x + 0.08, ranked, "s", color="tab:red", label="R5 ranked")
    ax.set_xticks(x, names)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("test Macro-F1")
    ax.set_title("Ranked vs random-H control across UWave datasets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig5_ranked_vs_random_all.png"),
                dpi=150)
    plt.close(fig)
    # Figure 6: ranked-minus-random difference
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    diffs = [r["diff"] for r in all_res.values()]
    colors = ["tab:green" if d > 0 else "tab:orange" for d in diffs]
    ax.bar(names, diffs, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    for i, d in enumerate(diffs):
        ax.text(i, d, f"{d:+.4f}", ha="center",
                va="bottom" if d > 0 else "top", fontsize=9)
    ax.set_ylabel("test Macro-F1: ranked − random mean")
    ax.set_title("Selection value of the F-statistic ranking (Δ select)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_delta_select.png"), dpi=150)
    plt.close(fig)
