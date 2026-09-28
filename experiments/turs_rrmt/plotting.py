"""Publication figures for the TURS-RRMT experiment (PNG + PDF)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, fig_dir, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)


def make_figures(all_results, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # 1. ablation Macro-F1 comparison across datasets
    fig, ax = plt.subplots(figsize=(8, 4.5))
    variants = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    width = 0.8 / len(variants)
    tags = list(all_results)
    for i, v in enumerate(variants):
        vals = [R["variants"].get(v, {}).get("test", {}).get("macro_f1")
                for R in all_results.values()]
        pos = np.arange(len(tags)) + i * width - 0.4 + width / 2
        ax.bar(pos, [x or 0 for x in vals], width, label=v)
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Ablation ladder (A4 routed vs A3 uniform = central test)")
    ax.legend(ncol=4, fontsize=8)
    ax.set_ylim(0, 1)
    _save(fig, fig_dir, "ablation_comparison")

    # 2. A4 vs A3 per dataset with CI
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (tag, R) in enumerate(all_results.items()):
        d = R.get("A4_vs_A3", {})
        if not d:
            continue
        ax.errorbar(i, d["acc_diff"], fmt="o",
                    yerr=[[d["acc_diff"] - d["acc_diff_ci95"][0]],
                          [d["acc_diff_ci95"][1] - d["acc_diff"]]],
                    capsize=4, label=tag)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(all_results)))
    ax.set_xticklabels(all_results, rotation=15)
    ax.set_ylabel("Acc diff (A4 - A3), 95% CI")
    ax.set_title("Routing benefit: A4 vs A3 paired accuracy difference")
    _save(fig, fig_dir, "a4_vs_a3_ci")

    # 3-6. per-dataset routing timelines, entropy, degradation, D4
    for tag, R in all_results.items():
        fp = os.path.join(os.path.dirname(fig_dir), tag,
                          "routing_outputs_test.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp)
        W = z["routing_weights"]                     # [N, J, T]

        # routing timelines for first 3 samples
        fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
        for s in range(3):
            for j in range(W.shape[1]):
                axes[s].plot(W[s, j], lw=0.8, label=f"flavor {j}" if s == 0 else None)
            axes[s].set_ylim(0, 1)
        axes[0].legend(ncol=4, fontsize=7)
        axes[2].set_xlabel("temporal position")
        fig.suptitle(f"{tag}: routing weight timelines (3 test samples)")
        _save(fig, fig_dir, f"{tag}_routing_timelines")

        # routing entropy histogram
        re = -(W * np.log(W + 1e-12)).sum(1).mean(1)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(re, bins=40, color="steelblue")
        ax.set_xlabel("mean routing entropy")
        ax.set_title(f"{tag}: routing entropy distribution")
        _save(fig, fig_dir, f"{tag}_routing_entropy")

        # flavor selection frequency
        fig, ax = plt.subplots(figsize=(5, 3.5))
        freq = np.bincount(W.argmax(1).ravel(), minlength=W.shape[1]) / W[..., 0].size
        ax.bar(range(W.shape[1]), freq)
        ax.set_xticks(range(W.shape[1]))
        ax.set_xticklabels(["standard", "tail", "fine", "multilag"])
        ax.set_ylabel("top-1 selection frequency")
        ax.set_title(f"{tag}: flavor selection")
        _save(fig, fig_dir, f"{tag}_flavor_frequency")

        # degradation curves if available
        d5 = R.get("diagnostics", {}).get("D5", {})
        if d5.get("kinds"):
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            for kind, recs in d5["kinds"].items():
                lev = [r["level"] for r in recs]
                axes[0].plot(lev, [r["error_rate"] for r in recs], marker="o", label=kind)
                axes[1].plot(lev, [r["routing_entropy"] for r in recs], marker="o")
                axes[2].plot(lev, [r["mean_conf"] for r in recs], marker="o")
            axes[0].set_title("error rate"); axes[1].set_title("routing entropy")
            axes[2].set_title("confidence")
            axes[0].legend(fontsize=7)
            fig.suptitle(f"{tag}: degradation response")
            _save(fig, fig_dir, f"{tag}_degradation")

        # targeted vs random faithfulness
        d4 = R.get("diagnostics", {}).get("D4", {})
        if d4:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.bar(["targeted", "random"], [d4["target_drop"], d4["random_drop"]],
                   color=["firebrick", "gray"])
            ax.set_ylabel("mean confidence drop")
            ax.set_title(f"{tag}: D4 faithfulness (p={d4['p_perm']:.4f})")
            _save(fig, fig_dir, f"{tag}_faithfulness")
    print(f"  figures -> {fig_dir}", flush=True)
