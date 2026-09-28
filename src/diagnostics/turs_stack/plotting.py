"""Stack publication figures (Phase 26). Deterministic; PNG + PDF."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"lite": "#4878CF", "rv": "#EE854A", "cs": "#6ACC64", "cmr": "#D65F5F"}


def _save(fig, name, tag):
    out = os.path.join(C.FIG_DIR, tag)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_branch_pca(ext, tag):
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, b in zip(axes, ["lite", "rv", "cs", "cmr"]):
        z = ext["test"][f"z_t_{b}"].mean(1)
        y = ext["test"]["y"]
        if len(z) > 2000:
            idx = np.random.default_rng(0).permutation(len(z))[:2000]
            z, y = z[idx], y[idx]
        p = PCA(n_components=2, random_state=0).fit_transform(z)
        for c in np.unique(y):
            m = y == c
            ax.scatter(p[m, 0], p[m, 1], s=3, alpha=0.4, label=f"c{c}")
        ax.set_title(f"{b} branch z")
    axes[0].legend(markerscale=2, fontsize=6)
    _save(fig, "01_branch_latent_pca", tag)


def fig_traces(ext, tag, n=4):
    e = ext["test"]
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.0 * n))
    rng = np.random.default_rng(0)
    for i in range(n):
        k = int(rng.integers(0, len(e["y"])))
        axes[i, 0].plot(np.linalg.norm(e["v_t_cs"][k], axis=1), color=COLORS["cs"])
        axes[i, 0].set_ylabel("||v_t|| cs")
        axes[i, 0].set_title(f"sample {k} y={e['y'][k]} pred={e['final_pred'][k]}",
                             fontsize=7)
        axes[i, 1].plot(e["u_t_cs"][k][:, 0], color=COLORS["cmr"])
        axes[i, 1].set_ylabel("u_t cs")
        axes[i, 2].plot(e["js_disagreement"][k] * np.ones(
            e["u_t_cs"].shape[1]), color="#555")
        axes[i, 2].plot(e["beta_cs"][k], lw=0.8)
        axes[i, 2].set_ylabel("beta / JS")
    fig.suptitle(f"Temporal diagnostic traces — {tag}", y=1.01)
    fig.tight_layout()
    _save(fig, "02_velocity_uncertainty_traces", tag)


def fig_unc_distributions(ext, tag):
    e = ext["test"]; u = e["u_cs_mean"]; corr = e["final_correct"] == 1
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    axes[0].hist(u[corr], bins=40, alpha=0.6, label="correct", density=True)
    axes[0].hist(u[~corr], bins=40, alpha=0.6, label="incorrect", density=True)
    axes[0].set_xlabel("u_cs mean"); axes[0].legend()
    axes[0].set_title("Intrinsic u: correct vs incorrect")
    js = e["js_disagreement"]
    axes[1].hist(js[corr], bins=40, alpha=0.6, label="correct", density=True)
    axes[1].hist(js[~corr], bins=40, alpha=0.6, label="incorrect", density=True)
    axes[1].set_xlabel("JS branch disagreement"); axes[1].legend()
    axes[1].set_title("Ensemble disagreement (NOT intrinsic u)")
    _save(fig, "03_uncertainty_disagreement_dist", tag)


def fig_error_detection(ext, tag):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    e = ext["test"]; yerr = (e["final_correct"] == 0).astype(int)
    if yerr.sum() in (0, len(yerr)):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for name, score in [("u_cs (intrinsic)", e["u_cs_mean"]),
                        ("JS disagreement (ensemble)", e["js_disagreement"]),
                        ("1-maxsoftmax (final)", 1 - e["final_conf"]),
                        ("entropy (final)", e["final_entropy"])]:
        fpr, tpr, _ = roc_curve(yerr, score)
        axes[0].plot(fpr, tpr, label=f"{name} ({auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(yerr, score)
        axes[1].plot(rc, pr, label=f"{name} ({auc(rc, pr):.3f})")
    axes[0].set_title(f"Error-detection ROC — {tag}")
    axes[1].set_title(f"Error-detection PR — {tag}")
    for ax in axes:
        ax.legend(fontsize=6)
    _save(fig, "04_error_detection_roc_pr", tag)


def fig_reliability(ext, tag):
    from src.diagnostics.calibration import reliability_curve
    rc = reliability_curve(ext["test"]["final_probs"], ext["test"]["y"])
    x = [r["bin_center"] for r in rc]
    yv = [r["acc"] if r["acc"] is not None else np.nan for r in rc]
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(x, yv, "o-", color="#4878CF")
    ax.set_title(f"Final-stack reliability — {tag}")
    _save(fig, "05_reliability", tag)


def fig_risk_coverage(rc_res, tag):
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for name, res in rc_res.items():
        covs = [r["coverage"] for r in res["rows"]]
        risks = [r["risk"] for r in res["rows"]]
        ax.plot(np.array(covs) * 100, risks, "o-", ms=4,
                label=name.replace("_", " "))
    ax.set_xlabel("coverage %"); ax.set_ylabel("risk")
    ax.set_title(f"Risk-coverage — {tag}"); ax.legend(fontsize=6)
    _save(fig, "06_risk_coverage", tag)


def fig_corruption(deg_res, tag):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))
    sigs = [("u_cs", "mean u (cs)"), ("branch_disagreement_js", "JS disagreement"),
            ("final_confidence", "final confidence")]
    for ax, (sig, lbl) in zip(axes, sigs):
        for kind, res in deg_res["kinds"].items():
            levels = sorted(res["levels"], key=float)
            x = [abs(1 - float(l)) if kind == "amplitude_scale" else float(l)
                 for l in levels]
            yv = [res["levels"][l].get(sig) for l in levels]
            if any(v is not None and np.isfinite(v) for v in yv):
                ax.plot(x, yv, "o-", ms=3, label=kind)
        ax.set_xlabel("severity"); ax.set_ylabel(lbl)
        ax.legend(fontsize=5)
    fig.suptitle(f"Corruption response — {tag}", y=1.02)
    fig.tight_layout()
    _save(fig, "07_corruption_response", tag)


def fig_alpha(alpha_res, tag):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for ax, b in zip(axes, ["lite", "rv", "cs", "cmr"]):
        rec = alpha_res.get("test", {}).get(b, {})
        names = [k for k in rec if isinstance(rec[k], dict) and "rho" in rec[k]]
        vals = [rec[k]["rho"] for k in names]
        axes_pos = np.arange(len(names))
        ax.bar(axes_pos, vals, color=COLORS[b])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(axes_pos)
        ax.set_xticklabels([n[:12] for n in names], rotation=30, ha="right",
                           fontsize=5)
        ax.set_title(f"alpha_{b}", fontsize=9)
    fig.suptitle(f"alpha vs independent input descriptors — {tag}", y=1.04)
    fig.tight_layout()
    _save(fig, "08_alpha_validation", tag)


def fig_faithfulness(faith_res, tag):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    sources = list(faith_res["by_source"])
    w = 0.35
    for j, src in enumerate(sources):
        r = faith_res["by_source"][src]["pdrop"]
        ax.bar(j + (w / 2 if j else -w / 2), r["targeted_mean"], w,
               label=f"{src} targeted", color="#4878CF")
        ax.bar(j - (w / 2 if j else -w / 2), r["random_mean"], w,
               label=f"{src} random", color="#bbbbbb")
    ax.set_xticks(range(len(sources))); ax.set_xticklabels(sources)
    ax.set_ylabel("true-class prob drop")
    ax.set_title(f"Faithfulness — {tag}"); ax.legend(fontsize=7)
    _save(fig, "09_faithfulness", tag)


def fig_benign(ben, tag):
    names = [k for k in ben if isinstance(ben[k], dict) and "final_agreement" in ben[k]]
    agree = [ben[k]["final_agreement"] for k in names]
    zc = [ben[k].get("z_cs_cosine", np.nan) for k in names]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.plot(names, agree, "o-", label="final pred agreement")
    ax.plot(names, zc, "s-", label="z_cs cosine")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title(f"Benign-transformation stability — {tag}")
    _save(fig, "10_benign_stability", tag)


def fig_complementarity(comp, tag):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    names = list(comp.get("per_branch_solo_mf1", {}))
    vals = [comp["per_branch_solo_mf1"][n] for n in names]
    ax.bar(names, vals, color=[COLORS[n] for n in names])
    ax.set_ylabel("solo test MF1")
    ax.set_title(f"Branch solo performance — {tag}")
    _save(fig, "11_branch_complementarity", tag)


def fig_baseline_comparison(bl, tag):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    names = list(bl)
    aucs = [bl[k]["auroc"] for k in names]
    colors = ["#D65F5F" if k.startswith(("branch", "js", "vote"))
              else "#4878CF" for k in names]
    ax.barh(range(len(names)), aucs, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("error-detection AUROC")
    ax.set_title(f"Diagnostic baselines (red=stack signals) — {tag}")
    ax.invert_yaxis()
    _save(fig, "12_baseline_comparison", tag)


def fig_cross_dataset(master_rows):
    dims = ["latent", "velocity", "intrinsic_u", "ensemble_disagreement",
            "calibration", "degradation", "alpha", "faithfulness"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for t in sorted(set(r["dataset"] for r in master_rows)):
        vals = [r.get("score_0to1", np.nan) for r in master_rows
                if r["dataset"] == t]
        if len(vals) == len(dims):
            ax.plot(range(len(dims)), vals, "o-", ms=4, label=t)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(dims, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("score")
    ax.legend(fontsize=7)
    ax.set_title("TURS-Stack cross-dataset diagnostic summary")
    _save(fig, "13_cross_dataset_summary", "ALL")
