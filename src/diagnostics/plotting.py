"""Publication figures (Phase 25). Deterministic; PNG + PDF per figure."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name, tag):
    out = os.path.join(C.FIG_DIR, tag)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_latent_pca(ext, tag):
    from sklearn.decomposition import PCA
    z = ext["test"]["z"]; y = ext["test"]["y"]
    if len(z) > 2500:
        idx = np.random.default_rng(0).permutation(len(z))[:2500]
        z, y = z[idx], y[idx]
    p = PCA(n_components=2, random_state=0).fit_transform(z)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for c in np.unique(y):
        m = y == c
        ax.scatter(p[m, 0], p[m, 1], s=4, alpha=0.5, label=f"class {c}")
    ax.legend(markerscale=2, fontsize=7)
    ax.set_title(f"TURS-Lite latent z (pooled) — {tag}")
    _save(fig, "01_latent_pca", tag)


def fig_latent_separability(latent_res, tag):
    ks = ["silhouette", "kmeans_ari", "kmeans_nmi"]
    vals = [latent_res["test"].get(k, np.nan) for k in ks]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(ks, vals, color="#4878CF")
    ax.set_title(f"Latent class structure (test) — {tag}")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "02_latent_separability", tag)


def fig_traces(ext, tag, n=4):
    e = ext["test"]
    L = e["s_t"].shape[1]
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.2 * n))
    rng = np.random.default_rng(0)
    for i in range(n):
        k = int(rng.integers(0, len(e["y"])))
        axes[i, 0].plot(e["s_t"][k]); axes[i, 0].set_ylabel("||v_t||")
        axes[i, 0].set_title(f"sample {k} y={e['y'][k]} pred={e['pred'][k]}", fontsize=8)
        axes[i, 1].plot(e["u_t"][k].mean(1), color="#D65F5F")
        axes[i, 1].set_ylabel("mean u_t")
    fig.suptitle(f"Temporal traces — {tag}", y=1.01)
    fig.tight_layout()
    _save(fig, "03_velocity_uncertainty_traces", tag)


def fig_unc_distributions(ext, tag):
    e = ext["test"]; u = e["u_mean"]; corr = e["correct"] == 1
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.hist(u[corr], bins=40, alpha=0.6, label="correct", density=True)
    ax.hist(u[~corr], bins=40, alpha=0.6, label="incorrect", density=True)
    ax.set_xlabel("mean u"); ax.legend()
    ax.set_title(f"Uncertainty: correct vs incorrect — {tag}")
    _save(fig, "04_uncertainty_distributions", tag)


def fig_error_detection(ext, tag):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    e = ext["test"]; yerr = (e["correct"] == 0).astype(int)
    if yerr.sum() == 0 or yerr.sum() == len(yerr):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for name, score in [("turs_u_mean", e["u_mean"]),
                        ("1-maxsoftmax", 1 - e["confidence"]),
                        ("entropy", e["entropy"])]:
        fpr, tpr, _ = roc_curve(yerr, score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(yerr, score)
        axes[1].plot(rc, pr, label=f"{name} (AP={auc(rc, pr):.3f})")
    axes[0].set_title(f"Error-detection ROC — {tag}")
    axes[1].set_title(f"Error-detection PR — {tag}")
    for ax in axes:
        ax.legend(fontsize=7)
    _save(fig, "05_error_detection_roc_pr", tag)


def fig_reliability(ext, tag):
    from .calibration import reliability_curve
    rc = reliability_curve(ext["test"]["probs"], ext["test"]["y"])
    x = [r["bin_center"] for r in rc]
    yv = [r["acc"] if r["acc"] is not None else np.nan for r in rc]
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(x, yv, "o-", color="#4878CF")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability — {tag}")
    _save(fig, "06_reliability", tag)


def fig_risk_coverage(risk_comp, tag):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for name, res in risk_comp.items():
        covs = [r["coverage"] for r in res["rows"]]
        risks = [r["risk"] for r in res["rows"]]
        ax.plot(np.array(covs) * 100, risks, "o-", label=name, ms=4)
    ax.set_xlabel("coverage %"); ax.set_ylabel("risk (error rate)")
    ax.set_title(f"Risk-coverage — {tag}"); ax.legend(fontsize=7)
    _save(fig, "07_risk_coverage", tag)


def fig_corruption(corr_res, tag):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for kind, res in corr_res["kinds"].items():
        levels = sorted(res["levels"], key=float)
        x = [abs(1 - float(l)) if kind == "amplitude_scale" else float(l)
             for l in levels]
        axes[0].plot(x, [res["levels"][l]["mean_u"] for l in levels], "o-", ms=4,
                     label=kind)
        axes[1].plot(x, [res["levels"][l]["error_rate"] for l in levels], "o-",
                     ms=4, label=kind)
    axes[0].set_ylabel("mean u"); axes[0].set_title(f"Severity vs u — {tag}")
    axes[1].set_ylabel("error rate"); axes[1].set_title(f"Severity vs error — {tag}")
    for ax in axes:
        ax.legend(fontsize=6)
        ax.set_xlabel("degradation severity")
    _save(fig, "08_corruption_response", tag)


def fig_alpha(alpha_res, tag):
    r = alpha_res["test"]
    names = [k for k in r if isinstance(r[k], dict) and "rho" in r[k]]
    vals = [r[k]["rho"] for k in names]
    los = [r[k]["rho"] - r[k]["ci_lo"] for k in names]
    his = [r[k]["ci_hi"] - r[k]["rho"] for k in names]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.errorbar(range(len(names)), vals, yerr=[los, his], fmt="o", capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25,
                                                         ha="right", fontsize=7)
    ax.set_ylabel("Spearman rho (alpha vs descriptor)")
    ax.set_title(f"alpha vs independent input descriptors — {tag}")
    _save(fig, "09_alpha_validation", tag)


def fig_faithfulness(faith_res, tag):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sources = list(faith_res["by_source"])
    w = 0.35
    for j, src in enumerate(sources):
        r = faith_res["by_source"][src]
        ax.bar(j + (w / 2 if j else -w / 2), r["pdrop"]["targeted_mean"], w,
               label=f"{src} targeted", color="#4878CF")
        ax.bar(j - (w / 2 if j else -w / 2), r["pdrop"]["random_mean"], w,
               label=f"{src} random", color="#bbbbbb")
    ax.set_xticks(range(len(sources))); ax.set_xticklabels(sources)
    ax.set_ylabel("true-class prob drop")
    ax.set_title(f"Faithfulness: targeted vs random — {tag}")
    ax.legend(fontsize=7)
    _save(fig, "10_faithfulness", tag)


def fig_benign(benign_res, tag):
    names = list(benign_res)
    agree = [benign_res[k]["pred_agreement"] for k in names]
    zc = [benign_res[k]["z_cosine"] for k in names]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(names, agree, "o-", label="prediction agreement")
    ax.plot(names, zc, "s-", label="z cosine")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
    ax.set_title(f"Benign-transformation stability — {tag}")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "11_benign_stability", tag)


def fig_counterfactual(cf_res, tag):
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.bar(["high-s_t edit", "random edit"],
           [cf_res["probdrop_target"], cf_res["probdrop_random"]],
           color=["#D65F5F", "#bbbbbb"])
    ax.set_ylabel("true-class prob drop")
    ax.set_title(f"Counterfactual response — {tag}")
    _save(fig, "12_counterfactual", tag)


def fig_baseline_comparison(baseline_res, tag):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    names = list(baseline_res)
    aucs = [baseline_res[k]["auroc"] for k in names]
    los = [baseline_res[k]["auroc"] - baseline_res[k]["ci"][0] for k in names]
    his = [baseline_res[k]["ci"][1] - baseline_res[k]["auroc"] for k in names]
    colors = ["#D65F5F" if n.startswith("turs") else "#4878CF" for n in names]
    ax.barh(range(len(names)), aucs, xerr=[los, his], color=colors, capsize=3)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("error-detection AUROC (95% CI)")
    ax.set_title(f"Uncertainty baselines — {tag}")
    ax.invert_yaxis()
    _save(fig, "13_baseline_comparison", tag)


def fig_cross_dataset(master_rows):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    tags = sorted(set(r["dataset"] for r in master_rows))
    dims = ["latent", "velocity", "uncertainty", "calibration", "faithfulness",
            "robustness"]
    scores = {t: [] for t in tags}
    for r in master_rows:
        scores[r["dataset"]].append(r.get("score_0to1", np.nan))
    x = np.arange(len(dims))
    for t in tags:
        vals = [r.get("score_0to1", np.nan) for r in master_rows if r["dataset"] == t]
        if len(vals) == len(dims):
            ax.plot(x, vals, "o-", label=t, ms=4)
    ax.set_xticks(x); ax.set_xticklabels(dims, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("normalized diagnostic score")
    ax.legend(fontsize=7)
    ax.set_title("Cross-dataset diagnostic summary")
    _save(fig, "14_cross_dataset_summary", "ALL")
