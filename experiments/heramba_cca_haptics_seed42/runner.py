"""HERAMBA-CCA experiment: Haptics, seed 42, single controlled run.

Pipeline (per the CCA + label-complementarity specification):
  1. Frozen canonical banks G/H (reused, identity-verified; exact R2
     representation: [G||H] -> R2 Ridge reproduces 0.5500).
  2. CCA fitted on TRAIN ONLY (label-free): pre-declared numerical-rank
     PCA reduction (energy 1-1e-10) -> whitened cross-covariance SVD.
  3. Shared subspace: predeclared tau = 0.5 canonical-correlation rule.
  4. H_unique: frozen OLS residualization of H on the shared canonical
     scores, applied unchanged to val/test.
  5. Effective-rank + variance-partition diagnostics.
  6. Label-aware complementarity: paired 5-fold CV macro-F1 increment
     ([G,Hu] vs G) on TRAIN + 2000-permutation column-permutation null
     (seed 42042). Dev-only; test untouched.
  7. Final comparison (single test evaluation each, canonical final fit
     train+val): MiniROCKET M0 vs R2/R5 vs CCA-HERAMBA.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from models.heramba_cca.cca import HerambaCCAProjection, effective_rank  # noqa
from models.heramba_cca.complementarity import (  # noqa: E402
    permutation_complementarity_test, paired_cv_increment)
from models.heramba_cca.diagnostics import rank_report, variance_partition  # noqa
from models.heramba_cca.model import (ALPHAS, HerambaCCAModel)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CACHE = r"C:/temp/results"
OUT = os.path.join(ROOT, "results", "heramba_cca", "haptics_seed42")
ALPHAS_ARR = ALPHAS


def log(m):
    print(m, flush=True)


def load_frozen_banks():
    """Frozen canonical G/H banks + labels (identity-verified artifacts)."""
    G_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_G_trva.npy"))
    H_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_H_trva.npy"))
    G_te = np.load(os.path.join(CACHE, "haptics_inference_banks_G_te.npy"))
    H_te = np.load(os.path.join(CACHE, "haptics_inference_banks_H_te.npy"))
    ctx = json.load(open(os.path.join(CACHE, "haptics_inference_ctx.json"))) \
        if os.path.exists(os.path.join(CACHE, "haptics_inference_ctx.json")) \
        else {}
    # labels: canonical Haptics split 132/23/308 -> dev=155, test=308
    from experiments.external_stack_generalization.data import load_dataset
    d = load_dataset("Haptics")
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]
    assert len(G_trva) == len(ytr) + len(yva) == 155
    assert len(G_te) == len(yte) == 308
    y_dev = np.concatenate([ytr, yva])
    return (G_trva.astype(np.float64), H_trva.astype(np.float64),
            G_te.astype(np.float64), H_te.astype(np.float64),
            ytr, yva, y_dev, yte, ctx)


def ridge_eval(Xtrva, ytrva, Xte, yte, Xva=None, yva=None):
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score)
    clf = RidgeClassifierCV(alphas=ALPHAS_ARR)
    clf.fit(Xtrva, ytrva)
    pred = clf.predict(Xte).astype(np.int64)
    res = {
        "macro_f1": round(float(f1_score(yte, pred, average="macro",
                                         zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "selected_alpha": float(clf.alpha_),
        "per_class_f1": [round(float(v), 4) for v in f1_score(
            yte, pred, average=None, zero_division=0,
            labels=list(range(5)))],
        "confusion_matrix": confusion_matrix(
            yte, pred, labels=list(range(5))).tolist(),
        "n_features": int(Xtrva.shape[1]),
    }
    if Xva is not None:
        res["val_macro_f1"] = round(float(f1_score(
            yva, clf.predict(Xva), average="macro", zero_division=0)), 4)
    return res, pred


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    log("=== HERAMBA-CCA — Haptics seed 42 ===")

    # ------------------------------------------------------------------
    # 1. frozen banks
    # ------------------------------------------------------------------
    G_trva, H_trva, G_te, H_te, ytr, yva, y_dev, yte, ctx = \
        load_frozen_banks()
    n_tr = len(ytr)
    log(f"banks: G{G_trva.shape} H{H_trva.shape} | dev={len(y_dev)} "
        f"test={len(yte)}")
    y_trva = y_dev

    # identity gate: [G||H] with canonical Ridge must reproduce R2 0.5500
    r2_res, _ = ridge_eval(np.hstack([G_trva, H_trva]), y_trva,
                           np.hstack([G_te, H_te]), yte)
    assert r2_res["macro_f1"] == 0.55, f"R2 identity gate failed: {r2_res}"
    log(f"[identity gate] R2 (G+H) test = {r2_res['macro_f1']} == 0.5500 OK")

    # ------------------------------------------------------------------
    # 2-4. CCA on TRAIN ONLY -> shared subspace -> H_unique
    # ------------------------------------------------------------------
    # NOTE on the predeclared reduction rule: the numerical-rank rule
    # (energy 1-1e-10) keeps k=131 of 132 possible components for both
    # banks (Haptics dev is small and close to low-rank), i.e. CCA runs
    # at full statistical rank with NO information discarded beyond the
    # mean/whitening.  This is the most conservative choice: it cannot
    # manufacture or destroy complementarity by dimensionality choice.
    G_tr, H_tr = G_trva[:n_tr], H_trva[:n_tr]
    proj = HerambaCCAProjection(tau_shared=0.5).fit(G_tr, H_tr)
    summ = proj.summary()
    log(f"CCA: k_G={summ['G_reduced_dim']} k_H={summ['H_reduced_dim']} "
        f"n_comp={summ['n_cca_components']} "
        f"shared={summ['n_shared_components']} "
        f"(tau=0.5, orth max|r|={summ['orth_max_abs_corr_Hunique_vs_shared']})")
    with open(os.path.join(OUT, "cca_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    with open(os.path.join(OUT, "canonical_correlations.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "canonical_correlation", "shared"])
        for j, r in enumerate(summ["canonical_correlations"]):
            w.writerow([j + 1, r, j < summ["n_shared_components"]])

    # ------------------------------------------------------------------
    # 5. diagnostics: ranks + variance partition (train statistics)
    # ------------------------------------------------------------------
    Hu_tr = proj.transform_H_unique(H_tr)
    Sc_tr = proj.cca_.project_H(proj.pca_H_.transform(H_tr))[:, :proj.k_shared_]
    H_shared_hat_tr = ((Sc_tr - proj._sc_mean_) @ proj._red_coef_
                       + proj.pca_H_.transform(H_tr).mean(0))
    vp = variance_partition(proj.pca_H_.transform(H_tr), Hu_tr,
                            H_shared_hat_tr)
    rr = {
        "H": rank_report(proj.pca_H_.transform(H_tr), "H_reduced"),
        "H_unique": rank_report(Hu_tr, "H_unique_reduced"),
        "G": rank_report(proj.pca_G_.transform(G_tr), "G_reduced"),
        "variance_partition": vp,
    }
    with open(os.path.join(OUT, "diagnostics.json"), "w") as f:
        json.dump(rr, f, indent=1)
    with open(os.path.join(OUT, "effective_rank.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["representation", "dim", "numerical_rank",
                    "effective_rank"])
        for k in ["G", "H", "H_unique"]:
            w.writerow([rr[k]["name"], rr[k]["dim"], rr[k]["numerical_rank"],
                        rr[k]["effective_rank"]])
    log(f"ranks: G eff={rr['G']['effective_rank']} "
        f"H eff={rr['H']['effective_rank']} "
        f"Hu eff={rr['H_unique']['effective_rank']} | "
        f"var shared={vp['fraction_H_variance_shared']} "
        f"unique={vp['fraction_H_variance_unique']}")

    # ------------------------------------------------------------------
    # 6. label-aware complementarity (dev-only)
    # ------------------------------------------------------------------
    Hu_trva = proj.transform_H_unique(H_trva)
    log("paired CV increment (dev-only)...")
    inc = paired_cv_increment(G_trva, Hu_trva, y_trva)
    log(f"  increment = {inc['observed_increment']:+.4f} "
        f"(base {inc['cv_base_mean']:.4f} -> aug {inc['cv_aug_mean']:.4f})")
    log("permutation test (2000 perms, seed 42042)...")
    perm = permutation_complementarity_test(G_trva, Hu_trva, y_trva)
    log(f"  empirical p = {perm['empirical_p_one_sided']}")
    comp = {
        "cv_increment": inc,
        "permutation": {k: v for k, v in perm.items()
                        if k != "null_distribution"},
    }
    with open(os.path.join(OUT, "complementarity.json"), "w") as f:
        json.dump(comp, f, indent=1)
    with open(os.path.join(OUT, "complementarity_evidence.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Quantity", "Value"])
        rows = [
            ("G dimension", summ["G_full_dim"]),
            ("H dimension", summ["H_full_dim"]),
            ("G reduced dim (train numerical rank)", summ["G_reduced_dim"]),
            ("H reduced dim (train numerical rank)", summ["H_reduced_dim"]),
            ("CCA valid dimensions", summ["n_cca_components"]),
            ("shared dimensions (tau=0.5)", summ["n_shared_components"]),
            ("effective rank H", rr["H"]["effective_rank"]),
            ("effective rank H_unique", rr["H_unique"]["effective_rank"]),
            ("fraction H variance shared", vp["fraction_H_variance_shared"]),
            ("fraction H variance unique", vp["fraction_H_variance_unique"]),
            ("CV macro-F1 G only", inc["cv_base_mean"]),
            ("CV macro-F1 G+H_unique", inc["cv_aug_mean"]),
            ("incremental predictive effect",
             inc["observed_increment"]),
            ("permutation null mean", perm["null_mean"]),
            ("empirical p (one-sided)",
             perm["empirical_p_one_sided"]),
            ("Cohen's d (fold deltas)", inc["cohens_d"]),
        ]
        w.writerows(rows)

    # ------------------------------------------------------------------
    # 7. final comparison (single test evaluation each)
    # ------------------------------------------------------------------
    log("final comparison (single test evaluations)...")
    m0_res, m0_pred = ridge_eval(G_trva, y_trva, G_te, yte)
    log(f"  MiniROCKET  : {m0_res['macro_f1']} (alpha={m0_res['selected_alpha']:.3f})")
    log(f"  R2 (G+H)    : {r2_res['macro_f1']} (alpha={r2_res['selected_alpha']:.3f})")

    model = HerambaCCAModel(tau_shared=0.5).fit_representation(G_tr, H_tr)
    X_dev = model.transform(G_trva, H_trva)
    X_te = model.transform(G_te, H_te)
    model.fit_classifier(X_dev, y_trva)
    pred_cca = model.predict(X_te).astype(np.int64)
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score)
    cca_res = {
        "macro_f1": round(float(f1_score(yte, pred_cca, average="macro",
                                         zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(yte, pred_cca)), 4),
        "selected_alpha": model.alpha_,
        "per_class_f1": [round(float(v), 4) for v in f1_score(
            yte, pred_cca, average=None, zero_division=0,
            labels=list(range(5)))],
        "confusion_matrix": confusion_matrix(
            yte, pred_cca, labels=list(range(5))).tolist(),
        "n_features": int(X_dev.shape[1]),
        "H_unique_dim": int(X_dev.shape[1] - G_trva.shape[1]),
        "n_shared_components": proj.k_shared_,
    }
    log(f"  CCA-HERAMBA : {cca_res['macro_f1']} (alpha={model.alpha_:.3f}, "
        f"feat={cca_res['n_features']})")
    np.save(os.path.join(OUT, "predictions_m0.npy"), m0_pred)
    np.save(os.path.join(OUT, "predictions_cca.npy"), pred_cca)

    with open(os.path.join(OUT, "final_comparison.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Representation", "Features", "Macro-F1", "Accuracy",
                    "Selected alpha", "Per-class F1"])
        w.writerow(["MiniROCKET", m0_res["n_features"], m0_res["macro_f1"],
                    m0_res["accuracy"], m0_res["selected_alpha"],
                    m0_res["per_class_f1"]])
        w.writerow(["R2/R5/HERAMBA (G+H)", r2_res["n_features"],
                    r2_res["macro_f1"], r2_res["accuracy"],
                    r2_res["selected_alpha"], r2_res["per_class_f1"]])
        w.writerow(["CCA-HERAMBA [G||H_unique]", cca_res["n_features"],
                    cca_res["macro_f1"], cca_res["accuracy"],
                    cca_res["selected_alpha"], cca_res["per_class_f1"]])
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({
            "seed": 42, "dataset": "Haptics",
            "minirocket_m0": m0_res, "r2_gh": r2_res,
            "cca_heramba": cca_res, "runtime_s": round(time.time() - t0, 1),
            "final_test_evaluations": {"MiniROCKET": 1, "R2": 1,
                                       "CCA-HERAMBA": 1},
        }, f, indent=1)

    # ------------------------------------------------------------------
    # 8. figures
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = np.array(summ["canonical_correlations"])
    k = summ["n_shared_components"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(rho) + 1), rho, "o-", ms=4)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.axvspan(0.5, k + 0.5, alpha=.15, color="tab:red",
               label=f"shared (tau=0.5): {k} components")
    ax.set_xlabel("CCA component")
    ax.set_ylabel("canonical correlation (train)")
    ax.set_title("CCA spectrum: G (MiniROCKET) vs H (HERAMBA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "cca_spectrum.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "figures", "cca_spectrum.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cum = np.cumsum(rho[:min(40, len(rho))] ** 2)
    ax.plot(np.arange(1, len(cum) + 1), cum / rho.sum() ** 2 * k, "s-",
            ms=3, label="cumulative shared variance (canonical)")
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative sum of squared correlations")
    ax.set_title("Cumulative shared-correlation diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures",
                             "cumulative_shared_variance.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "figures",
                             "cumulative_shared_variance.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, lbl, c in [("H", "H (reduced)", "tab:blue"),
                        ("H_unique", "H_unique", "tab:orange")]:
        ev = np.array(rr[key]["explained_variance_ratio_top50"])
        cum = np.cumsum(ev)
        ax.plot(np.arange(1, len(cum) + 1), cum, label=f"{lbl} "
                f"(eff rank {rr[key]['effective_rank']})", color=c)
    ax.set_xlabel("component")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("Effective-rank spectra: H vs H_unique")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "effective_rank.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "figures", "effective_rank.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["shared", "unique"], [vp["fraction_H_variance_shared"],
                                  vp["fraction_H_variance_unique"]],
           color=["tab:red", "tab:blue"])
    ax.set_ylabel("fraction of H variance (train)")
    ax.set_title("Variance partition of H")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures",
                             "H_variance_partition.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "figures",
                             "H_variance_partition.pdf"))
    plt.close(fig)

    log(f"complete in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
