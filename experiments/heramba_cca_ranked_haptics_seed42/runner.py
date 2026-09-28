"""RANK-CONTROLLED CCA HERAMBA — Haptics, seed 42 (new experiment).

Fixes the small-n/high-d full-rank CCA degeneracy of the previous
experiment (results/heramba_cca/haptics_seed42 — preserved unchanged)
with a PREDECLARED, label-free, TRAIN-ONLY rank rule:

    rank_rule = "minimum_train_PCA_components_for_95pct_variance"
    (retain the minimum number of TRAIN-PCA components (per bank,
    after z-scoring with train statistics) whose cumulative explained
    variance >= 0.95; the 95% threshold is a fixed methodological
    constant, never tuned on validation or test)

Pipeline (all fits on TRAIN = first 132 dev rows only):
    G_train -> train z-score -> PCA(95%) -> G_low
    H_train -> train z-score -> PCA(95%) -> H_low
    CCA(G_low, H_low) via whitened cross-covariance SVD
    shared := canonical components with rho >= 0.5 (tau predeclared)
    H_unique = OLS residual of H_low on the shared canonical scores
             (frozen; applied unchanged to val/test)
    final representation = [G || H_unique]  (canonical Ridge, train+val
    fit, ONE test evaluation)  — no rho, no mixture, no search.

Label-complementarity: paired 5-fold CV macro-F1 increment on dev +
plus-one permutation test (S = 1000, seed 42042). Test untouched.
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

from models.heramba_cca.cca import (CCAFit, effective_rank,  # noqa: E402
                                    numerical_rank)
from models.heramba_cca.complementarity import (  # noqa: E402
    plus_one_permutation_test)
from models.heramba_cca.model import ALPHAS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CACHE = r"C:/temp/results"
OUT = os.path.join(ROOT, "results", "heramba_cca_ranked", "haptics_seed42")

VAR_THRESHOLD = 0.95      # predeclared rank-rule constant
TAU_SHARED = 0.5          # predeclared shared-subspace threshold
S_PERM = 1000             # predeclared permutation count
PERM_SEED = 42042
SEED = 42


def log(m):
    print(m, flush=True)


# ----------------------------------------------------------------------
# frozen train-only PCA with the 95% rank rule
# ----------------------------------------------------------------------
class RankPCA:
    """Train-only z-score + PCA retaining min components for >=95% var."""

    def __init__(self, var_threshold: float = VAR_THRESHOLD):
        self.var_threshold = var_threshold

    def fit(self, X: np.ndarray) -> "RankPCA":
        X = np.asarray(X, dtype=np.float64)
        self.mu_ = X.mean(0)
        sd = X.std(0)
        self.ok_ = np.isfinite(sd) & (sd > 1e-12)
        Z = (X - self.mu_) / np.where(self.ok_, sd, 1.0)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        s2 = S ** 2
        tot = s2.sum()
        self.explained_variance_ratio_ = s2 / max(tot, 1e-300)
        cum = np.cumsum(self.explained_variance_ratio_)
        self.cumulative_variance_ = cum
        k = int(np.searchsorted(cum, self.var_threshold) + 1)
        self.k_ = max(1, min(k, len(S)))
        self.comp_ = Vt[:self.k_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mu_) \
            / np.where(self.ok_, np.sqrt((self.comp_.shape[0],)), 1.0)
        return None  # never used; see transform_low

    def transform_low(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mu_) \
            / np.where(self.ok_, 1.0, 1.0)
        Z[:, ~self.ok_] = 0.0
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return Z @ self.comp_.T


# ----------------------------------------------------------------------
# shared-subspace residualizer in the LOW-RANK H space
# ----------------------------------------------------------------------
class SharedResidualLow:
    """H_unique = H_low - (shared canonical scores -> OLS -> H_low)."""

    def fit(self, H_low: np.ndarray, shared_scores: np.ndarray) \
            -> "SharedResidualLow":
        H = np.asarray(H_low, dtype=np.float64)
        S = np.asarray(shared_scores, dtype=np.float64)
        self.mu_ = S.mean(0)
        Sc = S - self.mu_
        self.coef_, *_ = np.linalg.lstsq(Sc, H, rcond=None)
        self.H_mean_ = H.mean(0)
        # orthogonality verification on TRAIN (ignore zero-variance dirs)
        resid = H - Sc @ self.coef_
        rc = resid - resid.mean(0)
        ds = np.sqrt((Sc ** 2).sum(0))
        dr = np.sqrt((rc ** 2).sum(0))
        ir = np.where(dr > 1e-8)[0]
        isv = np.where(ds > 1e-8)[0]
        if ir.size and isv.size:
            C = (rc[:, ir].T @ Sc[:, isv]) / np.outer(dr[ir], ds[isv])
        else:
            C = np.zeros((1, 1))
        self.orth_max_abs_corr_ = float(np.nanmax(np.abs(C)))
        self.orth_mean_abs_corr_ = float(np.nanmean(np.abs(C)))
        self.resid_cov_norm_ = float(np.linalg.norm(
            np.cov(resid.T) if resid.shape[0] > 1 else resid))
        return self

    def transform(self, H_low: np.ndarray, shared_scores: np.ndarray) \
            -> np.ndarray:
        S = np.asarray(shared_scores, dtype=np.float64) - self.mu_
        return (np.asarray(H_low, dtype=np.float64)
                - (S @ self.coef_ + self.H_mean_))


# ----------------------------------------------------------------------
def ridge_eval(Xtrva, ytrva, Xte, yte):
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score)
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(Xtrva, ytrva)
    pred = clf.predict(Xte).astype(np.int64)
    return {
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
    }, pred


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    log("=== RANK-CONTROLLED CCA HERAMBA — Haptics seed 42 ===")

    # ---- frozen canonical banks (read-only reuse) --------------------
    G_trva = np.load(os.path.join(CACHE,
                                  "haptics_inference_banks_G_trva.npy")
                     ).astype(np.float64)
    H_trva = np.load(os.path.join(CACHE,
                                  "haptics_inference_banks_H_trva.npy")
                     ).astype(np.float64)
    G_te = np.load(os.path.join(CACHE, "haptics_inference_banks_G_te.npy")
                   ).astype(np.float64)
    H_te = np.load(os.path.join(CACHE, "haptics_inference_banks_H_te.npy")
                   ).astype(np.float64)
    from experiments.external_stack_generalization.data import load_dataset
    d = load_dataset("Haptics")
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]
    assert len(G_trva) == 155 and len(G_te) == 308
    n_tr = len(ytr)
    y_dev = np.concatenate([ytr, yva])

    # identity gate: [G||H] reproduces canonical R2 exactly
    r2_res, _ = ridge_eval(np.hstack([G_trva, H_trva]), y_dev,
                           np.hstack([G_te, H_te]), yte)
    assert r2_res["macro_f1"] == 0.55, f"R2 identity gate: {r2_res}"
    log(f"[identity gate] R2 (G+H) = {r2_res['macro_f1']} OK")
    m0_res, m0_pred = ridge_eval(G_trva, y_dev, G_te, yte)
    log(f"[M0 G-only] = {m0_res['macro_f1']}")

    # ---- Phase 3: train-only rank-controlled PCA ----------------------
    G_tr, H_tr = G_trva[:n_tr], H_trva[:n_tr]
    pcaG = RankPCA().fit(G_tr)
    pcaH = RankPCA().fit(H_tr)
    G_low_tr = pcaG.transform_low(G_tr)
    H_low_tr = pcaH.transform_low(H_tr)
    log(f"rank rule (>= {VAR_THRESHOLD:.0%} train variance): "
        f"g_rank={pcaG.k_}, h_rank={pcaH.k_} "
        f"(of {n_tr} max); energy kept "
        f"G={pcaG.cumulative_variance_[pcaG.k_-1]:.4f} "
        f"H={pcaH.cumulative_variance_[pcaH.k_-1]:.4f}")

    # ---- Phase 4: CCA on the low-rank representations -----------------
    cca = CCAFit().fit(G_low_tr, H_low_tr, tau_shared=TAU_SHARED)
    rho = np.asarray(cca.rho_)
    k_shared = cca.k_shared_
    log(f"CCA: {len(rho)} components; rho max={rho.max():.4f} "
        f"mean={rho.mean():.4f}; shared(rho>={TAU_SHARED})={k_shared}")
    assert not np.allclose(rho, 1.0), \
        "rank-controlled CCA still degenerate (all rho=1) — STOP"

    # ---- Phases 5-6: shared subspace + H_unique (train) ---------------
    shared_scores_tr = cca.project_H(H_low_tr)[:, :k_shared]
    resid = SharedResidualLow().fit(H_low_tr, shared_scores_tr)
    Hu_tr = resid.transform(H_low_tr, shared_scores_tr)
    log(f"orthogonality (train): max|r|={resid.orth_max_abs_corr_:.2e} "
        f"mean|r|={resid.orth_mean_abs_corr_:.2e}")

    # ---- Phase 7: frozen transforms -> val/test -----------------------
    G_low_trva = pcaG.transform_low(G_trva)
    H_low_trva = pcaH.transform_low(H_trva)
    shared_scores_trva = cca.project_H(H_low_trva)[:, :k_shared]
    Hu_trva = resid.transform(H_low_trva, shared_scores_trva)
    G_low_te = pcaG.transform_low(G_te)
    H_low_te = pcaH.transform_low(H_te)
    shared_scores_te = cca.project_H(H_low_te)[:, :k_shared]
    Hu_te = resid.transform(H_low_te, shared_scores_te)
    assert np.isfinite(Hu_trva).all() and np.isfinite(Hu_te).all()

    # ---- Phase 8: diagnostics -----------------------------------------
    vp_tot = float(((H_low_tr - H_low_tr.mean(0)) ** 2).sum())
    vp_uni = float(((Hu_tr - Hu_tr.mean(0)) ** 2).sum())
    frac_unique = vp_uni / max(vp_tot, 1e-300)
    vp = {"fraction_H_variance_shared": round(1.0 - frac_unique, 6),
          "fraction_H_variance_unique": round(frac_unique, 6),
          "H_low_total_var": vp_tot, "H_unique_var": vp_uni}
    ranks = {
        "H_low": {"numerical_rank": numerical_rank(H_low_tr),
                  "effective_rank": round(effective_rank(H_low_tr), 2)},
        "H_unique": {"numerical_rank": numerical_rank(Hu_tr),
                     "effective_rank": round(effective_rank(Hu_tr), 2)},
        "G_low": {"numerical_rank": numerical_rank(G_low_tr),
                  "effective_rank": round(effective_rank(G_low_tr), 2)},
    }
    log(f"ranks: G_low={ranks['G_low']} H_low={ranks['H_low']} "
        f"H_unique={ranks['H_unique']}")
    log(f"variance partition: shared={vp['fraction_H_variance_shared']} "
        f"unique={vp['fraction_H_variance_unique']}")

    # ---- Phase 9-10: label complementarity (dev-only) ------------------
    log("paired CV increment + plus-one permutation (S=1000)...")
    perm = plus_one_permutation_test(G_trva, Hu_trva, y_dev,
                                     n_perm=S_PERM, seed=PERM_SEED)
    inc = perm["base"]
    log(f"  increment={inc['observed_increment']:+.4f} "
        f"(base {inc['cv_base_mean']:.4f} -> aug {inc['cv_aug_mean']:.4f}) "
        f"p={perm['empirical_p_one_sided']} d={inc['cohens_d']}")

    # ---- Phase 12-13: final comparison (one test eval each) ------------
    X_dev = np.hstack([G_trva, Hu_trva])
    X_te = np.hstack([G_te, Hu_te])
    cca_res, cca_pred = ridge_eval(X_dev, y_dev, X_te, yte)
    log(f"[RankCCA-HERAMBA] = {cca_res['macro_f1']} "
        f"(feat={cca_res['n_features']}, "
        f"alpha={cca_res['selected_alpha']:.3f})")
    np.save(os.path.join(OUT, "predictions_m0.npy"), m0_pred)
    np.save(os.path.join(OUT, "predictions_rankcca.npy"), cca_pred)

    # ---- artifacts ------------------------------------------------------
    with open(os.path.join(OUT, "cca_ranked_summary.json"), "w") as f:
        json.dump({
            "rank_rule": "minimum_train_PCA_components_for_95pct_variance",
            "variance_threshold": VAR_THRESHOLD,
            "tau_shared": TAU_SHARED,
            "g_rank": pcaG.k_, "h_rank": pcaH.k_,
            "g_energy_kept": float(pcaG.cumulative_variance_[pcaG.k_ - 1]),
            "h_energy_kept": float(pcaH.cumulative_variance_[pcaH.k_ - 1]),
            "cumulative_variance_G": [
                round(float(v), 6) for v in
                pcaG.cumulative_variance_[:pcaG.k_]],
            "cumulative_variance_H": [
                round(float(v), 6) for v in
                pcaH.cumulative_variance_[:pcaH.k_]],
            "n_cca_components": int(len(rho)),
            "canonical_correlations": [round(float(r), 6) for r in rho],
            "shared_components": int(k_shared),
            "rho_max": round(float(rho.max()), 6),
            "rho_mean": round(float(rho.mean()), 6),
            "orth_max_abs_corr": resid.orth_max_abs_corr_,
            "orth_mean_abs_corr": resid.orth_mean_abs_corr_,
            "residual_cov_norm": resid.resid_cov_norm_,
            "ranks": ranks, "variance_partition": vp,
        }, f, indent=1)
    with open(os.path.join(OUT, "canonical_correlations.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "canonical_correlation", "shared"])
        for j, r in enumerate(rho):
            w.writerow([j + 1, round(float(r), 6), j < k_shared])
    with open(os.path.join(OUT, "final_comparison.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Representation", "Features", "Macro-F1", "Accuracy",
                    "Selected alpha", "Per-class F1"])
        w.writerow(["MiniROCKET (G only)", m0_res["n_features"],
                    m0_res["macro_f1"], m0_res["accuracy"],
                    m0_res["selected_alpha"], m0_res["per_class_f1"]])
        w.writerow(["R2/R5/HERAMBA [G||H]", r2_res["n_features"],
                    r2_res["macro_f1"], r2_res["accuracy"],
                    r2_res["selected_alpha"], r2_res["per_class_f1"]])
        w.writerow(["RankCCA-HERAMBA [G||H_unique]",
                    cca_res["n_features"], cca_res["macro_f1"],
                    cca_res["accuracy"], cca_res["selected_alpha"],
                    cca_res["per_class_f1"]])
    with open(os.path.join(OUT, "complementarity.json"), "w") as f:
        json.dump({k: v for k, v in perm.items() if k != "base"},
                  f, indent=1)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"seed": 42, "dataset": "Haptics",
                   "minirocket": m0_res, "r2_gh": r2_res,
                   "rankcca_heramba": cca_res,
                   "runtime_s": round(time.time() - t0, 1),
                   "final_test_evaluations": 1}, f, indent=1)
    with open(os.path.join(OUT, "effective_rank.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["representation", "dim", "numerical_rank",
                    "effective_rank"])
        for name, dim, r in [("H_low", pcaH.k_, ranks["H_low"]),
                             ("H_unique", pcaH.k_, ranks["H_unique"]),
                             ("G_low", pcaG.k_, ranks["G_low"])]:
            w.writerow([name, dim, r["numerical_rank"], r["effective_rank"]])

    # ---- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(rho) + 1), rho, "o-", ms=4)
    ax.axhline(TAU_SHARED, ls="--", c="gray", lw=1)
    if k_shared:
        ax.axvspan(0.5, k_shared + 0.5, alpha=.15, color="tab:red",
                   label=f"shared (tau={TAU_SHARED}): {k_shared}")
    ax.set_xlabel("CCA component (rank-controlled)")
    ax.set_ylabel("canonical correlation (train)")
    ax.set_title(f"Rank-controlled CCA spectrum "
                 f"(g_rank={pcaG.k_}, h_rank={pcaH.k_})")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"cca_ranked_spectrum.{ext}"))
    plt.close(fig)

    sH = np.asarray(pcaH.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(sH[:60]) + 1),
            np.cumsum(sH[:60]), label="H train-PCA cumulative variance")
    ax.axhline(VAR_THRESHOLD, ls="--", c="gray", lw=1)
    ax.axvline(pcaH.k_, ls=":", c="tab:red",
               label=f"retained rank h={pcaH.k_}")
    sU = np.linalg.svd(Hu_tr - Hu_tr.mean(0), compute_uv=False) ** 2
    sU = sU / max(sU.sum(), 1e-300)
    ax.plot(np.arange(1, len(sU[:60]) + 1), np.cumsum(sU[:60]),
            label="H_unique cumulative variance", color="tab:orange")
    ax.set_xlabel("component")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("H vs H_unique spectra (train)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"h_unique_spectrum.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["shared", "unique"], [vp["fraction_H_variance_shared"],
                                  vp["fraction_H_variance_unique"]],
           color=["tab:red", "tab:blue"])
    ax.set_ylabel("fraction of H_low variance (train)")
    ax.set_title("Variance partition (rank-controlled)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"variance_partition.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["G_low", "H_low", "H_unique"]
    vals = [ranks[n]["effective_rank"] for n in names]
    ax.bar(names, vals, color=["tab:green", "tab:blue", "tab:orange"])
    ax.set_ylabel("effective rank (entropy-based, train)")
    ax.set_title("Effective rank comparison")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"effective_rank_comparison.{ext}"))
    plt.close(fig)

    log(f"complete in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
