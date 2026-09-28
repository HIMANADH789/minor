"""CONFOUND-ISOLATED FULL-DIMENSION CCA-ADAPTIVE GENERALIZED RIDGE
("Full Canonical Ridge") — Haptics, seed 42.

Purpose: the previous Canonical Ridge experiment (results/
heramba_canonical_ridge/haptics_seed42, preserved unchanged) tested
CCA-adaptive shrinkage AND 4998->29 G compression simultaneously.  This
experiment isolates the shrinkage mechanism by keeping ALL original
features in the final predictive design:

    Z = [G_full(4998) || H_cca(29) || H_perp(4969)]  =  9996 columns

where H_cca/H_perp re-express the FULL H in the frozen orthonormal basis
obtained by completing the 29 train-only CCA-identified raw-H directions
(change of basis, no information discarded).  G keeps all 4998 original
MiniROCKET features.

Models (single test evaluation each):
  A  MiniROCKET          G_full, canonical RidgeClassifierCV (train+val)   gate 0.5037
  B  Raw full G+H        [G||H], canonical RidgeClassifierCV (train+val)   gate 0.5500
  C  Hard CCA unique     [G||H_unique], previous ranked-CCA pipeline       gate 0.5138
  D  Uniform rotated     [G||H_cca||H_perp], lam0=1, dual ridge, train-only GCV
                          (basis-invariance twin of B)
  E  Proposed adaptive   same design, lam0 = [1_G, 1+rho_k^2, 1_perp],
                          dual ridge, train-only GCV        (gamma = 1.0 fixed)

alpha_base via TRAIN-ONLY GCV over logspace(-4, 4, 81); no gamma/rho/
tau/mixing search of any kind.  Permutation-null CCA (S=500, seed
52042) is diagnostic only.
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

from models.canonical_ridge_full.model import (  # noqa: E402
    DualGeneralizedRidge, adaptive_delta, canonical_direction_matrix,
    complete_orthonormal_basis, unit_directions)
from models.heramba_cca.cca import CCAFit, effective_rank  # noqa: E402
from experiments.heramba_cca_ranked_haptics_seed42.runner import (  # noqa: E402
    RankPCA, SharedResidualLow, ridge_eval)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CACHE = r"C:/temp/results"
OUT = os.path.join(ROOT, "results", "heramba_canonical_ridge_full",
                   "haptics_seed42")

SEED = 42
GAMMA = 1.0                    # predeclared shrinkage constant (NOT tuned)
VAR_THRESHOLD = 0.95           # predeclared rank rule (as ranked experiment)
TAU_SHARED = 0.5               # diagnostics only (previous experiment's rule)
ALPHA_GRID = np.logspace(-4, 4, 81)
S_PERM = 500                   # permutation-null CCA, diagnostic only
PERM_SEED = 52042
GATE_A, GATE_B, GATE_C = 0.5037, 0.55, 0.5138


def log(m):
    print(m, flush=True)


# ----------------------------------------------------------------------
# data + identity gates
# ----------------------------------------------------------------------
def load_all():
    G_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_G_trva.npy")
                     ).astype(np.float64)
    H_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_H_trva.npy")
                     ).astype(np.float64)
    G_te = np.load(os.path.join(CACHE, "haptics_inference_banks_G_te.npy")
                   ).astype(np.float64)
    H_te = np.load(os.path.join(CACHE, "haptics_inference_banks_H_te.npy")
                   ).astype(np.float64)
    from experiments.external_stack_generalization.data import load_dataset
    d = load_dataset("Haptics")
    ytr, yva, yte = (np.asarray(d["ytr"]), np.asarray(d["yva"]),
                     np.asarray(d["yte"]))
    assert G_trva.shape == (155, 4998) and H_trva.shape == (155, 4998)
    assert G_te.shape == (308, 4998) and H_te.shape == (308, 4998)
    assert len(ytr) == 132 and len(yva) == 23 and len(yte) == 308
    assert np.isfinite(G_trva).all() and np.isfinite(H_te).all()
    return G_trva, H_trva, G_te, H_te, ytr, yva, yte


def build_h_unique(G_trva, H_trva, H_te, n_tr):
    """Reproduce the previous ranked-CCA [G||H_unique] pipeline EXACTLY
    (frozen implementation, imported unchanged)."""
    G_tr, H_tr = G_trva[:n_tr], H_trva[:n_tr]
    pcaG = RankPCA().fit(G_tr)
    pcaH = RankPCA().fit(H_tr)
    cca = CCAFit().fit(pcaG.transform_low(G_tr), pcaH.transform_low(H_tr),
                       tau_shared=TAU_SHARED)
    k_shared = cca.k_shared_
    resid = SharedResidualLow().fit(pcaH.transform_low(H_tr),
                                    cca.project_H(pcaH.transform_low(H_tr))[:, :k_shared])
    Hu_trva = resid.transform(pcaH.transform_low(H_trva),
                              cca.project_H(pcaH.transform_low(H_trva))[:, :k_shared])
    Hu_te = resid.transform(pcaH.transform_low(H_te),
                            cca.project_H(pcaH.transform_low(H_te))[:, :k_shared])
    return Hu_trva, Hu_te, k_shared


# ----------------------------------------------------------------------
# rank-controlled CCA discovery + full-dimension basis construction
# ----------------------------------------------------------------------
def fit_decomposition(G_trva, H_trva, n_tr):
    """Train-only RankPCA (95% rule) + CCA + raw-space mapping + full-QR
    completion.  Returns the frozen transform components and diagnostics."""
    G_tr, H_tr = G_trva[:n_tr], H_trva[:n_tr]
    pcaG = RankPCA().fit(G_tr)
    pcaH = RankPCA().fit(H_tr)
    G_low_tr = pcaG.transform_low(G_tr)
    H_low_tr = pcaH.transform_low(H_tr)
    cca = CCAFit().fit(G_low_tr, H_low_tr, tau_shared=TAU_SHARED)
    rho = np.asarray(cca.rho_, dtype=np.float64)
    K = len(rho)
    assert not np.allclose(rho, 1.0), "degenerate CCA — STOP"

    hsd = H_low_tr.std(0)
    D = canonical_direction_matrix(pcaH.comp_, pcaH.ok_, hsd, cca.B_)
    U = unit_directions(D)
    assert U.shape[1] == K, f"rank-deficient direction map: {U.shape} vs K={K}"

    Q_cca, Q_perp = complete_orthonormal_basis(U)
    Q = np.hstack([Q_cca, Q_perp])
    orth_err = float(np.abs(Q.T @ Q - np.eye(H_tr.shape[1])).max())
    assert orth_err < 1e-10, f"Q not orthonormal: {orth_err}"

    diag = {
        "g_rank": int(pcaG.k_), "h_rank": int(pcaH.k_),
        "g_energy_kept": float(pcaG.cumulative_variance_[pcaG.k_ - 1]),
        "h_energy_kept": float(pcaH.cumulative_variance_[pcaH.k_ - 1]),
        "n_cca_components": int(K), "rho": rho.tolist(),
        "rho_max": float(rho.max()), "rho_min": float(rho.min()),
        "rho_mean": float(rho.mean()),
        "orthonormality_error": orth_err,
    }
    return pcaG, pcaH, cca, rho, Q_cca, Q_perp, diag


def transform_designs(G_trva, H_trva, G_te, H_te, Q_cca, Q_perp, n_tr):
    """Frozen change of basis: G stays raw (train-centered), H -> CCA
    coordinates + orthogonal complement.  Returns train/val/test designs
    (assembler only; the dual solver centers on the train rows it sees)."""
    mu_G = G_trva[:n_tr].mean(axis=0, keepdims=True)
    mu_H = H_trva[:n_tr].mean(axis=0, keepdims=True)
    out = {}
    for name, G, H in (("train", G_trva[:n_tr], H_trva[:n_tr]),
                       ("val", G_trva[n_tr:], H_trva[n_tr:]),
                       ("test", G_te, H_te)):
        Hc = H - mu_H
        out[name] = np.hstack([G - mu_G, Hc @ Q_cca, Hc @ Q_perp])
    return out


def reconstruction_check(H_trva, H_te, Q_cca, Q_perp, mu_H, n_tr):
    """||H_c - H_c QQ^T|| / ||H_c|| must be ~ machine precision."""
    errs = {}
    for name, H in (("train", H_trva[:n_tr]), ("val", H_trva[n_tr:]),
                    ("test", H_te)):
        Hc = H - mu_H
        Q = np.hstack([Q_cca, Q_perp])
        errs[name] = float(np.linalg.norm(Hc @ Q @ Q.T - Hc) /
                           max(np.linalg.norm(Hc), 1e-300))
    return errs


# ----------------------------------------------------------------------
# permutation-null CCA (diagnostic only)
# ----------------------------------------------------------------------
def permutation_null_cca(G_low_tr, H_low_tr, S=S_PERM, seed=PERM_SEED):
    rng = np.random.default_rng(seed)
    n, K = H_low_tr.shape[0], None
    null = []
    for _ in range(S):
        perm = rng.permutation(n)
        cca_n = CCAFit().fit(G_low_tr, H_low_tr[perm], tau_shared=1.1)
        null.append(np.asarray(cca_n.rho_, dtype=np.float64))
    L = max(len(r) for r in null)
    mat = np.full((S, L), np.nan)
    for i, r in enumerate(null):
        mat[i, :len(r)] = r
    return mat


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    for sub in ("predictions", "figures"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    log("=== FULL-DIMENSION CCA-ADAPTIVE RIDGE — Haptics seed 42 ===")

    G_trva, H_trva, G_te, H_te, ytr, yva, yte = load_all()
    n_tr = len(ytr)
    y_dev = np.concatenate([ytr, yva])

    # ---- identity gates A/B/C (canonical RidgeClassifierCV, train+val) --
    m0_res, m0_pred = ridge_eval(G_trva, y_dev, G_te, yte)
    assert abs(m0_res["macro_f1"] - GATE_A) < 5e-5, f"gate A: {m0_res}"
    log(f"[gate A] MiniROCKET = {m0_res['macro_f1']} OK")
    r2_res, r2_pred = ridge_eval(np.hstack([G_trva, H_trva]), y_dev,
                                 np.hstack([G_te, H_te]), yte)
    assert r2_res["macro_f1"] == GATE_B, f"gate B: {r2_res}"
    log(f"[gate B] Raw G+H   = {r2_res['macro_f1']} OK")
    Hu_trva, Hu_te, k_shared = build_h_unique(G_trva, H_trva, H_te, n_tr)
    hu_res, hu_pred = ridge_eval(np.hstack([G_trva, Hu_trva]), y_dev,
                                 np.hstack([G_te, Hu_te]), yte)
    assert abs(hu_res["macro_f1"] - GATE_C) < 5e-5, f"gate C: {hu_res}"
    log(f"[gate C] G+H_unique = {hu_res['macro_f1']} OK "
        f"(H_unique dim {Hu_trva.shape[1]}, shared {k_shared})")

    # ---- decomposition (train-only) -------------------------------------
    pcaG, pcaH, cca, rho, Q_cca, Q_perp, diag = fit_decomposition(
        G_trva, H_trva, n_tr)
    K = diag["n_cca_components"]
    log(f"ranks g={diag['g_rank']} h={diag['h_rank']}; CCA K={K} "
        f"rho[min,max,mean]={diag['rho_min']:.3f}/{diag['rho_max']:.3f}/"
        f"{diag['rho_mean']:.3f}")

    mu_H = H_trva[:n_tr].mean(axis=0, keepdims=True)
    recon = reconstruction_check(H_trva, H_te, Q_cca, Q_perp, mu_H, n_tr)
    log(f"basis reconstruction rel.err: {recon}")
    assert max(recon.values()) < 1e-10

    Z = transform_designs(G_trva, H_trva, G_te, H_te, Q_cca, Q_perp, n_tr)
    nG, nH = 4998, 4998
    assert Z["train"].shape == (n_tr, nG + nH)
    assert Z["test"].shape[1] == 9996

    # ---- basis-invariance gate (spec 21): ordinary ridge on raw vs
    #      transformed H, same rows, same solver conventions --------------
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import f1_score
    Zraw_dev = np.hstack([G_trva, H_trva])
    Zraw_te = np.hstack([G_te, H_te])
    rc_raw = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)).fit(Zraw_dev, y_dev)
    rc_rot = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)).fit(
        np.vstack([Z["train"], Z["val"]]), y_dev)
    d_raw = rc_raw.decision_function(Zraw_te)
    d_rot = rc_rot.decision_function(Z["test"])
    inv_sklearn = float(np.abs(d_raw - d_rot).max())
    p_raw = rc_raw.predict(Zraw_te)
    p_rot = rc_rot.predict(Z["test"])
    f1_raw = f1_score(yte, p_raw, average="macro", zero_division=0)
    f1_rot = f1_score(yte, p_rot, average="macro", zero_division=0)
    # dual-solver invariance (same solver on both representations)
    dr_raw = DualGeneralizedRidge().fit(Zraw_dev[:n_tr], ytr)
    dr_rot = DualGeneralizedRidge().fit(Z["train"], ytr)
    inv_dual = float(np.abs(
        dr_raw.decision_function(Zraw_te) -
        dr_rot.decision_function(Z["test"])).max())
    log(f"[basis invariance] sklearn max|dlogit|={inv_sklearn:.2e} "
        f"(F1 {f1_raw:.4f} vs {f1_rot:.4f}); dual max|dlogit|={inv_dual:.2e}")
    assert inv_sklearn < 1e-8 and inv_dual < 1e-8 and f1_raw == f1_rot, \
        "basis-invariance gate FAILED"

    # ---- uniform (D) and adaptive (E) full-dimensional dual ridge -------
    lam_uniform = np.ones(nG + nH)
    lam_adapt = np.ones(nG + nH)
    delta = adaptive_delta(rho, GAMMA)
    lam_adapt[nG:nG + K] = 1.0 + delta
    log(f"adaptive lam0 CCA block: [{(1+delta).min():.4f}, {(1+delta).max():.4f}]")

    uni = DualGeneralizedRidge(lam0=lam_uniform, alpha_grid=ALPHA_GRID
                               ).fit(Z["train"], ytr)
    ada = DualGeneralizedRidge(lam0=lam_adapt, alpha_grid=ALPHA_GRID
                               ).fit(Z["train"], ytr)
    for nm, mdl in (("uniform", uni), ("adaptive", ada)):
        dg = mdl.diagnostics
        log(f"[{nm}] alpha_base={mdl.alpha_base_:.4g} df={dg['df']:.1f} "
            f"rank(S)={dg['rank_Z_train']} cond={dg['cond_regularized']:.3g} "
            f"cholesky={dg['cholesky_ok']}")

    def eval_dual(mdl, name):
        pred_te = mdl.predict(Z["test"])
        pred_va = mdl.predict(Z["val"])
        from sklearn.metrics import accuracy_score
        return {
            "macro_f1": round(float(f1_score(yte, pred_te, average="macro",
                                             zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
            "val_macro_f1": round(float(f1_score(yva, pred_va, average="macro",
                                                 zero_division=0)), 4),
            "alpha_base": float(mdl.alpha_base_),
            "df": mdl.diagnostics["df"],
            "n_features": int(mdl.diagnostics["p"]),
            "per_class_f1": [round(float(v), 4) for v in f1_score(
                yte, pred_te, average=None, zero_division=0,
                labels=list(range(5)))],
        }, pred_te

    uni_res, uni_pred = eval_dual(uni, "uniform")
    ada_res, ada_pred = eval_dual(ada, "adaptive")
    log(f"[D uniform]  test MF1={uni_res['macro_f1']} (val {uni_res['val_macro_f1']})")
    log(f"[E adaptive] test MF1={ada_res['macro_f1']} (val {ada_res['val_macro_f1']})")

    # ---- supplementary final-fit arm (train+val rows, alpha FROZEN from
    #      train-only GCV; validation labels contribute rows only, never
    #      selection) — mirrors the canonical final-fit convention --------
    uniF = DualGeneralizedRidge(lam0=lam_uniform, alpha_grid=ALPHA_GRID
                                ).fit(np.vstack([Z["train"], Z["val"]]), y_dev)
    adaF = DualGeneralizedRidge(lam0=lam_adapt, alpha_grid=ALPHA_GRID
                                ).fit(np.vstack([Z["train"], Z["val"]]), y_dev)
    # freeze alpha at the train-only GCV value (no selection on val)
    uniF.refit_at(uni.alpha_base_)
    adaF.refit_at(ada.alpha_base_)
    adaF_res, adaF_pred = eval_dual(adaF, "adaptive_finalfit")
    log(f"[E' adaptive final-fit (train+val, alpha frozen)] test MF1="
        f"{adaF_res['macro_f1']} (alpha={adaF.alpha_base_:.4g})")

    # ---- permutation-null CCA (diagnostic only) --------------------------
    log(f"permutation-null CCA S={S_PERM} seed={PERM_SEED} ...")
    G_low_tr = pcaG.transform_low(G_trva[:n_tr])
    H_low_tr = pcaH.transform_low(H_trva[:n_tr])
    null_mat = permutation_null_cca(G_low_tr, H_low_tr)
    null_mean = np.nanmean(null_mat, axis=0)
    null_std = np.nanstd(null_mat, axis=0)
    emp_p = (1.0 + (null_mat >= rho[None, :]).sum(axis=0)) / (S_PERM + 1.0)
    # BH-FDR over the K components
    order = np.argsort(emp_p)
    ranked = emp_p[order]
    fdr_sorted = np.minimum.accumulate((ranked * K /
                                        np.arange(1, K + 1))[::-1])[::-1]
    fdr = np.empty(K)
    fdr[order] = np.clip(fdr_sorted, 0, 1)
    n_exceed = int((emp_p <= 0.05).sum())
    log(f"null: real rho max {rho.max():.3f} vs null mean max "
        f"{null_mean.max():.3f}; components with emp p<=0.05: {n_exceed}/{K}")

    # ---- diagnostics CSVs ------------------------------------------------
    H_cca_te = Z["test"][:, nG:nG + K]
    coef_cca = ada.coef_[nG:nG + K, :]          # (K, n_classes)
    # per-direction test contribution: coord_k * (class-summed coef row k)
    contrib_te = H_cca_te * coef_cca.sum(axis=1)[None, :]   # (n_test, K)
    with open(os.path.join(OUT, "canonical_ridge_diagnostics.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "rho", "delta", "alpha_k", "alpha_ratio",
                    "coord_var_train", "coef_norm", "test_contrib_std"])
        for k in range(K):
            w.writerow([
                k + 1, round(float(rho[k]), 6), round(float(delta[k]), 6),
                round(float(ada.alpha_base_ * (1 + delta[k])), 6),
                round(float(1 + delta[k]), 6),
                round(float(Z["train"][:, nG + k].var()), 6),
                round(float(np.linalg.norm(coef_cca[k])), 6),
                round(float(contrib_te[:, k].std()), 6)])
    with open(os.path.join(OUT, "canonical_correlations.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "rho", "null_mean", "null_std",
                    "emp_p", "fdr_q"])
        for k in range(K):
            w.writerow([k + 1, round(float(rho[k]), 6),
                        round(float(null_mean[k]), 6),
                        round(float(null_std[k]), 6),
                        round(float(emp_p[k]), 5), round(float(fdr[k]), 5)])
    with open(os.path.join(OUT, "alpha_schedule.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "rho", "delta", "alpha_k"])
        w.writerow(["G_block", "", "", ada.alpha_base_])
        w.writerow(["H_perp_block", "", "", ada.alpha_base_])
        for k in range(K):
            w.writerow([f"H_cca_{k+1}", round(float(rho[k]), 6),
                        round(float(delta[k]), 6),
                        round(float(ada.alpha_base_ * (1 + delta[k])), 6)])

    # ---- predictions + JSON artifacts ------------------------------------
    np.save(os.path.join(OUT, "predictions", "minirocket.npy"), m0_pred)
    np.save(os.path.join(OUT, "predictions", "raw_gh.npy"), r2_pred)
    np.save(os.path.join(OUT, "predictions", "cca_unique.npy"), hu_pred)
    np.save(os.path.join(OUT, "predictions", "uniform_rotated.npy"), uni_pred)
    np.save(os.path.join(OUT, "predictions", "canonical_ridge.npy"), ada_pred)
    np.save(os.path.join(OUT, "predictions", "canonical_ridge_finalfit.npy"),
            adaF_pred)

    inv_report = {
        "sklearn_ridge_max_abs_decision_diff": inv_sklearn,
        "sklearn_ridge_f1_raw": round(float(f1_raw), 6),
        "sklearn_ridge_f1_rotated": round(float(f1_rot), 6),
        "dual_uniform_max_abs_decision_diff": inv_dual,
        "tolerance": 1e-8, "passed": bool(inv_sklearn < 1e-8 and inv_dual < 1e-8),
    }
    with open(os.path.join(OUT, "basis_invariance_report.json"), "w") as f:
        json.dump(inv_report, f, indent=1)

    numerical = {
        "uniform": {k: v for k, v in uni.diagnostics.items()
                    if k not in ("gcv_grid", "gcv_values")},
        "adaptive": {k: v for k, v in ada.diagnostics.items()
                     if k not in ("gcv_grid", "gcv_values")},
        "basis_reconstruction_rel_err": recon,
        "orthonormality_error": diag["orthonormality_error"],
    }
    with open(os.path.join(OUT, "numerical_diagnostics.json"), "w") as f:
        json.dump(numerical, f, indent=1)

    deltas_pp = {
        "vs_minirocket": round((ada_res["macro_f1"] - m0_res["macro_f1"]) * 100, 2),
        "vs_raw_gh": round((ada_res["macro_f1"] - r2_res["macro_f1"]) * 100, 2),
        "vs_h_unique": round((ada_res["macro_f1"] - hu_res["macro_f1"]) * 100, 2),
        "vs_uniform_rotated": round((ada_res["macro_f1"] - uni_res["macro_f1"]) * 100, 2),
    }
    results = {
        "seed": SEED, "dataset": "Haptics", "gamma": GAMMA,
        "var_threshold": VAR_THRESHOLD,
        "g_rank": diag["g_rank"], "h_rank": diag["h_rank"],
        "n_cca_components": K, "rho": diag["rho"],
        "alpha_base_uniform": uni.alpha_base_,
        "alpha_base_adaptive": ada.alpha_base_,
        "alpha_cca_range": [float((ada.alpha_base_ * (1 + delta)).min()),
                            float((ada.alpha_base_ * (1 + delta)).max())],
        "minirocket": m0_res, "raw_gh": r2_res, "cca_unique": hu_res,
        "uniform_rotated": uni_res, "canonical_ridge": ada_res,
        "canonical_ridge_finalfit_trainval_alpha_frozen": adaF_res,
        "canonical_ridge_deltas_pp": deltas_pp,
        "permutation_null": {"S": S_PERM, "seed": PERM_SEED,
                             "n_components_p_le_0.05": n_exceed,
                             "null_mean_max": float(null_mean.max()),
                             "real_rho_max": float(rho.max())},
        "basis_invariance": inv_report,
        "gates": {"A_minirocket": GATE_A, "B_raw_gh": GATE_B,
                  "C_h_unique": GATE_C},
        "final_test_evaluations": 5,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({
            "seed": SEED, "gamma": GAMMA, "var_threshold": VAR_THRESHOLD,
            "alpha_grid": "logspace(-4,4,81) train-only GCV",
            "gcv_rule": "n*RSS/(n-df)^2, exact df, train rows only",
            "classifier": "one-hot generalized ridge, dual n-space solver",
            "lam0_uniform": "ones(9996)",
            "lam0_adaptive": "[1_G(4998), 1+gamma*rho_k^2 (K=29), 1_perp(4969)]",
            "permutation_null": {"S": S_PERM, "seed": PERM_SEED,
                                 "role": "diagnostic only"},
            "fit_sets": {"gates_ABC": "train+val (canonical convention)",
                         "dual_models": "train only (GCV)"},
            "paths": {"G_trva": os.path.join(CACHE, "haptics_inference_banks_G_trva.npy"),
                      "H_trva": os.path.join(CACHE, "haptics_inference_banks_H_trva.npy"),
                      "G_te": os.path.join(CACHE, "haptics_inference_banks_G_te.npy"),
                      "H_te": os.path.join(CACHE, "haptics_inference_banks_H_te.npy")},
        }, f, indent=1)

    # ---- final comparison table ------------------------------------------
    with open(os.path.join(OUT, "final_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "G dims retained", "H dims retained",
                    "Total dims", "Ridge type", "alpha_base",
                    "Test Macro-F1", "Validation Macro-F1", "Notes"])
        w.writerow(["MiniROCKET", 4998, 0, 4998, "scalar RidgeClassifierCV",
                    m0_res["selected_alpha"], m0_res["macro_f1"], "",
                    "identity gate A"])
        w.writerow(["Raw full G+H", 4998, 4998, 9996,
                    "scalar RidgeClassifierCV", r2_res["selected_alpha"],
                    r2_res["macro_f1"], "", "identity gate B"])
        w.writerow(["Hard CCA unique", 4998, int(Hu_trva.shape[1]),
                    4998 + int(Hu_trva.shape[1]), "scalar RidgeClassifierCV",
                    hu_res["selected_alpha"], hu_res["macro_f1"], "",
                    "previous ranked-CCA pipeline, reference row"])
        w.writerow(["Uniform rotated full Ridge", 4998, 4998, 9996,
                    "generalized ridge lam0=1 (dual, GCV)", uni.alpha_base_,
                    uni_res["macro_f1"], uni_res["val_macro_f1"],
                    "basis-invariance twin of B; train-only fit"])
        w.writerow(["Proposed full Canonical Ridge", 4998, 4998, 9996,
                    "CCA-adaptive lam0 (dual, GCV)", ada.alpha_base_,
                    ada_res["macro_f1"], ada_res["val_macro_f1"],
                    f"gamma={GAMMA} fixed; alpha_k=alpha*(1+rho^2)"])
        w.writerow(["Proposed full Canonical Ridge (train+val final fit)",
                    4998, 4998, 9996,
                    "CCA-adaptive lam0 (dual, alpha frozen)", adaF.alpha_base_,
                    adaF_res["macro_f1"], "",
                    "alpha from train-only GCV, frozen; val rows only add rows"])

    # ---- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(rho, ada.alpha_base_ * (1 + delta), "o", ms=5)
    ax.axhline(ada.alpha_base_, ls="--", c="gray", lw=1,
               label=f"alpha_base = {ada.alpha_base_:.3g}")
    ax.set_xlabel(r"canonical correlation $\rho_k$ (train)")
    ax.set_ylabel(r"$\alpha_k=\alpha_{base}(1+\gamma\rho_k^2)$")
    ax.set_title(f"Predeclared adaptive schedule (gamma={GAMMA})")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"rho_vs_alpha.{ext}"))
    plt.close(fig)

    xs = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(xs, null_mean, yerr=null_std, fmt="s", ms=3, capsize=2,
                label=f"null (S={S_PERM}) mean±SD", color="tab:gray")
    ax.plot(xs, rho, "o", ms=5, label="real rho (train)", color="tab:red")
    ax.set_xlabel("CCA component")
    ax.set_ylabel("canonical correlation")
    ax.set_title(f"CCA permutation-null spectrum "
                 f"({n_exceed}/{K} components p<=0.05)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"cca_null_spectrum.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    diff = np.abs(d_raw - d_rot).max(axis=1)
    ax.hist(np.log10(np.maximum(diff, 1e-18)), bins=30, color="tab:blue")
    ax.axvline(np.log10(1e-8), ls="--", c="tab:red",
               label="tolerance 1e-8")
    ax.set_xlabel(r"$\log_{10}$ per-sample max |decision diff| (raw vs rotated)")
    ax.set_ylabel("test samples")
    ax.set_title(f"Basis invariance: max diff {inv_sklearn:.1e}")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"basis_invariance.{ext}"))
    plt.close(fig)

    log(f"complete in {time.time()-t0:.0f}s -> {OUT}")
    return results


if __name__ == "__main__":
    main()
