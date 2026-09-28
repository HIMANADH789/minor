"""CCA-adaptive generalized ridge ("Canonical Ridge") — Haptics, seed 42.

PROPOSED experimental model (not a literature-established formulation):
direction-specific shrinkage in the rank-controlled CCA basis of H,
    alpha_k = alpha_base * (1 + rho_k^2),   gamma = 1.0 (fixed),
retaining ALL canonical directions (no hard tau=0.5 discard) plus the
non-canonical orthogonal complement of H_low.

Pipeline (all fits on TRAIN = 132 dev rows; labels never enter any fit):
    G_raw/H_raw (frozen canonical banks, identity-gated)
    -> RankPCA (95% train variance; rule predeclared) -> G_low(29)/H_low(88)
    -> CCA(G_low, H_low) -> rho_1..rho_29, B (H-side directions)
    -> CanonicalBasis: H_CCA(29) + H_perp(59), exact reconstruction
    -> Z = [G_low || H_CCA || H_perp]  (117 features)
    -> generalized ridge (one-hot least squares), Lambda = diag blocks:
       G: alpha_base * 1, H_CCA: alpha_base*(1+rho_k^2), H_perp: alpha_base*1
       alpha_base selected by TRAIN-ONLY GCV over logspace(-4,4,81)
    -> final fit on TRAIN+VAL dev (canonical convention), ONE test eval.

Variants in the same run:
    A MiniROCKET G (canonical RidgeClassifierCV)         [gate: 0.5037]
    B raw [G||H] (canonical RidgeClassifierCV)           [gate: 0.5500]
    C hard-CCA [G||H_unique] (ranked experiment rerun)   [gate: 0.5138]
    E uniform generalized ridge (all H dirs alpha_base)  [ablation]
    F adaptive canonical ridge (the proposed model)      [primary]
Diagnostic (model-independent): train-only permutation-null CCA
    (S=500, seed 52042) reporting null mean/std + empirical p per rho_k.
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

from experiments.heramba_cca_ranked_haptics_seed42.runner import (  # noqa
    RankPCA, SharedResidualLow)
from models.canonical_ridge.model import (  # noqa: E402
    CanonicalBasis, GeneralizedRidgeClassifier, alpha_schedule, onehot)
from models.heramba_cca.cca import CCAFit  # noqa: E402
from models.heramba_cca.model import ALPHAS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CACHE = r"C:/temp/results"
OUT = os.path.join(ROOT, "results", "heramba_canonical_ridge",
                   "haptics_seed42")

GAMMA = 1.0
ALPHA_GRID = np.logspace(-4, 4, 81)
N_PERM_CCA = 500
PERM_SEED_CCA = 52042


def log(m):
    print(m, flush=True)


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


def genridge_eval(Z_dev, Z_te, y_dev, yte, n_classes, lam_pattern,
                  alpha_grid=ALPHA_GRID):
    """Generalized-ridge fit on dev (train+val), one test eval."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score)
    clf = GeneralizedRidgeClassifier().fit(
        Z_dev, onehot(y_dev, n_classes), lam_pattern, alpha_grid=alpha_grid)
    pred = clf.predict(Z_te)
    res = {
        "macro_f1": round(float(f1_score(yte, pred, average="macro",
                                         zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "per_class_f1": [round(float(v), 4) for v in f1_score(
            yte, pred, average=None, zero_division=0,
            labels=list(range(n_classes)))],
        "confusion_matrix": confusion_matrix(
            yte, pred, labels=list(range(n_classes))).tolist(),
        "n_features": int(Z_dev.shape[1]),
        "alpha_base": float(clf.alpha_base_),
        "df_final": round(clf.df_, 3),
        "gcv": round(clf.gcv_, 5),
        "diagnostics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in clf.diagnostics_.items()},
    }
    return res, pred, clf


def permutation_null_cca(G_low_tr, H_low_tr, S=N_PERM_CCA,
                         seed=PERM_SEED_CCA):
    """TRAIN-ONLY null: shuffle H rows, refit CCA, compare spectra."""
    rng = np.random.RandomState(seed)
    real = CCAFit().fit(G_low_tr, H_low_tr, tau_shared=0.5)
    rho_real = np.asarray(real.rho_)
    K = len(rho_real)
    null = np.empty((S, K))
    for s in range(S):
        Hp = H_low_tr[rng.permutation(len(H_low_tr))]
        null[s] = CCAFit().fit(G_low_tr, Hp, tau_shared=0.5).rho_
        if (s + 1) % 100 == 0:
            log(f"    cca-null {s+1}/{S}")
    rows = []
    for k in range(K):
        p = float((1 + int((null[:, k] >= rho_real[k]).sum())) / (S + 1))
        rows.append({
            "component": k + 1, "rho_real": round(float(rho_real[k]), 6),
            "null_mean": round(float(null[:, k].mean()), 6),
            "null_std": round(float(null[:, k].std()), 6),
            "empirical_p": round(p, 5),
        })
    return real, rows


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    log("=== CCA-ADAPTIVE GENERALIZED RIDGE — Haptics seed 42 ===")

    # ---- frozen banks + labels ----------------------------------------
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
    n_tr = len(ytr)
    y_dev = np.concatenate([ytr, yva])
    n_classes = d["n_classes"]
    assert n_classes == 5 and len(G_trva) == 155 and len(G_te) == 308

    # ---- baselines A/B (canonical Ridge, gates) ------------------------
    m0_res, m0_pred = ridge_eval(G_trva, y_dev, G_te, yte)
    assert m0_res["macro_f1"] == 0.5037, m0_res
    log(f"[A gate] MiniROCKET = {m0_res['macro_f1']} OK")
    r2_res, r2_pred = ridge_eval(np.hstack([G_trva, H_trva]), y_dev,
                                 np.hstack([G_te, H_te]), yte)
    assert r2_res["macro_f1"] == 0.55, r2_res
    log(f"[B gate] raw [G||H] = {r2_res['macro_f1']} OK")

    # ---- train-only PCA (predeclared 95% rule) -------------------------
    G_tr, H_tr = G_trva[:n_tr], H_trva[:n_tr]
    pcaG, pcaH = RankPCA().fit(G_tr), RankPCA().fit(H_tr)
    G_low_tr = pcaG.transform_low(G_tr)
    H_low_tr = pcaH.transform_low(H_tr)
    G_low_dev = pcaG.transform_low(G_trva)
    H_low_dev = pcaH.transform_low(H_trva)
    G_low_te = pcaG.transform_low(G_te)
    H_low_te = pcaH.transform_low(H_te)
    log(f"rank rule: g={pcaG.k_}, h={pcaH.k_}")

    # ---- CCA on TRAIN + permutation-null diagnostic ---------------------
    cca, null_rows = permutation_null_cca(G_low_tr, H_low_tr)
    rho = np.asarray(cca.rho_)
    K = len(rho)
    log(f"CCA: K={K}, rho max={rho.max():.4f} min={rho.min():.4f} "
        f"mean={rho.mean():.4f}")
    n_sig = sum(1 for r in null_rows if r["empirical_p"] <= 0.05)
    log(f"permutation-null CCA (S={N_PERM_CCA}, seed {PERM_SEED_CCA}): "
        f"{n_sig}/{K} directions with p<=0.05")

    # ---- hard-CCA baseline C (ranked experiment rerun, gate) ------------
    k_shared = cca.k_shared_
    shared_scores_tr = cca.project_H(H_low_tr)[:, :k_shared]
    resid_tr = SharedResidualLow().fit(H_low_tr, shared_scores_tr)
    Hu_tr = resid_tr.transform(H_low_tr, shared_scores_tr)
    shared_scores_dev = cca.project_H(H_low_dev)[:, :k_shared]
    Hu_dev = resid_tr.transform(H_low_dev, shared_scores_dev)
    shared_scores_te = cca.project_H(H_low_te)[:, :k_shared]
    Hu_te = resid_tr.transform(H_low_te, shared_scores_te)
    log(f"[C] hard tau={0.5}: shared={k_shared}, "
        f"H_unique eff dim={int((np.abs(Hu_tr - Hu_tr.mean(0)).sum(0) > 1e-8).sum())}")
    # [G || H_unique] in the same reduced space as the ranked experiment:
    # the ranked experiment used G(raw 4998) || Hu(reduced 88) -> reproduce
    # with the frozen Hu transform applied identically:
    hu_res, hu_pred = ridge_eval(np.hstack([G_trva, Hu_dev]), y_dev,
                                 np.hstack([G_te, Hu_te]), yte)
    log(f"[C gate] hard-CCA [G||H_unique] = {hu_res['macro_f1']} "
        f"(expected 0.5138)")
    assert hu_res["macro_f1"] == 0.5138, hu_res

    # ---- canonical basis + full frozen H transform ----------------------
    basis = CanonicalBasis.fit(H_low_tr, cca)
    Hc_tr, Hp_tr = basis.transform(H_low_tr)
    Hc_dev, Hp_dev = basis.transform(H_low_dev)
    Hc_te, Hp_te = basis.transform(H_low_te)
    log(f"canonical basis: K={basis.K_}, H_perp dim={Hp_tr.shape[1]}, "
        f"recon err={basis.recon_max_err_:.2e}")
    Z_tr = np.hstack([G_low_tr, Hc_tr, Hp_tr])
    Z_dev = np.hstack([G_low_dev, Hc_dev, Hp_dev])
    Z_te = np.hstack([G_low_te, Hc_te, Hp_te])
    assert np.isfinite(Z_dev).all() and np.isfinite(Z_te).all()

    # ---- variants E (uniform) and F (adaptive) --------------------------
    dG = G_low_dev.shape[1]
    dHp = Hp_dev.shape[1]
    lam_uniform = np.concatenate([np.ones(dG), np.ones(K), np.ones(dHp)])
    lam_adaptive = np.concatenate([np.ones(dG),
                                   1.0 + GAMMA * rho ** 2,
                                   np.ones(dHp)])
    log("[E] uniform generalized ridge (dev fit + GCV)...")
    uni_res, uni_pred, uni_clf = genridge_eval(
        Z_dev, Z_te, y_dev, yte, n_classes, lam_uniform)
    log(f"    alpha_base={uni_res['alpha_base']:.4f} "
        f"df={uni_res['df_final']} test={uni_res['macro_f1']}")
    log("[F] adaptive canonical ridge (proposed model)...")
    ada_res, ada_pred, ada_clf = genridge_eval(
        Z_dev, Z_te, y_dev, yte, n_classes, lam_adaptive)
    log(f"    alpha_base={ada_res['alpha_base']:.4f} "
        f"df={ada_res['df_final']} test={ada_res['macro_f1']}")
    # validation Macro-F1 (reporting only)
    from sklearn.metrics import f1_score as _f1
    va_sl = slice(n_tr, len(y_dev))
    uni_res["val_macro_f1"] = round(float(_f1(
        y_dev[va_sl], uni_clf.predict(Z_dev[va_sl]),
        average="macro", zero_division=0)), 4)
    ada_res["val_macro_f1"] = round(float(_f1(
        y_dev[va_sl], ada_clf.predict(Z_dev[va_sl]),
        average="macro", zero_division=0)), 4)

    # ---- per-direction diagnostics --------------------------------------
    W_ada = ada_clf.W_
    coef_norms = np.linalg.norm(W_ada, axis=1)
    rows = []
    for k in range(K):
        col = K + k
        rows.append({
            "component": k + 1,
            "rho": round(float(rho[k]), 6),
            "alpha_k": round(float(alpha_schedule(
                np.array([rho[k]]), ada_res["alpha_base"])[0]), 6),
            "alpha_uniform": round(float(uni_res["alpha_base"]), 6),
            "variance_HCCA": round(float(Hc_tr[:, k].var()), 8),
            "coefficient_norm_adaptive": round(float(coef_norms[dG + k]), 6),
            "coefficient_norm_uniform": round(float(
                np.linalg.norm(uni_clf.W_[dG + k])), 6),
            "null_mean": null_rows[k]["null_mean"],
            "null_std": null_rows[k]["null_std"],
            "empirical_p": null_rows[k]["empirical_p"],
        })
    with open(os.path.join(OUT, "canonical_ridge_diagnostics.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "canonical_correlations.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "rho", "shared_at_tau_0.5",
                    "null_mean", "null_std", "empirical_p"])
        for k in range(K):
            w.writerow([k + 1, round(float(rho[k]), 6), k < k_shared,
                        null_rows[k]["null_mean"], null_rows[k]["null_std"],
                        null_rows[k]["empirical_p"]])
    with open(os.path.join(OUT, "alpha_schedule.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha_base", "gamma", "rho", "alpha_k"])
        for k in range(K):
            w.writerow([ada_res["alpha_base"], GAMMA,
                        round(float(rho[k]), 6),
                        round(float(alpha_schedule(
                            np.array([rho[k]]),
                            ada_res["alpha_base"])[0]), 6)])

    # ---- artifacts -------------------------------------------------------
    np.save(os.path.join(OUT, "predictions", "minirocket.npy"), m0_pred)
    np.save(os.path.join(OUT, "predictions", "raw_gh.npy"), r2_pred)
    np.save(os.path.join(OUT, "predictions", "hard_cca_unique.npy"),
            hu_pred)
    np.save(os.path.join(OUT, "predictions", "uniform_genridge.npy"),
            uni_pred)
    np.save(os.path.join(OUT, "predictions",
                         "adaptive_canonical_ridge.npy"), ada_pred)

    deltas = {
        "vs_minirocket_pp": round((ada_res["macro_f1"]
                                   - m0_res["macro_f1"]) * 100, 2),
        "vs_raw_gh_pp": round((ada_res["macro_f1"]
                               - r2_res["macro_f1"]) * 100, 2),
        "vs_hard_cca_pp": round((ada_res["macro_f1"]
                                 - hu_res["macro_f1"]) * 100, 2),
        "vs_uniform_pp": round((ada_res["macro_f1"]
                                - uni_res["macro_f1"]) * 100, 2),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({
            "seed": 42, "dataset": "Haptics",
            "model_status": "PROPOSED experimental CCA-adaptive "
                            "generalized ridge; not a literature-"
                            "established Canonical Ridge formula",
            "shrinkage_rule": f"alpha_k = alpha_base * (1 + {GAMMA} * "
                              f"rho_k^2), gamma fixed, not tuned",
            "alpha_selection": "TRAIN-ONLY GCV on dev fit "
                               "(logspace(-4,4,81))",
            "ranks": {"g_rank": int(pcaG.k_), "h_rank": int(pcaH.k_)},
            "K_canonical": int(K),
            "minirocket": m0_res, "raw_gh": r2_res, "hard_cca": hu_res,
            "uniform_genridge": uni_res,
            "adaptive_canonical_ridge": ada_res,
            "deltas_pp": deltas,
            "permutation_null_cca": {"S": N_PERM_CCA,
                                     "seed": PERM_SEED_CCA,
                                     "n_dirs_p_le_0.05": n_sig},
            "runtime_s": round(time.time() - t0, 1),
        }, f, indent=1)
    with open(os.path.join(OUT, "final_comparison.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Representation", "Test Macro-F1",
                    "Validation Macro-F1", "Dimensions", "alpha_base",
                    "Notes"])
        w.writerow(["MiniROCKET", "G", m0_res["macro_f1"], "-",
                    m0_res["n_features"], m0_res["selected_alpha"],
                    "canonical RidgeClassifierCV"])
        w.writerow(["Raw G+H", "[G||H]", r2_res["macro_f1"], "-",
                    r2_res["n_features"], r2_res["selected_alpha"],
                    "canonical R2"])
        w.writerow(["Hard CCA unique", "[G||H_unique]", hu_res["macro_f1"],
                    "-", hu_res["n_features"], hu_res["selected_alpha"],
                    "tau=0.5 discard (ranked experiment)"])
        w.writerow(["Uniform reduced gen-ridge",
                    f"[G_low||H_CCA||H_perp] ({dG}+{K}+{dHp})",
                    uni_res["macro_f1"], uni_res["val_macro_f1"],
                    uni_res["n_features"], round(uni_res["alpha_base"], 4),
                    "all dirs alpha_base; GCV alpha_base (train-only)"])
        w.writerow(["Proposed Canonical Ridge",
                    f"[G_low||H_CCA||H_perp] ({dG}+{K}+{dHp})",
                    ada_res["macro_f1"], ada_res["val_macro_f1"],
                    ada_res["n_features"], round(ada_res["alpha_base"], 4),
                    f"alpha_k=alpha_base*(1+{GAMMA}*rho_k^2); "
                    f"GCV alpha_base (train-only)"])
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({
            "seed": 42, "dataset": "Haptics", "split": "132/23/308",
            "variance_threshold": 0.95, "gamma": GAMMA,
            "alpha_grid": [float(x) for x in ALPHA_GRID],
            "alpha_selection": "train-only GCV",
            "permutation_null": {"S": N_PERM_CCA, "seed": PERM_SEED_CCA},
            "basis": "H_CCA = whitened H_low @ B (CCA dirs); "
                     "H_perp = complement via SVD of projector",
            "solver": "eigendecomposition of SPD pencil; no explicit "
                      "inverse",
        }, f, indent=1)

    # ---- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(rho, ada_res["alpha_base"] * (1 + GAMMA * rho ** 2),
               c=np.arange(K), cmap="viridis", s=28)
    ax.set_xlabel("canonical correlation rho_k (train CCA)")
    ax.set_ylabel("alpha_k")
    ax.set_title("CCA-adaptive shrinkage schedule "
                 r"$\alpha_k=\alpha_{base}(1+\rho_k^2)$")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures",
                                 f"canonical_rho_vs_alpha.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    nr = np.array([r["null_mean"] for r in null_rows])
    ns = np.array([r["null_std"] for r in null_rows])
    ax.plot(np.arange(1, K + 1), rho, "o-", ms=4, label="real rho_k")
    ax.errorbar(np.arange(1, K + 1), nr, yerr=ns, fmt="s--", ms=3,
                capsize=2, label="permutation null mean±sd (S=500)")
    ax.set_xlabel("CCA component")
    ax.set_ylabel("canonical correlation")
    ax.set_title("Train-only permutation-null CCA diagnostic")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"cca_permutation_null.{ext}"))
    plt.close(fig)

    names = ["MiniROCKET", "Raw G+H", "G+H_unique", "Uniform gen-ridge",
             "Canonical Ridge"]
    vals = [m0_res["macro_f1"], r2_res["macro_f1"], hu_res["macro_f1"],
            uni_res["macro_f1"], ada_res["macro_f1"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals,
                  color=["tab:gray", "tab:blue", "tab:cyan",
                         "tab:orange", "tab:red"])
    ax.bar_label(bars, fmt="%.4f")
    ax.set_ylabel("test Macro-F1 (single seed-42 evaluation)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("Haptics seed-42 comparison")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"final_comparison.{ext}"))
    plt.close(fig)

    log(f"complete in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
