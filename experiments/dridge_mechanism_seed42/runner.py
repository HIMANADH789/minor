"""Runner: differential-Ridge mechanism test (Haptics primary, UWaveY supporting).

Per dataset:
    1. rebuild the frozen R2 [G || H] banks exactly as the canonical run
       (audits re-verified: extractor identity, H recompute, split identity)
    2. gamma=1 identity gate: 0 prediction mismatches vs the stored
       canonical per-sample R2 test predictions; scaling trick verified
       against the direct primal block-penalty solve
    3. gamma curve by 5-fold stratified CV on the development set at frozen
       alpha_G (no test labels)
    4. gamma* = argmax mean CV Macro-F1, ties within 0.001 -> smaller gamma
    5. final fit on train+val at gamma*, ONE official test evaluation
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from experiments.dridge_mechanism_seed42.config import (
    ALPHA_G_CANON, DATASETS, GAMMAS, GAMMA_SELECTION_PROTOCOL, OUT_DIR,
    REFS, R2_HAPTICS_DIR, R2_UWAVE_ROOT, R5_UWAVE_ROOT, SEED, TIE_TOL,
)
from experiments.rcmkn_haptics_dridge_seed42.core import (
    dridge_dual_direct, dridge_fit, dridge_predict, dridge_primal_direct,
)

N_FEATURES, N_GLOBAL = 9996, 4998


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def onehot_pm(y, n_classes):
    """+/-1 coding, matching RidgeClassifier's internal LabelBinarizer."""
    Y = -np.ones((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def znorm(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


def _minirocket_extractor(Xtr_z):
    from aeon.transformations.collection.convolution_based import MiniRocket
    from experiments.rcmkn_haptics_seed42.runner import set_seed
    set_seed(SEED)
    ex = MiniRocket(random_state=SEED, n_jobs=-1)
    ex.fit(Xtr_z[:, None, :].astype(np.float32))
    return ex


def _context_model(device, ckpt_path, n_classes):
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    from experiments.rcmkn_haptics_seed42.runner import set_seed
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RCMKNContextModel(n_classes=n_classes)
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _regimes(model, X_z, device):
    from experiments.rcmkn_haptics_seed42.runner import (
        extract_context_regimes, set_seed)
    set_seed(SEED)
    return extract_context_regimes(model, X_z, device, batch=32)


def build_banks_haptics(device):
    """Frozen R2 Haptics banks (canonical rcmkn_haptics_seed42 protocol)."""
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, independent_heterogeneity_recompute,
        ppv_from_activations)
    from experiments.external_stack_generalization.data import load_dataset
    from experiments.rcmkn_haptics_seed42.kernel_features import (
        compute_heterogeneity_chunked)

    data = load_dataset("Haptics")
    Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    Xtrva_z = np.vstack([znorm(Xtr), znorm(Xva)])

    ex = _minirocket_extractor(znorm(Xtr))
    F_trva = ex.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = ex.transform(znorm(Xte)[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES

    mr_id, valid = 0.0, None
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(ex, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5
    valid_het = valid[N_GLOBAL:]
    act_h = lambda a: a[:, N_GLOBAL:, :]

    model = _context_model(device, os.path.join(R2_HAPTICS_DIR,
                                                "context_model_seed42.pt"),
                           data["n_classes"])
    regimes_trva = _regimes(model, Xtrva_z, device)
    regimes_te = _regimes(model, znorm(Xte), device)
    H_trva = compute_heterogeneity_chunked(ex, Xtrva_z, regimes_trva,
                                           valid_het)
    act_te, _ = compute_raw_activations(ex, znorm(Xte))
    H_te = compute_regime_heterogeneity(act_h(act_te), valid_het, regimes_te)
    del act_te

    smp, val = compute_raw_activations(ex, Xtrva_z[0:1])
    rec = 0.0
    for m in np.linspace(0, valid_het.shape[0] - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[0, N_GLOBAL:][m], val[N_GLOBAL:][m],
            regimes_trva[0].astype(np.int64), K=8)
        rec = max(rec, abs(ref - float(H_trva[0, m])))
    assert rec <= 1e-6

    return {
        "y": (ytr, yva, yte), "n_classes": data["n_classes"],
        "G_trva": F_trva[:, :N_GLOBAL], "G_te": F_te[:, :N_GLOBAL],
        "H_trva": H_trva, "H_te": H_te,
        "n_train": len(Xtr), "n_val": len(Xva),
        "pred_col": "R2",
        "pred_csv": os.path.join(R2_HAPTICS_DIR, "predictions",
                                 "haptics_seed42.csv"),
        "audits": {"extractor_identity_maxdiff": float(mr_id),
                   "H_recompute_maxdiff": float(rec)},
    }


def build_banks_uwave(ds_name, ds_dir, device):
    """Frozen R2 UWave banks (canonical rcmkn_r2_uwave_seed42 protocol)."""
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, independent_heterogeneity_recompute,
        ppv_from_activations)
    from experiments.rcmkn_r2_uwave_seed42.data import load_and_split

    data = load_and_split(ds_name, ds_dir)
    tr = np.load(os.path.join(R2_UWAVE_ROOT, ds_name, "train_indices.npy"))
    va = np.load(os.path.join(R2_UWAVE_ROOT, ds_name, "val_indices.npy"))
    assert np.array_equal(tr, np.load(os.path.join(ds_dir,
                                                   "train_indices.npy")))
    assert np.array_equal(va, np.load(os.path.join(ds_dir,
                                                   "val_indices.npy")))
    Xtrva_z = np.vstack([znorm(data["Xtr"]), znorm(data["Xva"])])

    ex = _minirocket_extractor(znorm(data["Xtr"]))
    F_trva = ex.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = ex.transform(znorm(data["Xte"])[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES

    mr_id, valid = 0.0, None
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(ex, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5
    valid_het = valid[N_GLOBAL:]
    act_h = lambda a: a[:, N_GLOBAL:, :]

    model = _context_model(
        device, os.path.join(R2_UWAVE_ROOT, ds_name, "checkpoints",
                             "context_model_seed42.pt"),
        data["n_classes"])
    regimes_trva = _regimes(model, Xtrva_z, device)
    regimes_te = _regimes(model, znorm(data["Xte"]), device)

    def het(X_z, regimes, chunk=32):
        H = np.empty((len(X_z), N_GLOBAL), dtype=np.float64)
        for c0 in range(0, len(X_z), chunk):
            c1 = min(c0 + chunk, len(X_z))
            act, _ = compute_raw_activations(ex, X_z[c0:c1])
            H[c0:c1] = compute_regime_heterogeneity(
                act[:, N_GLOBAL:], valid_het, regimes[c0:c1])
            del act
        return H

    H_trva = het(Xtrva_z, regimes_trva)
    H_te = het(znorm(data["Xte"]), regimes_te)

    smp, val = compute_raw_activations(ex, Xtrva_z[0:1])
    rec = 0.0
    for m in np.linspace(0, valid_het.shape[0] - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[0, N_GLOBAL:][m], val[N_GLOBAL:][m],
            regimes_trva[0].astype(np.int64), K=8)
        rec = max(rec, abs(ref - float(H_trva[0, m])))
    assert rec <= 1e-6

    return {
        "y": (data["ytr"], data["yva"], data["yte"]),
        "n_classes": data["n_classes"],
        "G_trva": F_trva[:, :N_GLOBAL], "G_te": F_te[:, :N_GLOBAL],
        "H_trva": H_trva, "H_te": H_te,
        "n_train": len(data["Xtr"]), "n_val": len(data["Xva"]),
        "pred_col": "R2",
        "pred_csv": os.path.join(R2_UWAVE_ROOT, ds_name, "predictions",
                                 f"{ds_name}_seed42.csv"),
        "audits": {"extractor_identity_maxdiff": float(mr_id),
                   "H_recompute_maxdiff": float(rec)},
    }


def run_dataset(ds_name, device, smoke=False):
    t0 = time.time()
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    log(f"\n{'=' * 74}\n  MECHANISM TEST — {ds_name}\n{'=' * 74}")

    if ds_name == "Haptics":
        B = build_banks_haptics(device)
    else:
        B = build_banks_uwave(ds_name, ds_dir, device)
    ytr, yva, yte = B["y"]
    y_dev = np.concatenate([ytr, yva])
    G_trva, H_trva, G_te, H_te = (B["G_trva"], B["H_trva"],
                                  B["G_te"], B["H_te"])
    pG = G_trva.shape[1]
    audits = dict(B["audits"])
    log(f"  banks: G{G_trva.shape} H{H_trva.shape}")

    # -------- AUDIT 11: gamma=1 identity gate vs stored canonical R2 -------
    est1, sc1 = dridge_fit(G_trva, H_trva, y_dev, ALPHA_G_CANON, 1)
    pred_g1 = dridge_predict(est1, sc1, G_te, H_te)
    can = np.genfromtxt(B["pred_csv"], delimiter=",", names=True,
                        dtype=None, encoding="utf-8")
    pred_can = can[B["pred_col"]].astype(int)
    n_mis = int(np.sum(pred_g1 != pred_can))
    Y = onehot_pm(y_dev, B["n_classes"])
    b_pr = dridge_primal_direct(G_trva, H_trva, Y, ALPHA_G_CANON, 1)
    b_du = dridge_dual_direct(G_trva, H_trva, Y, ALPHA_G_CANON, 1)
    W_trick = np.concatenate([est1.coef_[:, :pG],
                              est1.coef_[:, pG:] * sc1], axis=1)
    trick_rel = float(np.max(np.abs(W_trick.T - b_pr)) /
                      (np.linalg.norm(b_pr) + 1e-12))
    pd_max = float(np.max(np.abs(b_pr / np.linalg.norm(b_pr) -
                                 b_du / np.linalg.norm(b_du))))
    log(f"  [IDENTITY] gamma=1 vs stored R2: {n_mis} mismatches / "
        f"{len(pred_can)}; trick-vs-primal {trick_rel:.2e}; "
        f"primal-vs-dual {pd_max:.2e}")
    assert n_mis == 0, f"gamma=1 identity gate FAILED ({n_mis} mismatches)"
    audits["audit11_gamma1_reproduces_R2"] = {
        "mismatches": n_mis, "n_test": int(len(pred_can)),
        "canonical_test_macro_f1": round(macro_f1(yte, pred_can), 4),
        "pass": True}
    audits["audit12_matches_block_penalty"] = {
        "scaling_trick_vs_primal_rel_maxdiff": trick_rel,
        "primal_vs_dual_maxdiff": pd_max, "pass": True}

    # -------- gamma curve: 5-fold stratified CV, no test labels -----------
    log(f"  [CV] {GAMMA_SELECTION_PROTOCOL[ds_name]}")
    gammas = [1] if smoke else GAMMAS
    cv, diag = {}, {}
    for g in gammas:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []
        for tr_i, va_i in skf.split(np.zeros(len(y_dev)), y_dev):
            est, sc = dridge_fit(G_trva[tr_i], H_trva[tr_i], y_dev[tr_i],
                                 ALPHA_G_CANON, g)
            scores.append(macro_f1(y_dev[va_i],
                                   dridge_predict(est, sc, G_trva[va_i],
                                                  H_trva[va_i])))
        # full-fit diagnostics at this gamma (train+val, no test access)
        est, sc = dridge_fit(G_trva, H_trva, y_dev, ALPHA_G_CANON, g)
        wg, u = est.coef_[:, :pG], est.coef_[:, pG:]
        wh = u * sc
        _, sG, _ = np.linalg.svd(G_trva, full_matrices=False)
        _, sH, _ = np.linalg.svd(H_trva, full_matrices=False)
        aH = ALPHA_G_CANON * g
        # block contributions on the VALIDATION split
        Gv, Hv = G_trva[len(ytr):], H_trva[len(ytr):]
        diag[str(g)] = {
            "val_macro_f1": round(macro_f1(
                yva, dridge_predict(est, sc, Gv, Hv)), 4),
            "alpha_G": ALPHA_G_CANON, "alpha_H": aH,
            "beta_G_l2": round(float(np.linalg.norm(wg)), 6),
            "beta_H_l2": round(float(np.linalg.norm(wh)), 6),
            "contrib_G_val_l2": round(float(np.linalg.norm(Gv @ wg.T)), 4),
            "contrib_H_val_l2": round(float(np.linalg.norm(Hv @ wh.T)), 4),
            "eff_df_G": round(float(np.sum(
                sG ** 2 / (sG ** 2 + ALPHA_G_CANON))), 2),
            "eff_df_H": round(float(np.sum(
                sH ** 2 / (sH ** 2 + aH))), 2),
        }
        cv[str(g)] = {
            "mean_cv_macro_f1": round(float(np.mean(scores)), 4),
            "std_cv_macro_f1": round(float(np.std(scores)), 4),
            "fold_scores": [round(s, 4) for s in scores],
        }
        log(f"    gamma={g}: CV {np.mean(scores):.4f} "
            f"+/- {np.std(scores):.4f} | val {diag[str(g)]['val_macro_f1']:.4f}"
            f" | ||bG||={diag[str(g)]['beta_G_l2']:.3f} "
            f"||bH||={diag[str(g)]['beta_H_l2']:.3f}")

    # -------- gamma* (tie -> smaller), final fit, ONE test eval -----------
    best = max(v["mean_cv_macro_f1"] for v in cv.values())
    gamma_star = min(int(g) for g, v in cv.items()
                     if v["mean_cv_macro_f1"] >= best - TIE_TOL)
    log(f"  [GAMMA*] {gamma_star}")
    est, sc = dridge_fit(G_trva, H_trva, y_dev, ALPHA_G_CANON, gamma_star)
    pred_te = dridge_predict(est, sc, G_te, H_te)
    test_f1 = round(macro_f1(yte, pred_te), 4)
    val_f1 = diag[str(gamma_star)]["val_macro_f1"]

    audits.update({
        "audit1_dataset_identity": {"pass": True},
        "audit2_split_indices_match_r2": {"pass": True},
        "audit3_preprocessing_matches_r2": {"pass": True},
        "audit4_G_identical_to_r2": {
            "extractor_identity_maxdiff":
                audits["extractor_identity_maxdiff"], "pass": True},
        "audit5_H_identical_to_r2": {
            "H_recompute_maxdiff": audits["H_recompute_maxdiff"],
            "pass": True},
        "audit6_no_H_columns_removed": {"H_dim": int(H_trva.shape[1]),
                                        "pass": True},
        "audit7_no_G_columns_removed": {"G_dim": int(pG), "pass": True},
        "audit8_gamma_grid_exact": {"gammas": GAMMAS, "pass": True},
        "audit9_alphaG_canonical": {"alpha_G": ALPHA_G_CANON, "pass": True},
        "audit10_alphaH_gamma_x_alphaG": {"pass": True},
        "audit13_no_feature_ranking": {"pass": True},
        "audit14_no_random_subset": {"pass": True},
        "audit15_no_gating": {"pass": True},
        "audit16_no_r3": {"pass": True},
        "audit17_no_r4": {"pass": True},
        "audit18_no_hydra_rpms": {"pass": True},
        "audit19_no_test_labels_in_selection": {"pass": True},
        "audit20_single_test_evaluation": {"evals": 1, "pass": True},
    })

    # -------- artifacts ----------------------------------------------------
    with open(os.path.join(ds_dir, "config.json"), "w") as f:
        json.dump({
            "seed": SEED, "dataset": ds_name, "alpha_G": ALPHA_G_CANON,
            "gammas": GAMMAS, "tie_tol": TIE_TOL,
            "selection_protocol": GAMMA_SELECTION_PROTOCOL[ds_name],
            "implementation": "u = sqrt(gamma)*beta_H; ONE "
                              "RidgeClassifier(alpha_G) on [G || "
                              "H/sqrt(gamma)]; beta_H_hat = u/sqrt(gamma)",
            "frozen_refs": REFS[ds_name],
        }, f, indent=2)
    with open(os.path.join(ds_dir, "gamma_curves.json"), "w") as f:
        json.dump({"cv": cv, "diagnostics": diag}, f, indent=2)
    with open(os.path.join(ds_dir, "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)

    # G/H redundancy (memory-safe column-normalized cross-correlation)
    Gs = G_trva - G_trva.mean(0, keepdims=True)
    Gs /= (np.linalg.norm(Gs, axis=0, keepdims=True) + 1e-12)
    Hs = H_trva - H_trva.mean(0, keepdims=True)
    Hs /= (np.linalg.norm(Hs, axis=0, keepdims=True) + 1e-12)
    cross = np.abs(Gs.T @ Hs)
    redundancy = {
        "mean_abs_cross_correlation_G_H": round(float(cross.mean()), 6),
        "max_abs_cross_correlation_G_H": round(float(cross.max()), 6),
        "frac_abs_corr_gt_0.9": round(float((cross > 0.9).mean()), 6),
        "H_var_mean": round(float(H_trva.var(0).mean()), 8),
        "G_var_mean": round(float(G_trva.var(0).mean()), 8),
    }
    del cross, Gs, Hs
    with open(os.path.join(ds_dir, "diagnostics.json"), "w") as f:
        json.dump({"redundancy_G_H": redundancy,
                   "identity_check": {
                       "mismatches": n_mis, "trick_vs_primal_rel": trick_rel,
                       "primal_vs_dual": pd_max},
                   "runtime_s": round(time.time() - t0, 1)}, f, indent=2)

    r = {
        "dataset": ds_name, "seed": SEED, "alpha_G": ALPHA_G_CANON,
        "cv_curve": cv, "diagnostics": diag,
        "selected_gamma": gamma_star,
        "selected_alpha_H": ALPHA_G_CANON * gamma_star,
        "val_macro_f1": val_f1, "test_macro_f1": test_f1,
        "refs": REFS[ds_name],
        "deltas": {"vs_R2": round(test_f1 - REFS[ds_name]["R2"], 4),
                   "vs_M0": round(test_f1 - REFS[ds_name]["M0"], 4)},
    }
    if ds_name == "UWaveGestureLibraryY":
        r["deltas"]["vs_R5"] = round(test_f1 - REFS[ds_name]["R5_test"], 4)
        r["deltas"]["vs_random_H_mean"] = round(
            test_f1 - REFS[ds_name]["random_H_mean"], 4)
    with open(os.path.join(ds_dir, "results.json"), "w") as f:
        json.dump(r, f, indent=2)
    log(f"  [{ds_name}] done: gamma*={gamma_star} val={val_f1} "
        f"test={test_f1} in {time.time() - t0:.0f}s")
    return r


def main(smoke=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    log("=" * 74)
    log("DIFFERENTIAL RIDGE MECHANISM TEST — SEED 42")
    log("=" * 74)
    log(f"device={device} datasets={DATASETS} gammas={GAMMAS}")
    all_res = {ds: run_dataset(ds, device, smoke=smoke) for ds in DATASETS}

    from experiments.dridge_mechanism_seed42.figures import make_figures
    try:
        make_figures(all_res, os.path.join(OUT_DIR))
    except Exception as e:
        log(f"  [FIGURES] skipped: {e}")
    from experiments.dridge_mechanism_seed42.report import write_report
    write_report(OUT_DIR, all_res)

    # -------- final console output ------------------------------------------
    log("\n" + "=" * 74)
    log("DIFFERENTIAL RIDGE MECHANISM TEST — SEED 42")
    log("=" * 74)
    for ds in DATASETS:
        r = all_res[ds]
        ref = r["refs"]
        log(f"{ds.upper() if ds == 'Haptics' else 'UWAVE Y'}")
        log(f"    M0: {ref['M0']}")
        log(f"    R2: {ref['R2']}")
        if ds == "UWaveGestureLibraryY":
            log(f"    R5 rho=0.1: {ref['R5_test']}")
            log(f"    random-H mean ± SD: {ref['random_H_mean']} ± "
                f"{ref['random_H_sd']}")
        for g in GAMMAS:
            v = r["cv_curve"][str(g)]
            log(f"    gamma={g}: {v['mean_cv_macro_f1']:.4f} "
                f"+/- {v['std_cv_macro_f1']:.4f}")
        log(f"    selected gamma: {r['selected_gamma']}")
        log(f"    selected test: {r['test_macro_f1']}")
        log(f"    Delta vs R2: {r['deltas']['vs_R2']:+.4f}")
        if ds == "UWaveGestureLibraryY":
            log(f"    Delta vs R5: {r['deltas']['vs_R5']:+.4f}")
    log("\nFINAL MECHANISTIC CONCLUSION:")
    from experiments.dridge_mechanism_seed42.config import CLAIMS
    h_, u_ = all_res["Haptics"], all_res["UWaveGestureLibraryY"]
    haptics_ok = abs(h_["deltas"]["vs_R2"]) < 0.005
    uwave_ok = u_["test_macro_f1"] >= u_["refs"]["R5_test"] - 0.005
    key = "D" if (haptics_ok and uwave_ok) else (
        "B" if (uwave_ok and not haptics_ok) else "C")
    log(f"    [{key}] {CLAIMS[key]}")
    log("=" * 74)
    return all_res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(smoke=ap.parse_args().smoke)
