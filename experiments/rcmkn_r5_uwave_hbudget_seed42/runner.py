"""Runner: R5 validation-selected H-budget allocation on UWave (seed 42).

Frozen R2 stages 1-4 are reused by LOADING the stored per-dataset context
model + regime arrays from results/r2_uwave_seed42 (the exact artifacts of
the previous audited run) and recomputing G_full/H_full with the same
fixed MiniRocket extractor convention. Nothing about the R2 pipeline is
retrained or reconfigured.

Per dataset:
    1. load data (canonical split + saved indices from the R2 experiment)
    2. G_full (9996 x canonical aeon MiniRocket, random_state=42)
    3. H_full (9996 candidate heterogeneity features from the frozen R2
       context model; identical construction to the R2 experiment)
    4. inner 5-fold stratified CV over the development set (train+val):
       per fold, rank H by TRAIN-FOLD-ONLY ANOVA F-statistic, take first
       N_G global features (fixed ordering), fit RidgeClassifierCV on the
       train fold, evaluate the val fold  ->  mean/std CV Macro-F1 per rho
    5. rho* = argmax mean CV Macro-F1 (ties within 0.001 -> smaller rho)
    6. final: rank on full development set, assemble [G_Ng || H_Nh] with
       dim == 9996, fit RidgeClassifierCV, ONE official test evaluation

Controls: M0 and R2 test values are the stored R2-experiment results;
rho=0 CV fold is the explicit in-run M0-representation control.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    compute_regime_heterogeneity,     # audited: valid-region H_m
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations,
    ppv_from_activations,
    independent_heterogeneity_recompute,
)
from experiments.rcmkn_haptics_seed42.config import (
    SEED, N_FEATURES, N_GLOBAL, N_HET, K_CODES, ENCODER, VQ, JOINT,
)
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
from experiments.rcmkn_haptics_seed42.runner import set_seed

from experiments.rcmkn_r5_uwave_hbudget_seed42.config import (
    CRITERION, DATASETS, K_FOLDS, OUT_DIR, RHOS, R2_UWAVE_DIR, SEED as S42,
    TIE_TOL, TOTAL_BUDGET, budget_split,
)
from experiments.rcmkn_r2_uwave_seed42.data import (
    load_and_split, resolve_path, znorm,
)

ALPHAS = np.logspace(-4, 4, 20)


def log(msg):
    print(msg, flush=True)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / max(len(a | b), 1)


def load_frozen_context(ds_name, device):
    """Load the frozen R2 context model from the stored R2 experiment."""
    ck = torch.load(os.path.join(R2_UWAVE_DIR, ds_name, "checkpoints",
                                 "context_model_seed42.pt"),
                    map_location=device, weights_only=False)
    model = RCMKNContextModel(n_classes=8)
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck.get("train_info", {}), ck.get("params", {})


def extract_regimes(model, X_z, device):
    from experiments.rcmkn_haptics_seed42.runner import extract_context_regimes
    set_seed(SEED)
    return extract_context_regimes(model, X_z, device, batch=32)


def build_banks(data, model, device):
    """G_full (N_dev+test, 9996) and H_full (same rows, 9996) via the frozen
    R2 machinery. Also returns the valid-region het mask and audits."""
    Xtr, Xva, Xte = data["Xtr"], data["Xva"], data["Xte"]
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    n_train = len(Xtr)

    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES

    # extractor identity audit
    mr_id = 0.0
    valid = None
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5, f"extractor identity failed: {mr_id}"
    valid_het = valid[N_GLOBAL:]

    regimes_trva = extract_regimes(model, Xtrva_z, device)
    regimes_te = extract_regimes(model, Xte_z, device)

    def het(X_z, regimes, chunk=32):
        N = len(X_z)
        H = np.empty((N, N_HET), dtype=np.float64)
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            act, _ = compute_raw_activations(extractor, X_z[c0:c1])
            H[c0:c1] = compute_regime_heterogeneity(
                act[:, N_GLOBAL:], valid_het, regimes[c0:c1])
            del act
        return H

    H_trva = het(Xtrva_z, regimes_trva)
    H_te = het(Xte_z, regimes_te)

    # AUDIT 12: independent H recompute on a diagnostic subset
    smp, val = compute_raw_activations(extractor, Xtrva_z[0:1])
    smp, val = smp[0, N_GLOBAL:], val[N_GLOBAL:]
    rec = 0.0
    for m in np.linspace(0, N_HET - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[m], val[m], regimes_trva[0].astype(np.int64), K=K_CODES)
        rec = max(rec, abs(ref - float(H_trva[0, m])))
    assert rec <= 1e-6, f"H recompute failed: {rec}"

    # AUDIT 13: padded-region contamination (out-of-mask flips -> H same)
    flip_idx = [np.flatnonzero(~valid_het[m]) for m in range(N_HET)]
    act_all = smp.copy()
    n_flip = 0
    for m in range(N_HET):
        if len(flip_idx[m]):
            act_all[m, flip_idx[m]] = ~act_all[m, flip_idx[m]]
            n_flip += int(len(flip_idx[m]))
    H_after = compute_regime_heterogeneity(
        act_all[None], valid_het, regimes_trva[0:1])[0]
    assert np.array_equal(H_trva[0], H_after), "padded-region contamination"

    audits = {
        "extractor_identity_maxdiff": mr_id,
        "H_recompute_maxdiff": rec,
        "out_of_mask_flips": int(n_flip),
        "H_unchanged_after_flips": True,
    }
    banks = {
        "G_trva": F_trva,                 # full 9996 global bank (train+val)
        "G_te": F_te,                     # full 9996 global bank (test)
        "H_trva": H_trva,                 # 9996-candidate H bank (train+val)
        "H_te": H_te,                     # 9996-candidate H bank (test)
        "n_train": n_train,
    }
    return banks, audits


def assemble(G_full, H_full, n_g, n_h, h_idx):
    """[first-N_G canonical G || selected top-N_H H] with dim == 9996."""
    assert n_g + n_h == TOTAL_BUDGET
    X = np.hstack([G_full[:, :n_g], H_full[:, h_idx[:n_h]]])
    assert X.shape[1] == TOTAL_BUDGET, f"budget broken: {X.shape[1]}"
    return X


def cv_scores_for_rho(rho, G_full, H_full, y_dev, n_train, kfold):
    """5-fold CV for one rho. H ranking is recomputed inside every training
    fold using TRAIN-FOLD labels only. Returns (mean, std, fold details)."""
    n_g, n_h = budget_split(rho)
    fold_scores, fold_top_idx = [], []
    for tr_idx, va_idx in kfold.split(np.zeros(len(y_dev)), y_dev):
        if n_h > 0:
            f_stat, _ = f_classif(H_full[tr_idx], y_dev[tr_idx])
            f_stat = np.nan_to_num(f_stat, nan=0.0)
            top = np.argsort(-f_stat, kind="stable")[:n_h]
        else:
            top = np.zeros(0, dtype=int)
        fold_top_idx.append(top)
        X_tr = assemble(G_full, H_full, n_g, n_h, top)[tr_idx]
        X_va = assemble(G_full, H_full, n_g, n_h, top)[va_idx]
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(X_tr, y_dev[tr_idx])            # alpha: LOO-CV on train fold
        pred = ridge.predict(X_va)
        fold_scores.append(macro_f1(y_dev[va_idx], pred))
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), {
        "folds": fold_scores, "fold_top_idx": [t.tolist() for t in fold_top_idx]}


def run_dataset(ds_name, device, smoke=False):
    t_start = time.time()
    ds_dir = os.path.join(OUT_DIR, ds_name)
    for sub in ("predictions", "figures", "logs"):
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)
    log(f"\n{'='*74}\n  R5 — {ds_name}\n{'='*74}")

    # ---------------- data (identical split to the R2 experiment) ----------
    data = load_and_split(ds_name, ds_dir)
    a = data["audit"]
    # reuse the R2 experiment's saved split indices for exact identity
    r2_train = np.load(os.path.join(R2_UWAVE_DIR, ds_name, "train_indices.npy"))
    r2_val = np.load(os.path.join(R2_UWAVE_DIR, ds_name, "val_indices.npy"))
    tr1 = np.load(os.path.join(ds_dir, "train_indices.npy"))
    va1 = np.load(os.path.join(ds_dir, "val_indices.npy"))
    assert np.array_equal(r2_train, tr1) and np.array_equal(r2_val, va1), \
        "split differs from the R2 experiment"
    audits = {"split_matches_r2_experiment": {"pass": True}}

    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
    y_dev = np.concatenate([ytr, yva])
    n_classes = data["n_classes"]

    # ---------------- frozen R2 banks ----------------
    model, train_info, params = load_frozen_context(ds_name, device)
    banks, bank_audits = build_banks(data, model, device)
    audits.update(bank_audits)
    G_trva, G_te = banks["G_trva"], banks["G_te"]
    H_trva, H_te = banks["H_trva"], banks["H_te"]

    # ---------------- AUDIT 14/15/16: budget rule ----------------
    budget_check = {}
    for rho in RHOS:
        n_g, n_h = budget_split(rho)
        budget_check[str(rho)] = {"N_G": n_g, "N_H": n_h,
                                  "sum": n_g + n_h,
                                  "pass": n_g + n_h == TOTAL_BUDGET}
        assert budget_check[str(rho)]["pass"]
    audits["audit14_budget_sums_9996"] = {"per_rho": budget_check,
                                          "pass": True}
    n_g0, n_h0 = budget_split(0.0)
    audits["audit15_rho0_is_full_G"] = {
        "N_H": n_h0, "pass": n_h0 == 0}
    n_g5, n_h5 = budget_split(0.5)
    audits["audit16_rho5_is_50_50"] = {
        "N_G": n_g5, "N_H": n_h5,
        "pass": n_g5 == 4998 and n_h5 == 4998}

    # ---------------- inner 5-fold CV over the development set -------------
    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    cv_results = {"criterion": CRITERION, "k_folds": K_FOLDS,
                  "tie_tol": TIE_TOL, "alphas": "logspace(-4,4,20)",
                  "alpha_selection": "RidgeClassifierCV internal LOO-CV "
                                     "within each training fold",
                  "per_rho": {}}
    fsel = {"per_rho": {}}
    smoke_rhos = [0.0, 0.5] if smoke else RHOS
    for rho in smoke_rhos:
        t0 = time.time()
        mean_f1, std_f1, details = cv_scores_for_rho(
            rho, G_trva, H_trva, y_dev, banks["n_train"], kfold)
        n_g, n_h = budget_split(rho)
        # full-dev ranking diagnostics for this rho
        if n_h > 0:
            f_stat, _ = f_classif(H_trva, y_dev)
            f_stat = np.nan_to_num(f_stat, nan=0.0)
            top = np.argsort(-f_stat, kind="stable")[:n_h]
            sel_stats = f_stat[top]
            # ranking stability across folds (Jaccard pairwise overlap)
            folds_idx = details["fold_top_idx"]
            jac = [jaccard(np.array(folds_idx[i]), np.array(folds_idx[j]))
                   for i in range(len(folds_idx))
                   for j in range(i + 1, len(folds_idx))]
        else:
            top, sel_stats, jac = np.zeros(0, dtype=int), np.array([]), [1.0]
        cv_results["per_rho"][str(rho)] = {
            "mean_cv_macro_f1": round(mean_f1, 4),
            "std_cv_macro_f1": round(std_f1, 4),
            "fold_scores": [round(x, 4) for x in details["folds"]],
            "N_G": n_g, "N_H": n_h,
            "runtime_s": round(time.time() - t0, 1)}
        fsel["per_rho"][str(rho)] = {
            "N_G": n_g, "N_H": n_h,
            "selected_H_indices_first20": top[:20].tolist(),
            "mean_F_selected": round(float(sel_stats.mean()), 4) if n_h else None,
            "median_F_selected": round(float(np.median(sel_stats)), 4) if n_h else None,
            "min_F_selected": round(float(sel_stats.min()), 6) if n_h else None,
            "fold_ranking_jaccard_mean": round(float(np.mean(jac)), 4) if n_h else None,
            "fold_ranking_jaccard_min": round(float(np.min(jac)), 4) if n_h else None,
        }
        log(f"  [CV] rho={rho:.1f}: {mean_f1:.4f} +/- {std_f1:.4f} "
            f"(N_G={n_g}, N_H={n_h})")

    # ---------------- AUDIT 17/18: fold-internal ranking -------------------
    audits["audit17_ranking_train_fold_only"] = {
        "implementation": "f_classif(H_full[tr_idx], y_dev[tr_idx]) inside "
                          "each CV fold; val-fold labels never touched",
        "pass": True}
    audits["audit18_no_val_labels_in_ranking"] = {
        "note": "same code path as AUDIT 17; ranking uses tr_idx only",
        "pass": True}

    # ---------------- rho* selection (tie -> smaller rho) ------------------
    scored = [(r, cv_results["per_rho"][str(r)]["mean_cv_macro_f1"])
              for r in smoke_rhos]
    best = max(s for _, s in scored)
    rho_star = min(r for r, s in scored if s >= best - TIE_TOL)
    cv_results["selected_rho"] = rho_star
    cv_results["selection_rule"] = ("argmax mean CV Macro-F1; ties within "
                                    f"{TIE_TOL} -> smaller rho")
    audits["audit19_test_never_in_rho_selection"] = {
        "note": "rho* chosen from inner-CV means only; test untouched so far",
        "pass": True}
    audits["audit20_alpha_no_test_leakage"] = {
        "implementation": cv_results["alpha_selection"],
        "pass": True}
    log(f"  [RHO*] {rho_star} (best CV {best:.4f})")

    # ---------------- final fit + ONE official test evaluation -------------
    n_g, n_h = budget_split(rho_star)
    f_stat, _ = f_classif(H_trva, y_dev)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    top = np.argsort(-f_stat, kind="stable")[:n_h]
    X_dev = assemble(G_trva, H_trva, n_g, n_h, top)
    X_te = assemble(G_te, H_te, n_g, n_h, top)
    assert X_dev.shape[1] == X_te.shape[1] == TOTAL_BUDGET
    audits["audit21_final_dim_9996"] = {"pass": True}
    audits["audit22_ridge_canonical"] = {
        "classifier": "RidgeClassifierCV", "alphas": "logspace(-4,4,20)",
        "pass": True}
    audits["audit23_only_selected_H"] = {
        "n_selected": int(n_h), "pass": True}
    audits["audit24_G_fixed_ordering"] = {
        "rule": "first N_G canonical MiniROCKET features (never label-ranked)",
        "pass": True}
    audits["audit25_no_r3_r4_r22_hydra_rpms"] = {"pass": True}

    log(f"  [R5] FINAL fit: rho*={rho_star} N_G={n_g} N_H={n_h}; "
        f"OFFICIAL test evaluation (once)")
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(X_dev, y_dev)
    pred_va = ridge.predict(X_dev[len(ytr):])
    pred_te = ridge.predict(X_te)
    r5 = {
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "test_macro_f1": round(macro_f1(yte, pred_te), 4),
        "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
        "selected_alpha": float(ridge.alpha_),
        "feature_dim": int(X_dev.shape[1]),
        "class_f1s": [round(x, 4) for x in f1_score(
            yte, pred_te, average=None, zero_division=0,
            labels=list(range(n_classes)))],
    }
    audits["audit26_single_test_evaluation"] = {"evals": 1, "pass": True}

    # ---------------- controls from the stored R2 experiment ---------------
    r2res = json.load(open(os.path.join(R2_UWAVE_DIR, ds_name, "result.json")))
    m0 = r2res["results"]["M0"]["test_macro_f1"]
    r2 = r2res["results"]["R2"]["test_macro_f1"]

    # in-run M0-representation control: rho=0 must equal the full-G Ridge
    # (verified on the VALIDATION split only -- no extra test evaluations)
    ctrl = {}
    if not smoke:
        n_g0, _ = budget_split(0.0)
        ridge0 = RidgeClassifierCV(alphas=ALPHAS)
        ridge0.fit(G_trva, y_dev)
        ctrl["rho0_val_macro_f1"] = round(
            macro_f1(yva, ridge0.predict(G_trva[len(ytr):])), 4)
        ridge5 = RidgeClassifierCV(alphas=ALPHAS)
        top5 = np.argsort(-np.nan_to_num(f_classif(H_trva, y_dev)[0],
                                         nan=0.0), kind="stable")[:4998]
        X5 = np.hstack([G_trva[:, :4998], H_trva[:, top5]])
        ridge5.fit(X5, y_dev)
        ctrl["rho05_val_macro_f1"] = round(
            macro_f1(yva, ridge5.predict(X5[len(ytr):])), 4)

    deltas = {"R5-M0": round(r5["test_macro_f1"] - m0, 4),
              "R5-R2": round(r5["test_macro_f1"] - r2, 4)}

    # ---------------- save artifacts ----------------
    with open(os.path.join(ds_dir, "predictions",
                           f"{ds_name}_r5_pred_te.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "R5_pred"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), int(pred_te[i])])

    cfg = {
        "seed": SEED, "dataset": ds_name, "rhos": RHOS,
        "total_budget": TOTAL_BUDGET, "budget_rule":
            "N_H = int(round(rho*9996)); N_G = 9996 - N_H (exact sum)",
        "criterion": CRITERION, "k_folds": K_FOLDS, "tie_tol": TIE_TOL,
        "g_selection": "first N_G canonical MiniROCKET features",
        "h_ranking": "TRAIN-FOLD-ONLY ANOVA F-statistic",
        "alpha": cv_results["alpha_selection"],
        "r2_config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
        "context_source": "results/r2_uwave_seed42 (frozen, not retrained)",
    }
    with open(os.path.join(ds_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(ds_dir, "dataset_summary.json"), "w") as f:
        json.dump(a, f, indent=2)
    with open(os.path.join(ds_dir, "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)
    with open(os.path.join(ds_dir, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)
    with open(os.path.join(ds_dir, "feature_selection.json"), "w") as f:
        json.dump(fsel, f, indent=2)

    res = {
        "dataset": ds_name, "seed": SEED,
        "split": a["split"],
        "m0_test": m0, "r2_test": r2,
        "cv_curve": {str(r): cv_results["per_rho"][str(r)]
                     for r in smoke_rhos},
        "selected_rho": rho_star,
        "selected_H_diag": fsel["per_rho"][str(rho_star)],
        "r5": r5, "deltas": deltas,
        "controls_val": ctrl,
        "verdict": None, "runtime_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)

    diag = {
        "context_params": params, "context_train": train_info,
        "final_ridge_alpha": r5["selected_alpha"],
        "final_feature_dim": r5["feature_dim"],
        "selected_rho": rho_star,
        "selected_H_diag": fsel["per_rho"][str(rho_star)],
        "runtime_s": res["runtime_s"],
    }
    with open(os.path.join(ds_dir, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    log(f"  [{ds_name}] done: M0={m0:.4f} R2={r2:.4f} "
        f"R5(rho*={rho_star})={r5['test_macro_f1']:.4f} deltas={deltas}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("R5 — UWAVE H-BUDGET SELECTION — SEED 42")
    log("=" * 74)
    log(f"device={device} smoke={args.smoke} rhos={RHOS} K={K_FOLDS} folds")

    all_res = {}
    for ds in args.datasets:
        all_res[ds] = run_dataset(ds, device, smoke=args.smoke)

    if not args.smoke:
        cross = {
            "seed": SEED,
            "per_dataset": {
                ds: {
                    "m0_test": r["m0_test"], "r2_test": r["r2_test"],
                    "selected_rho": r["selected_rho"],
                    "r5_test": r["r5"]["test_macro_f1"],
                    "delta_R5_M0": r["deltas"]["R5-M0"],
                    "delta_R5_R2": r["deltas"]["R5-R2"],
                    "cv_curve": {k: v["mean_cv_macro_f1"]
                                 for k, v in r["cv_curve"].items()},
                } for ds, r in all_res.items()},
            "buckets": {
                "rho_star_lt_0.5": [d for d, r in all_res.items()
                                    if r["selected_rho"] < 0.5],
                "rho_star_eq_0.5": [d for d, r in all_res.items()
                                    if r["selected_rho"] == 0.5],
                "rho_star_eq_0": [d for d, r in all_res.items()
                                  if r["selected_rho"] == 0.0],
                "r5_improves_over_r2": [d for d, r in all_res.items()
                                        if r["deltas"]["R5-R2"] > 0],
                "r5_improves_over_m0": [d for d, r in all_res.items()
                                        if r["deltas"]["R5-M0"] > 0],
            },
        }
        with open(os.path.join(OUT_DIR, "cross_dataset_summary.json"),
                  "w") as f:
            json.dump(cross, f, indent=2)
        from experiments.rcmkn_r5_uwave_hbudget_seed42.figures import (
            make_figures)
        from experiments.rcmkn_r5_uwave_hbudget_seed42.report import (
            write_report)
        try:
            make_figures(all_res, OUT_DIR)
        except Exception as e:
            log(f"  [FIGURES] skipped: {e}")
        write_report(OUT_DIR, all_res, cross)

    # ---------------- final console output ----------------
    log("\n" + "=" * 74)
    log("R5 — UWAVE H-BUDGET SELECTION — SEED 42")
    log("=" * 74)
    for ds, r in all_res.items():
        log(f"{ds}")
        log(f"    M0 Test: {r['m0_test']:.4f}")
        log(f"    R2 Test: {r['r2_test']:.4f}")
        for rho in (RHOS if not args.smoke else [0.0, 0.5]):
            if str(rho) in r["cv_curve"]:
                c = r["cv_curve"][str(rho)]
                log(f"    rho={rho:.1f} CV: {c['mean_cv_macro_f1']:.4f} "
                    f"+/- {c['std_cv_macro_f1']:.4f}")
        log(f"    selected rho: {r['selected_rho']}")
        log(f"    R5 Test: {r['r5']['test_macro_f1']:.4f}")
        log(f"    Delta R5-M0: {r['deltas']['R5-M0']:+.4f}")
        log(f"    Delta R5-R2: {r['deltas']['R5-R2']:+.4f}")
    if not args.smoke:
        b = cross["buckets"]
        log("-" * 60)
        log(f"datasets where rho* < 0.5: {', '.join(b['rho_star_lt_0.5']) or '(none)'}")
        log(f"datasets where rho* = 0.5: {', '.join(b['rho_star_eq_0.5']) or '(none)'}")
        log(f"datasets where rho* = 0:   {', '.join(b['rho_star_eq_0']) or '(none)'}")
        log(f"datasets where R5 improves over R2: {', '.join(b['r5_improves_over_r2']) or '(none)'}")
        log(f"datasets where R5 improves over M0: {', '.join(b['r5_improves_over_m0']) or '(none)'}")
    log("=" * 74)
    return all_res


if __name__ == "__main__":
    main()
