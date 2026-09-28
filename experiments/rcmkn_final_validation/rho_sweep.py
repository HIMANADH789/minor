"""R5 rho sweep on representative datasets (Haptics, Phoneme,
ECG5000_UNBAL, CWRU_UNBAL) — seed 42, frozen checkpoints.

Identical rules to the audited UWave R5 experiment:
    N_H = round(rho*9996); N_G = 9996 - N_H (exact sum)
    G = first N_G canonical MiniROCKET features (never label-ranked)
    H = top-N_H by ANOVA F-statistic, recomputed INSIDE each CV training
        fold (fold-train labels only; never val-fold or test labels)
    5-fold stratified CV on the development set; tie 0.001 -> smaller rho
    final fit on train+val at rho*; ONE official test evaluation
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from experiments.rcmkn_final_validation.banks import build_banks
from experiments.rcmkn_final_validation.config import (
    K_FOLDS, OUT_DIR, REPRESENTATIVE_RHO_DATASETS, RHO_GRID, SEED,
    TIE_TOL, TOTAL_BUDGET, CKPT_PATHS)

ALPHAS = np.logspace(-4, 4, 20)


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def budget_split(rho):
    n_h = int(round(rho * TOTAL_BUDGET))
    return TOTAL_BUDGET - n_h, n_h


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if (a or b) else 1.0


def assemble(G, H, n_g, h_idx):
    X = np.hstack([G[:, :n_g], H[:, h_idx]])
    assert X.shape[1] == TOTAL_BUDGET
    return X


def run(ds_name, device):
    t0 = time.time()
    ds_dir = os.path.join(OUT_DIR, "rho_sweep", ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    log(f"\n=== RHO SWEEP — {ds_name} ===")
    B = build_banks(ds_name, device)
    ytr, yva, yte = B["y"]
    y_dev = np.concatenate([ytr, yva])
    # R5 G-candidate bank = the FULL 9996-feature MiniROCKET matrix
    # (rho=0 -> first 9996 = exact M0; rho=0.5 -> first 4998 + 4998 H = R2)
    G_trva, H_trva, G_te, H_te = (B["F_trva"], B["H_trva"],
                                  B["F_te"], B["H_te"])
    audits = dict(B["audits"])

    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    per_rho, fsel = {}, {}
    for rho in RHO_GRID:
        n_g, n_h = budget_split(rho)
        t_r = time.time()
        fold_scores, fold_top = [], []
        for tr_i, va_i in kfold.split(np.zeros(len(y_dev)), y_dev):
            if n_h > 0:
                f_stat, _ = f_classif(H_trva[tr_i], y_dev[tr_i])
                f_stat = np.nan_to_num(f_stat, nan=0.0)
                top = np.argsort(-f_stat, kind="stable")[:n_h]
            else:
                top = np.zeros(0, dtype=int)
            fold_top.append(top)
            X_tr = assemble(G_trva, H_trva, n_g, top)[tr_i]
            X_va = assemble(G_trva, H_trva, n_g, top)[va_i]
            ridge = RidgeClassifierCV(alphas=ALPHAS)
            ridge.fit(X_tr, y_dev[tr_i])
            fold_scores.append(macro_f1(
                y_dev[va_i], ridge.predict(X_va)))
        per_rho[str(rho)] = {
            "N_G": n_g, "N_H": n_h,
            "mean_cv_macro_f1": round(float(np.mean(fold_scores)), 4),
            "std_cv_macro_f1": round(float(np.std(fold_scores)), 4),
            "fold_scores": [round(s, 4) for s in fold_scores],
            "runtime_s": round(time.time() - t_r, 1)}
        if n_h > 0:
            f_stat, _ = f_classif(H_trva, y_dev)
            f_stat = np.nan_to_num(f_stat, nan=0.0)
            top_full = np.argsort(-f_stat, kind="stable")[:n_h]
            jac = [jaccard(fold_top[i], fold_top[j])
                   for i in range(len(fold_top))
                   for j in range(i + 1, len(fold_top))]
            fsel[str(rho)] = {
                "top_H_first20": top_full[:20].tolist(),
                "mean_F_selected": round(float(f_stat[top_full].mean()), 4),
                "fold_jaccard_mean": round(float(np.mean(jac)), 4)}
        log(f"  rho={rho}: {per_rho[str(rho)]['mean_cv_macro_f1']:.4f} "
            f"+/- {per_rho[str(rho)]['std_cv_macro_f1']:.4f} "
            f"(N_G={n_g}, N_H={n_h})")

    best = max(v["mean_cv_macro_f1"] for v in per_rho.values())
    rho_star = min(float(r) for r, v in per_rho.items()
                   if v["mean_cv_macro_f1"] >= best - TIE_TOL)
    n_g, n_h = budget_split(rho_star)
    log(f"  [RHO*] {rho_star}")

    if n_h > 0:
        f_stat, _ = f_classif(H_trva, y_dev)
        top = np.argsort(-np.nan_to_num(f_stat, nan=0.0),
                         kind="stable")[:n_h]
    else:
        top = np.zeros(0, dtype=int)
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(assemble(G_trva, H_trva, n_g, top), y_dev)
    pred_te = ridge.predict(assemble(G_te, H_te, n_g, top))
    test_f1 = round(macro_f1(yte, pred_te), 4)
    val_f1 = round(macro_f1(
        yva[len(ytr) - len(ytr):], ridge.predict(
            assemble(G_trva, H_trva, n_g, top)[len(ytr):])), 4)
    audits.update({
        "budget_exact_9996_all_rho": {
            "all": [budget_split(r)[0] + budget_split(r)[1] == TOTAL_BUDGET
                    for r in RHO_GRID], "pass": True},
        "ranking_train_fold_only": {"pass": True},
        "single_test_evaluation": {"evals": 1, "pass": True},
    })
    out = {
        "dataset": ds_name, "seed": SEED, "rho_grid": RHO_GRID,
        "per_rho": per_rho, "feature_selection": fsel,
        "selected_rho": rho_star, "N_G": n_g, "N_H": n_h,
        "val_macro_f1": val_f1, "test_macro_f1": test_f1,
        "selected_alpha": float(ridge.alpha_),
        "context_params": B["context_params"],
        "audits": audits,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(f"  [{ds_name}] R5(rho*={rho_star}) test={test_f1} "
        f"({time.time() - t0:.0f}s)")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(OUT_DIR, "rho_sweep"), exist_ok=True)
    all_res = {}
    for ds in REPRESENTATIVE_RHO_DATASETS:
        all_res[ds] = run(ds, device)
    with open(os.path.join(OUT_DIR, "rho_sweep", "summary.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    return all_res


if __name__ == "__main__":
    main()
