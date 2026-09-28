"""Secondary multi-seed stability: UWaveGestureLibraryY, seeds 42/43/44.

M0: deterministic (fixed MiniROCKET rs=42 + Ridge) -> one run for all seeds.
R2: seed 42 = canonical stored (0.7551); seeds 43/44 = retrained context
    (exact Haptics schedule), one test eval each.
R5: per-seed rho selection (dev-set 5-fold CV) + one test eval per seed.
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

from experiments.rcmkn_final_validation.banks import build_banks, znorm
from experiments.rcmkn_final_validation.config import (
    K_FOLDS, OUT_DIR, RHO_GRID, SEED, TIE_TOL, TOTAL_BUDGET)

ALPHAS = np.logspace(-4, 4, 20)
DS = "UWaveGestureLibraryY"


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def budget_split(rho):
    n_h = int(round(rho * TOTAL_BUDGET))
    return TOTAL_BUDGET - n_h, n_h


def assemble(G, H, n_g, h_idx):
    X = np.hstack([G[:, :n_g], H[:, h_idx]])
    assert X.shape[1] == TOTAL_BUDGET
    return X


def context_for_seed(seed, Xtr_z, ytr, Xva_z, yva, device, n_classes):
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    from experiments.rcmkn_haptics_seed42.runner import (
        set_seed, train_context_model, extract_context_regimes)
    set_seed(seed)
    model = RCMKNContextModel(n_classes=n_classes).to(device)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    set_seed(seed)
    return model, extract_context_regimes(model, Xtrva_z, device, batch=32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = build_banks(DS, device, ds_dir=os.path.join(
        OUT_DIR, "_uwavey_split"))
    ytr, yva, yte = B["y"]
    y_dev = np.concatenate([ytr, yva])
    n_classes = B["n_classes"]
    rows = []
    log(f"[{DS}] banks ready: {B['F_trva'].shape} {B['H_trva'].shape}")

    # M0 (deterministic)
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(B["F_trva"], y_dev)
    m0_val = round(macro_f1(yva, ridge.predict(B["F_trva"][len(ytr):])), 4)
    m0_test = round(macro_f1(yte, ridge.predict(B["F_te"])), 4)
    log(f"  [M0] val={m0_val} test={m0_test} alpha={ridge.alpha_:.4g}")

    # seed-42 R2/R5 context is the canonical stored one: reuse stored banks
    # via fresh pipeline with seed 42 (protocol-faithful)
    for seed in (42, 43, 44):
        t0 = time.time()
        Xtr = B.get("_Xtr")
        model, reg_trva = context_for_seed(
            seed, B["_Xtr_z"], ytr, B["_Xva_z"], yva, device, n_classes)
        from experiments.rcmkn_haptics_seed42.runner import (
            extract_context_regimes, set_seed)
        set_seed(seed)
        reg_te = extract_context_regimes(model, B["_Xte_z"], device,
                                         batch=32)

        def het(X_z, regimes, chunk=32):
            from experiments.drtn_conditioned_minirocket_transfer_seed42.core \
                import compute_raw_activations
            from experiments.drtn_conditioned_minirocket_haptics_3seed.runner \
                import compute_regime_heterogeneity
            H = np.empty((len(X_z), 4998), dtype=np.float64)
            for c0 in range(0, len(X_z), chunk):
                c1 = min(c0 + chunk, len(X_z))
                act, _ = compute_raw_activations(extractor := B["_extractor"],
                                                 X_z[c0:c1])
                H[c0:c1] = compute_regime_heterogeneity(
                    act[:, 4998:], B["_valid_het"], regimes[c0:c1])
                del act
            return H

        H_trva = het(B["_Xtrva_z"], reg_trva)
        H_te = het(B["_Xte_z"], reg_te)

        # R2 (skip test for seed 42 -- canonical stored result reused)
        ridge2 = RidgeClassifierCV(alphas=ALPHAS)
        ridge2.fit(np.hstack([B["G_trva"], H_trva]), y_dev)
        r2_val = round(macro_f1(
            yva, ridge2.predict(np.hstack([B["G_trva"], H_trva])[len(ytr):])),
            4)
        if seed == 42:
            r2_test = 0.7551
            r2_src = "canonical_stored"
        else:
            r2_test = round(macro_f1(
                yte, ridge2.predict(np.hstack([B["G_te"], H_te]))), 4)
            r2_src = "fresh"
        log(f"  seed{seed} R2: val={r2_val} test={r2_test} ({r2_src})")
        rows.append({"dataset": DS, "model": "R2", "seed": seed,
                     "val_macro_f1": r2_val, "test_macro_f1": r2_test,
                     "selected_alpha": float(ridge2.alpha_), "rho": "",
                     "N_G": 4998, "N_H": 4998, "source": r2_src,
                     "runtime_s": round(time.time() - t0, 1)})

        # R5
        kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                                random_state=42)
        per_rho = {}
        for rho in RHO_GRID:
            n_g, n_h = budget_split(rho)
            scores = []
            for tr_i, va_i in kfold.split(np.zeros(len(y_dev)), y_dev):
                if n_h > 0:
                    f_stat, _ = f_classif(H_trva[tr_i], y_dev[tr_i])
                    top = np.argsort(-np.nan_to_num(f_stat, nan=0.0),
                                     kind="stable")[:n_h]
                else:
                    top = np.zeros(0, dtype=int)
                r5 = RidgeClassifierCV(alphas=ALPHAS)
                r5.fit(assemble(B["F_trva"], H_trva, n_g, top)[tr_i],
                       y_dev[tr_i])
                scores.append(macro_f1(
                    y_dev[va_i],
                    r5.predict(assemble(B["F_trva"], H_trva, n_g,
                                        top)[va_i])))
            per_rho[str(rho)] = {
                "N_G": n_g, "N_H": n_h,
                "mean_cv_macro_f1": round(float(np.mean(scores)), 4),
                "std_cv_macro_f1": round(float(np.std(scores)), 4)}
        best = max(v["mean_cv_macro_f1"] for v in per_rho.values())
        rho_star = min(float(r) for r, v in per_rho.items()
                       if v["mean_cv_macro_f1"] >= best - TIE_TOL)
        n_g, n_h = budget_split(rho_star)
        if n_h > 0:
            f_stat, _ = f_classif(H_trva, y_dev)
            top = np.argsort(-np.nan_to_num(f_stat, nan=0.0),
                             kind="stable")[:n_h]
        else:
            top = np.zeros(0, dtype=int)
        r5m = RidgeClassifierCV(alphas=ALPHAS)
        r5m.fit(assemble(B["F_trva"], H_trva, n_g, top), y_dev)
        r5_val = round(macro_f1(
            yva, r5m.predict(assemble(B["F_trva"], H_trva, n_g,
                                      top)[len(ytr):])), 4)
        r5_test = round(macro_f1(
            yte, r5m.predict(assemble(B["F_te"], H_te, n_g, top))), 4)
        log(f"  seed{seed} R5: rho*={rho_star} val={r5_val} test={r5_test}")
        rows.append({"dataset": DS, "model": "R5", "seed": seed,
                     "val_macro_f1": r5_val, "test_macro_f1": r5_test,
                     "selected_alpha": float(r5m.alpha_),
                     "rho": rho_star, "N_G": n_g, "N_H": n_h,
                     "source": "fresh", "cv_curve": per_rho,
                     "runtime_s": round(time.time() - t0, 1)})

    for seed in (42, 43, 44):
        rows.insert(0, {"dataset": DS, "model": "M0", "seed": seed,
                        "val_macro_f1": m0_val, "test_macro_f1": m0_test,
                        "selected_alpha": float(ridge.alpha_), "rho": "",
                        "N_G": 9996, "N_H": 0,
                        "source": "fresh_deterministic"})
    with open(os.path.join(OUT_DIR, "multiseed_uwavey.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    log("saved")


if __name__ == "__main__":
    main()
