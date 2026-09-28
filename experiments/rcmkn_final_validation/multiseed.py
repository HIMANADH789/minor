"""Multi-seed stability for the final validation (seed 42/43/44).

Primary: Haptics M0/R2/R5.
    M0: deterministic MiniROCKET+Ridge (seed-42 extractor) -> ONE run,
        reused for all three seeds (canonical convention: MiniROCKET is
        fixed at random_state=42; RidgeClassifierCV LOO is deterministic)
    R2 seed 42: canonical 0.5500 (stored, not re-evaluated)
    R2 seeds 43/44: reused from the stored rcmkn_r2_haptics_3seed run
        (exact protocol, fixed MiniROCKET, learned context retrained per
        seed) -- no rerun needed
    R5: selected-rho pipeline per seed (rho re-selected per seed on the
        development set; MiniROCKET fixed at 42; context retrained per
        seed via the audited 3-seed core)
Secondary (optional, --secondary): UWaveGestureLibraryY and Phoneme
    M0/R2/R5 at seed 42 only if not already canonical.
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
    K_FOLDS, OUT_DIR, REPO_ROOT, RHO_GRID, SEED, TIE_TOL, TOTAL_BUDGET)

ALPHAS = np.logspace(-4, 4, 20)
HAPTICS_3SEED = os.path.join(REPO_ROOT, "results", "rcmkn_r2_haptics_3seed")


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


def r5_for_seed_haptics(seed, device):
    """Full R5 for one Haptics seed via the audited 3-seed core."""
    from experiments.rcmkn_r2_haptics_3seed import core as c3
    from experiments.rcmkn_r2_haptics_3seed.runner import run_one_seed
    data = c3.load_data()
    Xtr, ytr, Xva, yva, Xte, yte = (data["Xtr"], data["ytr"], data["Xva"],
                                    data["yva"], data["Xte"], data["yte"])
    ytrva = np.concatenate([ytr, yva])
    y_dev = np.concatenate([ytr, yva])
    Xtr_z, Xva_z, Xte_z = c3.znorm(Xtr), c3.znorm(Xva), c3.znorm(Xte)
    Xtrva_z = c3.stack_trva(Xtr_z, Xva_z)
    extractor = c3.build_fixed_extractor(Xtr_z)
    valid_het = c3.compute_valid_het(extractor, Xtr_z[:8])
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))

    model, train_info, _, regimes_trva = c3.run_context_for_seed(
        seed, Xtr_z, ytr, Xva_z, yva, device)
    from experiments.rcmkn_haptics_seed42.runner import (
        extract_context_regimes, set_seed as _set_seed)
    _set_seed(seed)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=32)

    def het(X_z, regimes, chunk=64):
        H = np.empty((len(X_z), 4998), dtype=np.float64)
        for c0 in range(0, len(X_z), chunk):
            c1 = min(c0 + chunk, len(X_z))
            from experiments.drtn_conditioned_minirocket_transfer_seed42.core \
                import compute_raw_activations
            act, _ = compute_raw_activations(extractor, X_z[c0:c1])
            from experiments.drtn_conditioned_minirocket_haptics_3seed.runner \
                import compute_regime_heterogeneity
            H[c0:c1] = compute_regime_heterogeneity(
                act[:, 4998:], valid_het, regimes[c0:c1])
            del act
        return H

    H_trva = het(Xtrva_z, regimes_trva)
    H_te = het(Xte_z, regimes_te)
    y_dev_all = y_dev

    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    per_rho = {}
    for rho in RHO_GRID:
        n_g, n_h = budget_split(rho)
        scores = []
        for tr_i, va_i in kfold.split(np.zeros(len(y_dev_all)), y_dev_all):
            if n_h > 0:
                f_stat, _ = f_classif(H_trva[tr_i], y_dev_all[tr_i])
                top = np.argsort(-np.nan_to_num(f_stat, nan=0.0),
                                 kind="stable")[:n_h]
            else:
                top = np.zeros(0, dtype=int)
            ridge = RidgeClassifierCV(alphas=ALPHAS)
            ridge.fit(assemble(F_trva, H_trva, n_g, top)[tr_i],
                      y_dev_all[tr_i])
            scores.append(macro_f1(
                y_dev_all[va_i],
                ridge.predict(assemble(F_trva, H_trva, n_g, top)[va_i])))
        per_rho[str(rho)] = {
            "N_G": n_g, "N_H": n_h,
            "mean_cv_macro_f1": round(float(np.mean(scores)), 4),
            "std_cv_macro_f1": round(float(np.std(scores)), 4)}
        log(f"    seed{seed} rho={rho}: {np.mean(scores):.4f}")
    best = max(v["mean_cv_macro_f1"] for v in per_rho.values())
    rho_star = min(float(r) for r, v in per_rho.items()
                   if v["mean_cv_macro_f1"] >= best - TIE_TOL)
    n_g, n_h = budget_split(rho_star)
    if n_h > 0:
        f_stat, _ = f_classif(H_trva, y_dev_all)
        top = np.argsort(-np.nan_to_num(f_stat, nan=0.0),
                         kind="stable")[:n_h]
    else:
        top = np.zeros(0, dtype=int)
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(assemble(F_trva, H_trva, n_g, top), y_dev_all)
    pred_va = ridge.predict(assemble(F_trva, H_trva, n_g, top)[len(Xtr):])
    pred_te = ridge.predict(assemble(F_te, H_te, n_g, top))
    return {
        "seed": seed, "selected_rho": rho_star, "N_G": n_g, "N_H": n_h,
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "test_macro_f1": round(macro_f1(yte, pred_te), 4),
        "selected_alpha": float(ridge.alpha_),
        "cv_curve": per_rho,
    }


def main(secondary=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 74)
    log("MULTI-SEED STABILITY — M0/R2/R5 — seeds 42/43/44")
    log("=" * 74)
    rows = []

    # ---- Haptics M0: deterministic single run (canonical convention) ------
    B = build_banks("Haptics", device)
    ytr, yva, yte = B["y"]
    y_dev = np.concatenate([ytr, yva])
    from sklearn.linear_model import RidgeClassifierCV as RC
    ridge = RC(alphas=ALPHAS)
    ridge.fit(B["F_trva"], y_dev)
    m0_val = round(macro_f1(yva, ridge.predict(B["F_trva"][len(ytr):])), 4)
    m0_test = round(macro_f1(yte, ridge.predict(B["F_te"])), 4)
    m0_alpha = float(ridge.alpha_)
    log(f"  [M0 Haptics] val={m0_val} test={m0_test} "
        f"(deterministic; reused for all seeds)")
    for seed in (42, 43, 44):
        rows.append({"dataset": "Haptics", "model": "M0", "seed": seed,
                     "val_macro_f1": m0_val, "test_macro_f1": m0_test,
                     "selected_alpha": m0_alpha, "rho": "",
                     "N_G": 9996, "N_H": 0, "source": "fresh_deterministic"})

    # ---- Haptics R2: seed 42 canonical + 43/44 from stored 3-seed ---------
    rows.append({"dataset": "Haptics", "model": "R2", "seed": 42,
                 "val_macro_f1": 0.9014, "test_macro_f1": 0.5500,
                 "selected_alpha": 4.281332398719396, "rho": "",
                 "N_G": 4998, "N_H": 4998, "source": "canonical_stored"})
    for seed in (43, 44):
        d = json.load(open(os.path.join(HAPTICS_3SEED, f"seed{seed}",
                                        "result.json")))
        r = d["results"]
        rows.append({"dataset": "Haptics", "model": "R2", "seed": seed,
                     "val_macro_f1": r["val_macro_f1"],
                     "test_macro_f1": r["test_macro_f1"],
                     "selected_alpha": r["selected_alpha"], "rho": "",
                     "N_G": 4998, "N_H": 4998,
                     "source": "rcmkn_r2_haptics_3seed_stored"})

    # ---- Haptics R5 per seed ----------------------------------------------
    for seed in (42, 43, 44):
        t0 = time.time()
        r = r5_for_seed_haptics(seed, device)
        r["runtime_s"] = round(time.time() - t0, 1)
        r["dataset"], r["model"], r["source"] = ("Haptics", "R5",
                                                 "fresh_r5_pipeline")
        rows.append(r)
        log(f"  [R5 seed{seed}] rho*={r['selected_rho']} "
            f"test={r['test_macro_f1']}")

    out = {"rows": rows, "note": "n=3 seeds: descriptive stability only; "
                                   "no significance claims"}
    with open(os.path.join(OUT_DIR, "multiseed_haptics.json"), "w") as f:
        json.dump(out, f, indent=2)
    log(f"  saved {len(rows)} rows")
    return out


if __name__ == "__main__":
    main()
