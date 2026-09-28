"""CCWRU FINAL 3-SEED -- MiniRocket (MR) + HERAMBA R5 (rcmkn R2 family).

Datasets: CWRU_UNBAL (1156/204/240) and CWRU_BAL (2727/482/567), canonical
NPZ splits (data/cwru_{unbalanced,balanced}.npz), 4 classes, T=1024,
per-sample z-norm, per the audited transfer-line manifest (split seed 42,
stratified 85/15 then 15% of trainval as val).

Canonical seed protocol: 42/43/44 (repo standard, cf.
drtn_conditioned_minirocket_ecg5000_bal_3seed and rcmkn_r2_haptics_3seed).

Per seed the runner executes BOTH arms through the audited implementations
(no copies, no protocol changes):

  MR arm (MiniRocket):
    aeon MiniRocket(random_state=42, fit on TRAIN only) -> 9996 features.
    Val diagnostic: RidgeClassifierCV(alphas=logspace(-4,4,20)) fit on TRAIN
    only -> predict val.  Final: refit on TRAIN+VAL -> ONE test evaluation.
    (Repo MR convention, cf. external_stack_generalization.baselines.)
    Deterministic across seeds by canonical convention; seed-42 gated vs
    M0_REF (UNBAL 0.9917, BAL 0.9947; tol 0.0011).

  HERAMBA R5 arm (user-confirmed: R5 family; R2 fixed-rho [G||H] composition):
    learned context (RCMKNContextModel: SSL + VQ K=8 + joint) TRAINED PER
    OUTER SEED via the audited rcmkn_r2_haptics_3seed core
    (run_context_for_seed machinery, n_classes=4), then
    H = occupancy-weighted regime heterogeneity (compute_regime_heterogeneity,
    H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2, min_occupancy 0.01),
    X_R2 = [G(4998) || H(4998)] -> RidgeClassifierCV, same alpha grid,
    train-only fit -> val diagnostic, train+val refit -> ONE test evaluation.
    Seed-42 gated vs canonical R2 refs (UNBAL 0.9917
    [rcmkn_ssl_context_transfer_seed42], BAL 0.9982
    [rcmkn_ssl_context_important2_seed42]; tol 0.011, repo R2 gate scale).

Frozen context config (identical for both datasets, verified byte-equal in
results/rcmkn_ssl_context_{transfer,important2}_seed42/config.json and equal
to the audited 3-seed core config).  NO test-driven tuning; test set touched
exactly once per (model, seed).
"""
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "ccwru_final_3seed")

SEEDS = [42, 43, 44]
DATASETS = ["CWRU_UNBAL", "CWRU_BAL"]
EXPECTED = {
    "CWRU_UNBAL": {"T": 1024, "n_classes": 4, "train": 1156, "val": 204,
                   "test": 240},
    "CWRU_BAL": {"T": 1024, "n_classes": 4, "train": 2727, "val": 482,
                 "test": 567},
}
M0_REF = {"CWRU_UNBAL": 0.9917, "CWRU_BAL": 0.9947}
M0_TOL = 0.0011
R2_REF = {"CWRU_UNBAL": 0.9917, "CWRU_BAL": 0.9982}
R2_TOL = 0.011
N_FEATURES, N_GLOBAL, N_HET = 9996, 4998, 4998
MINIROCKET_SEED = 42
ALPHAS = np.logspace(-4, 4, 20)


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro", zero_division=0)


def sha16(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(
        a, dtype=np.int64).tobytes()).hexdigest()[:16]


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ridge_two_stage(Xtr, ytr, Xva, yva, Xte, yte):
    """Repo MR/R2 protocol: train-only fit -> val diagnostic;
    train+val refit -> ONE test evaluation."""
    from sklearn.linear_model import RidgeClassifierCV
    clf_val = RidgeClassifierCV(alphas=ALPHAS)
    clf_val.fit(Xtr, ytr)
    val_mf1 = macro_f1(yva, clf_val.predict(Xva))
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(Xtrva, ytrva)
    pred_te = clf.predict(Xte).astype(np.int64)
    res = {"val_macro_f1": round(float(val_mf1), 4),
           "test_macro_f1": round(float(macro_f1(yte, pred_te)), 4),
           "selected_alpha": float(clf.alpha_)}
    return res, pred_te


def run_mr_arm(extractor, Xtrva_z, Xte_z, ytr, yva, yte, n_train):
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES
    assert not np.isnan(F_trva).any() and not np.isinf(F_trva).any()
    res, pred = ridge_two_stage(F_trva[:n_train], ytr, F_trva[n_train:], yva,
                                F_te, yte)
    return res, pred, F_trva, F_te


def train_context_for_seed(seed, n_classes, Xtr_z, ytr, Xva_z, yva, Xte_z,
                           device):
    """Audited per-seed context training (rcmkn_r2_haptics_3seed core,
    n_classes adapted 5 -> 4; schedule/config unchanged)."""
    from experiments.rcmkn_haptics_seed42.runner import (
        train_context_model, extract_context_regimes, set_seed as core_set_seed)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    core_set_seed(seed)
    model = RCMKNContextModel(n_classes=n_classes).to(device)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    core_set_seed(seed)
    regimes_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    core_set_seed(seed)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=32)
    return model, train_info, Xtrva_z, regimes_trva, regimes_te


def compute_H(extractor, X_z, regimes, valid_het, chunk=32):
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations)
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    N = X_z.shape[0]
    H = np.empty((N, N_HET), dtype=np.float64)
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, _ = compute_raw_activations(extractor, X_z[c0:c1])
        H[c0:c1] = compute_regime_heterogeneity(
            act[:, N_GLOBAL:], valid_het, regimes[c0:c1])
        del act
    return H


def run_dataset(ds_name, device):
    from experiments.drtn_conditioned_minirocket_transfer_seed42.runner import (
        load_any_dataset)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations)
    from aeon.transformations.collection.convolution_based import MiniRocket

    ds_dir = os.path.join(OUT, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    data = load_any_dataset(ds_name)
    Xtr, ytr, Xva, yva = data["Xtr"], data["ytr"], data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    got = {"T": int(Xtr.shape[1]), "n_classes": data["n_classes"],
           "train": len(Xtr), "val": len(Xva), "test": len(Xte)}
    assert got == EXPECTED[ds_name], f"split identity FAILED {ds_name}: {got}"
    log(f"\n=== {ds_name}: {got} ===")

    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    n_train = len(Xtr)
    ytrva = np.concatenate([ytr, yva])

    # fixed MiniRocket (canonical: random_state=42, fit on train only)
    set_seed(MINIROCKET_SEED)
    extractor = MiniRocket(random_state=MINIROCKET_SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))

    # valid-region masks (audit: extractor identity)
    mr_id, valid = 0.0, None
    for c0 in range(0, len(Xtrva_z), 32):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 32])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - extractor.transform(
                Xtrva_z[c0:c0 + 32][:, None, :].astype(np.float32))))))
        del act
    assert mr_id < 1e-5, f"extractor identity FAILED: {mr_id}"
    valid_het = valid[N_GLOBAL:]
    log(f"  [AUDIT] extractor identity max|diff|={mr_id:.2e}")

    rows = []
    for seed in SEEDS:
        sdir = os.path.join(ds_dir, f"seed{seed}")
        os.makedirs(sdir, exist_ok=True)

        # ---------------- MR arm (deterministic) ----------------
        t0 = time.time()
        mr_res, mr_pred, F_trva, F_te = run_mr_arm(
            extractor, Xtrva_z, Xte_z, ytr, yva, yte, n_train)
        mr_res.update({"seed": seed, "runtime_s": round(time.time() - t0, 1)})
        np.save(os.path.join(sdir, "mr_predictions.npy"), mr_pred)
        with open(os.path.join(sdir, "mr_result.json"), "w") as f:
            json.dump(mr_res, f, indent=2)
        mr_status = "REUSED_DETERMINISTIC"
        if seed == 42:
            ok = abs(mr_res["test_macro_f1"] - M0_REF[ds_name]) <= M0_TOL
            log(f"  [MR GATE {ds_name}] {mr_res['test_macro_f1']} vs "
                f"canonical {M0_REF[ds_name]} (tol {M0_TOL}) -> "
                f"{'PASS' if ok else 'FAIL'}")
            assert ok, f"MR gate FAILED on {ds_name}"
            mr_status = "NEW_RUN_GATE_PASS"
        rows.append({"dataset": ds_name, "model": "MiniRocket", "seed": seed,
                     "val_macro_f1": mr_res["val_macro_f1"],
                     "test_macro_f1": mr_res["test_macro_f1"],
                     "selected_alpha": mr_res["selected_alpha"],
                     "status": mr_status,
                     "runtime_s": mr_res["runtime_s"]})
        log(f"  [MR seed{seed}] val={mr_res['val_macro_f1']} "
            f"test={mr_res['test_macro_f1']} "
            f"alpha={mr_res['selected_alpha']:.4f}")

        # ---------------- HERAMBA R5 arm ----------------
        t0 = time.time()
        model, train_info, _, regimes_trva, regimes_te = \
            train_context_for_seed(seed, data["n_classes"], Xtr_z, ytr,
                                   Xva_z, yva, Xte_z, device)
        det = bool(np.array_equal(
            _reextract(model, seed, Xtrva_z, device), regimes_trva))
        H_trva = compute_H(extractor, Xtrva_z, regimes_trva, valid_het)
        H_te = compute_H(extractor, Xte_z, regimes_te, valid_het)
        assert H_trva.shape[1] == N_HET and not np.isnan(H_trva).any()
        r5_res, r5_pred = ridge_two_stage(
            np.hstack([F_trva[:n_train, :N_GLOBAL], H_trva[:n_train]]), ytr,
            np.hstack([F_trva[n_train:, :N_GLOBAL], H_trva[n_train:]]), yva,
            np.hstack([F_te[:, :N_GLOBAL], H_te]), yte)
        r5_res.update({"seed": seed,
                       "runtime_s": round(time.time() - t0, 1),
                       "regimes_deterministic": det,
                       "regime_hash_te": sha16(regimes_te),
                       "context_train": train_info})
        np.save(os.path.join(sdir, "heramba_r5_predictions.npy"), r5_pred)
        with open(os.path.join(sdir, "heramba_r5_result.json"), "w") as f:
            json.dump(r5_res, f, indent=2)
        r5_status = "NEW_RUN"
        if seed == 42:
            ok = abs(r5_res["test_macro_f1"] - R2_REF[ds_name]) <= R2_TOL
            log(f"  [R5 GATE {ds_name}] {r5_res['test_macro_f1']} vs "
                f"canonical R2 {R2_REF[ds_name]} (tol {R2_TOL}) -> "
                f"{'PASS' if ok else 'FAIL'}")
            assert ok, f"R5 gate FAILED on {ds_name}"
            r5_status = "NEW_RUN_GATE_PASS"
        rows.append({"dataset": ds_name, "model": "HERAMBA R5", "seed": seed,
                     "val_macro_f1": r5_res["val_macro_f1"],
                     "test_macro_f1": r5_res["test_macro_f1"],
                     "selected_alpha": r5_res["selected_alpha"],
                     "status": r5_status, "runtime_s": r5_res["runtime_s"]})
        log(f"  [R5 seed{seed}] val={r5_res['val_macro_f1']} "
            f"test={r5_res['test_macro_f1']} "
            f"alpha={r5_res['selected_alpha']:.4f} ({r5_res['runtime_s']}s)")
        del model, H_trva, H_te, F_trva, F_te
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _reextract(model, seed, Xtrva_z, device):
    from experiments.rcmkn_haptics_seed42.runner import (
        extract_context_regimes, set_seed as core_set_seed)
    core_set_seed(seed)
    return extract_context_regimes(model, Xtrva_z[:64], device, batch=32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    log(f"CCWRU FINAL 3-SEED -- MR + HERAMBA R5 -- seeds={SEEDS} "
        f"device={device}")
    all_rows = []
    for ds in DATASETS:
        all_rows.extend(run_dataset(ds, device))
    with open(os.path.join(OUT, "per_run_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    log("\nCCWRU FINAL 3-SEED RUNS COMPLETE")
    for r in all_rows:
        log(f"  {r['dataset']} {r['model']} seed{r['seed']}: "
            f"test={r['test_macro_f1']}")


if __name__ == "__main__":
    main()
