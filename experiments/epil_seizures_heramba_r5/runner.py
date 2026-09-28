"""EpilepticSeizures -- HERAMBA R5 (rcmkn R2 family) 3-seed completion.

CWRU is PAUSED and preserved (results/ccwru_final_3seed/ complete).

Audit facts (results/rcmkn_ssl_context_transfer_seed42/EpilepticSeizures):
  canonical seed-42 R2 = 0.9457 test Macro-F1 (val 1.0, alpha 1.6238,
  feature_dim 9996 = [G 4998 || H 4998]); context config byte-identical to
  the frozen rcmkn recipe; canonical provided val split 80/20/11420,
  T=178, 2 classes (no resampling). Seeds 43/44 did not exist anywhere.
Canonical seed set {42,43,44} (repo standard).

Per seed: learned context (RCMKNContextModel SSL+VQ K=8+joint,
n_classes=2) trained PER OUTER SEED via the audited rcmkn_r2_haptics_3seed
core machinery; H = occupancy-weighted regime heterogeneity
(compute_regime_heterogeneity, min_occupancy 0.01, valid-region masks);
X_R2 = [G || H] -> RidgeClassifierCV(alphas=logspace(-4,4,20)):
train-only fit -> val diagnostic; train+val refit -> ONE test evaluation.
MiniRocket carriers fixed at random_state=42 (fit on train only).
Seed-42 gated vs canonical 0.9457 (tol 0.011, repo R2 gate scale).
No test-driven tuning; R5 frozen.
"""
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
OUT = os.path.join(ROOT, "results", "epil_seizures_heramba_r5")

SEEDS = [42, 43, 44]
EXPECTED = {"T": 178, "n_classes": 2, "train": 80, "val": 20, "test": 11420}
R2_REF = 0.9457
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


def ridge_two_stage(Xtr, ytr, Xva, yva, Xte, yte):
    from sklearn.linear_model import RidgeClassifierCV
    clf_val = RidgeClassifierCV(alphas=ALPHAS)
    clf_val.fit(Xtr, ytr)
    val_mf1 = macro_f1(yva, clf_val.predict(Xva))
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    pred_te = clf.predict(Xte).astype(np.int64)
    res = {"val_macro_f1": round(float(val_mf1), 4),
           "test_macro_f1": round(float(macro_f1(yte, pred_te)), 4),
           "selected_alpha": float(clf.alpha_)}
    return res, pred_te


def compute_H(extractor, X_z, regimes, valid_het, chunk=64):
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


def run_seed(seed, data, extractor, valid_het, Xtrva_z, Xte_z, F_trva, F_te,
             ytr, yva, yte, device):
    from experiments.rcmkn_haptics_seed42.runner import (
        train_context_model, extract_context_regimes, set_seed as core_set_seed)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel

    Xtr_z = Xtrva_z[:EXPECTED["train"]]
    Xva_z = Xtrva_z[EXPECTED["train"]:]
    t0 = time.time()
    core_set_seed(seed)
    model = RCMKNContextModel(n_classes=EXPECTED["n_classes"]).to(device)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    core_set_seed(seed)
    regimes_trva = extract_context_regimes(model, Xtrva_z, device, batch=64)
    core_set_seed(seed)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=64)
    # determinism re-extract (AUDIT-8 convention)
    core_set_seed(seed)
    reg_re = extract_context_regimes(model, Xtrva_z[:64], device, batch=64)
    det = bool(np.array_equal(reg_re, regimes_trva[:64]))

    H_trva = compute_H(extractor, Xtrva_z, regimes_trva, valid_het)
    H_te = compute_H(extractor, Xte_z, regimes_te, valid_het)
    assert H_trva.shape == (len(Xtrva_z), N_HET)
    assert not np.isnan(H_trva).any() and not np.isnan(H_te).any()
    n_tr = EXPECTED["train"]
    res, pred = ridge_two_stage(
        np.hstack([F_trva[:n_tr, :N_GLOBAL], H_trva[:n_tr]]), ytr,
        np.hstack([F_trva[n_tr:, :N_GLOBAL], H_trva[n_tr:]]), yva,
        np.hstack([F_te[:, :N_GLOBAL], H_te]), yte)
    res.update({"seed": seed, "runtime_s": round(time.time() - t0, 1),
                "regimes_deterministic": det,
                "regime_hash_te": sha16(regimes_te),
                "context_train": train_info})
    del model, H_trva, H_te
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    log(f"ES HERAMBA R5 3-SEED -- seeds={SEEDS} device={device}")

    from experiments.external_stack_generalization.data import (
        load_dataset, znorm)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations)
    from aeon.transformations.collection.convolution_based import MiniRocket

    data = load_dataset("EpilepticSeizures")
    Xtr, ytr, Xva, yva, Xte, yte = (data["Xtr"], data["ytr"], data["Xva"],
                                    data["yva"], data["Xte"], data["yte"])
    got = {"T": int(Xtr.shape[1]), "n_classes": data["n_classes"],
           "train": len(Xtr), "val": len(Xva), "test": len(Xte)}
    assert got == EXPECTED, f"split identity FAILED: {got}"
    log(f"split identity OK: {got} (provided canonical val, never resampled)")

    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    from experiments.rcmkn_haptics_seed42.runner import set_seed
    set_seed(MINIROCKET_SEED)
    extractor = MiniRocket(random_state=MINIROCKET_SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES
    assert not np.isnan(F_trva).any()

    mr_id, valid = 0.0, None
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5, f"extractor identity FAILED: {mr_id}"
    valid_het = valid[N_GLOBAL:]
    log(f"[AUDIT] extractor identity max|diff|={mr_id:.2e}")

    rows = []
    for seed in SEEDS:
        res, pred = run_seed(seed, data, extractor, valid_het, Xtrva_z,
                             Xte_z, F_trva, F_te, ytr, yva, yte, device)
        sdir = os.path.join(OUT, f"seed{seed}")
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "heramba_r5_result.json"), "w") as f:
            json.dump(res, f, indent=2)
        np.save(os.path.join(sdir, "heramba_r5_predictions.npy"), pred)
        status = "NEW_RUN"
        if seed == 42:
            ok = abs(res["test_macro_f1"] - R2_REF) <= R2_TOL
            log(f"  [R5 GATE] {res['test_macro_f1']} vs canonical R2 "
                f"{R2_REF} (tol {R2_TOL}) -> {'PASS' if ok else 'FAIL'}")
            assert ok, "R5 seed-42 gate FAILED vs canonical 0.9457"
            status = "NEW_RUN_GATE_PASS"
        rows.append({"seed": seed, "val_macro_f1": res["val_macro_f1"],
                     "test_macro_f1": res["test_macro_f1"],
                     "selected_alpha": res["selected_alpha"],
                     "runtime_s": res["runtime_s"], "status": status})
        log(f"  [R5 seed{seed}] val={res['val_macro_f1']} "
            f"test={res['test_macro_f1']} "
            f"alpha={res['selected_alpha']:.4f} ({res['runtime_s']}s)")

    with open(os.path.join(OUT, "per_run_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    log("ES HERAMBA R5 3-SEED COMPLETE")


if __name__ == "__main__":
    main()
