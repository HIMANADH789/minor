"""GunPoint / Phoneme / FordA -- MiniRocket vs HERAMBA R5, 3 seeds each.

CWRU/ES lines are paused and preserved. This task covers EXACTLY:
    GunPoint, Phoneme, FordA x {MR, HERAMBA R5} x {42,43,44} = 18 cells.

Audit facts (Phase 1, all seed-42 canonical references):
    GunPoint: results/rcmkn_r2_gunpoint_seed42/result.json
        M0 = 0.9933 (tol 0.0011, stored gate config) | R2 = 1.0000
        split 42/8/150, T=150, 2 classes; loader
        drtn_conditioned_minirocket_context3_seed42.load_kaggle_ucr
    Phoneme: results/rcmkn_ssl_context_transfer_seed42/Phoneme/result.json
        R2 = 0.1182 | MR ref 0.0808 (external_stack, tol 0.011)
        split 185/29/1896, T=1024, 39 classes (provided canonical val)
    FordA: results/rcmkn_r2_kaggle_context2_seed42/per_dataset_results.json
        M0 = 0.9499 | R2 = 0.9560
        split 3060/541/1320, T=500, 2 classes
Canonical seed set {42,43,44}. Frozen R2/HERAMBA recipe (same as the audited
CWRU/ES/Haptics runs): per-seed SSL+VQ context (RCMKNContextModel),
H = occupancy-weighted regime heterogeneity (min_occ 0.01),
X = [G 4998 || H 4998], RidgeClassifierCV(logspace(-4,4,20)),
train-only fit -> val diagnostic; train+val refit -> ONE test eval.
MiniRocket carriers fixed at random_state=42 (canonical convention).

FALLBACK PROTOCOL (task-defined, validation-stage only):
    final R5 arm for a seed = HERAMBA candidate if its TRAIN-ONLY-fit
    validation Macro-F1 >= MR's train-only-fit validation Macro-F1 on the
    SAME frozen features/split; else the identical MR model. Selection
    uses validation only; test is evaluated ONCE for the FINAL selected
    model. Raw-HERAMBA test metrics are ALSO recorded for the audit but
    are NEVER used for selection. No max() over test scores.

No test-driven tuning; no R5 changes; no MR retuning.
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
OUT = os.path.join(ROOT, "results", "gunpoint_phoneme_forda_r5")

SEEDS = [42, 43, 44]
N_FEATURES, N_GLOBAL, N_HET = 9996, 4998, 4998
MINIROCKET_SEED = 42
ALPHAS = np.logspace(-4, 4, 20)

SPEC = {
    "GunPoint": {
        "loader": "kaggle", "n_classes": 2,
        "expected": {"train": 42, "val": 8, "test": 150, "T": 150},
        "mr_ref": 0.9933, "mr_tol": 0.0011,
        "r2_ref": 1.0000, "r2_tol": 0.011,
    },
    "Phoneme": {
        "loader": "external", "n_classes": 39,
        "expected": {"train": 185, "val": 29, "test": 1896, "T": 1024},
        "mr_ref": 0.0808, "mr_tol": 0.011,
        "r2_ref": 0.1182, "r2_tol": 0.011,
    },
    "FordA": {
        "loader": "kaggle", "n_classes": 2,
        "expected": {"train": 3060, "val": 541, "test": 1320, "T": 500},
        "mr_ref": 0.9499, "mr_tol": 0.011,
        "r2_ref": 0.9560, "r2_tol": 0.011,
    },
}


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


def load_data(ds_name):
    if SPEC[ds_name]["loader"] == "kaggle":
        from experiments.drtn_conditioned_minirocket_context3_seed42.runner \
            import load_kaggle_ucr as loader
        d = loader(ds_name)
        keys = {"Xtr": "Xtr", "ytr": "ytr", "Xva": "Xva", "yva": "yva",
                "Xte": "Xte", "yte": "yte"}
    else:
        from experiments.external_stack_generalization.data import (
            load_dataset as loader)
        d = loader(ds_name)
        keys = {"Xtr": "Xtr", "ytr": "ytr", "Xva": "Xva", "yva": "yva",
                "Xte": "Xte", "yte": "yte"}
    return {k: d[keys[k]] for k in keys}


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


def run_seed(ds_name, spec, seed, data, extractor, valid_het, Xtrva_z,
             Xte_z, F_trva, F_te, ytr, yva, yte, device):
    from sklearn.linear_model import RidgeClassifierCV
    from experiments.rcmkn_haptics_seed42.runner import (
        train_context_model, extract_context_regimes, set_seed as core_set_seed)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel

    n_tr, n_va = spec["expected"]["train"], spec["expected"]["val"]
    Xtr_z, Xva_z = Xtrva_z[:n_tr], Xtrva_z[n_tr:]
    t0 = time.time()

    # ---------------- MR arm (frozen control) ----------------
    mr_val = macro_f1(yva, RidgeClassifierCV(alphas=ALPHAS).fit(
        F_trva[:n_tr], ytr).predict(F_trva[n_tr:]))
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(F_trva, np.concatenate([ytr, yva]))
    mr_test = macro_f1(yte, clf.predict(F_te))
    mr_alpha = float(clf.alpha_)

    # ---------------- HERAMBA R5 arm (frozen recipe) ----------------
    core_set_seed(seed)
    model = RCMKNContextModel(n_classes=spec["n_classes"]).to(device)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    core_set_seed(seed)
    regimes_trva = extract_context_regimes(model, Xtrva_z, device, batch=64)
    core_set_seed(seed)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=64)
    core_set_seed(seed)
    det = bool(np.array_equal(
        extract_context_regimes(model, Xtrva_z[:64], device, batch=64),
        regimes_trva[:64]))
    H_trva = compute_H(extractor, Xtrva_z, regimes_trva, valid_het)
    H_te = compute_H(extractor, Xte_z, regimes_te, valid_het)
    X_trva_r2 = np.hstack([F_trva[:, :N_GLOBAL], H_trva])
    X_te_r2 = np.hstack([F_te[:, :N_GLOBAL], H_te])
    assert not np.isnan(X_trva_r2).any() and not np.isnan(X_te_r2).any()

    # raw-HERAMBA validation (train-only fit) -- fallback criterion input
    her_val = macro_f1(yva, RidgeClassifierCV(alphas=ALPHAS).fit(
        X_trva_r2[:n_tr], ytr).predict(X_trva_r2[n_tr:]))
    # raw-HERAMBA refit test (audit only -- NEVER used for selection)
    clf_h = RidgeClassifierCV(alphas=ALPHAS)
    clf_h.fit(X_trva_r2, np.concatenate([ytr, yva]))
    her_test = macro_f1(yte, clf_h.predict(X_te_r2))
    her_alpha = float(clf_h.alpha_)

    # ---------------- VALIDATION-STAGE FALLBACK ----------------
    select_heramaba = bool(her_val >= mr_val)
    final_test = her_test if select_heramaba else mr_test
    final_pred = (clf_h.predict(X_te_r2) if select_heramaba
                  else clf.predict(F_te)).astype(np.int64)
    final_alpha = her_alpha if select_heramaba else mr_alpha
    del model, H_trva, H_te
    if device.type == "cuda":
        torch.cuda.empty_cache()

    res = {
        "seed": seed,
        "mr_val": round(float(mr_val), 4), "mr_test": round(float(mr_test), 4),
        "mr_alpha": mr_alpha,
        "her_val": round(float(her_val), 4),
        "her_test": round(float(her_test), 4), "her_alpha": her_alpha,
        "selected": "HERAMBA" if select_heramaba else "MiniRocket",
        "fallback_used": not select_heramaba,
        "final_test": round(float(final_test), 4),
        "final_alpha": final_alpha,
        "regimes_deterministic": det, "regime_hash_te": sha16(regimes_te),
        "context_train": train_info,
        "runtime_s": round(time.time() - t0, 1),
    }
    return res, final_pred, her_pred_audit(clf_h, X_te_r2), \
        mr_pred_audit(clf, F_te)


def her_pred_audit(clf_h, X_te_r2):
    return clf_h.predict(X_te_r2).astype(np.int64)


def mr_pred_audit(clf, F_te):
    return clf.predict(F_te).astype(np.int64)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    log(f"GunPoint/Phoneme/FordA -- MR vs HERAMBA R5 -- seeds={SEEDS} "
        f"device={device}")
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations)
    from experiments.rcmkn_haptics_seed42.runner import set_seed
    from aeon.transformations.collection.convolution_based import MiniRocket

    all_rows = []
    for ds_name, spec in SPEC.items():
        ds_dir = os.path.join(OUT, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        data = load_data(ds_name)
        ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
        got = {"train": len(data["Xtr"]), "val": len(data["Xva"]),
               "test": len(data["Xte"]), "T": int(data["Xtr"].shape[1])}
        assert got == spec["expected"], f"{ds_name} split identity: {got}"
        log(f"\n=== {ds_name}: {got} ===")

        Xtr_z, Xva_z, Xte_z = (znorm(data["Xtr"]), znorm(data["Xva"]),
                               znorm(data["Xte"]))
        Xtrva_z = np.vstack([Xtr_z, Xva_z])
        set_seed(MINIROCKET_SEED)
        extractor = MiniRocket(random_state=MINIROCKET_SEED, n_jobs=-1)
        extractor.fit(Xtr_z[:, None, :].astype(np.float32))
        F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
        F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
        assert F_trva.shape[1] == N_FEATURES
        mr_id, valid = 0.0, None
        for c0 in range(0, len(Xtrva_z), 32):
            act, valid = compute_raw_activations(
                extractor, Xtrva_z[c0:c0 + 32])
            mr_id = max(mr_id, float(np.max(np.abs(
                ppv_from_activations(act, valid) - F_trva[c0:c0 + 32]))))
            del act
        assert mr_id < 1e-5, f"{ds_name} extractor identity: {mr_id}"
        valid_het = valid[N_GLOBAL:]

        for seed in SEEDS:
            res, final_pred, her_pred, mr_pred = run_seed(
                ds_name, spec, seed, data, extractor, valid_het, Xtrva_z,
                Xte_z, F_trva, F_te, ytr, yva, yte, device)
            sdir = os.path.join(ds_dir, f"seed{seed}")
            os.makedirs(sdir, exist_ok=True)
            np.save(os.path.join(sdir, "final_predictions.npy"), final_pred)
            np.save(os.path.join(sdir, "heramba_raw_predictions.npy"),
                    her_pred)
            np.save(os.path.join(sdir, "mr_predictions.npy"), mr_pred)
            with open(os.path.join(sdir, "result.json"), "w") as f:
                json.dump(res, f, indent=2)
            row = {"dataset": ds_name, **res}
            all_rows.append(row)
            log(f"  [seed{seed}] MR val={res['mr_val']} test={res['mr_test']}"
                f" | HER val={res['her_val']} test={res['her_test']} | "
                f"selected={res['selected']} final={res['final_test']} "
                f"({res['runtime_s']}s)")

    with open(os.path.join(OUT, "per_run_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    log("\nGUNPOINT/PHONEME/FORDA RUNS COMPLETE")


if __name__ == "__main__":
    main()
