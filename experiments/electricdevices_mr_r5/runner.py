"""ElectricDevices -- MiniRocket vs HERAMBA R5 (HERAMBA), 3 seeds, NO FALLBACK.

Dataset (installed & verified, results/electricdevices_mr_r5/AUDIT.md):
    repo UCRArchive_2018 copy, TRAIN 8926x97 / TEST 7711x97, T=96,
    7 classes; canonical train/test TSVs never re-split; stratified 15%
    of train as val (random_state=42) -> 7587/1339/7711. sha256-16:
    TRAIN bf9b3e4ebb4bcecc, TEST 64ea1c73040b8c60.
    No prior ElectricDevices artifacts in the R2/rcmkn line -> seed-42
    values become the FIRST audited references (no gate; recorded).

Protocol (frozen recipe identical to the audited rcmkn lines):
    MR        : aeon MiniRocket(random_state=42, fit TRAIN only) -> 9996
                features; RidgeClassifierCV(logspace(-4,4,20)) train-only
                fit -> val diagnostic; train+val refit -> ONE test eval.
                Deterministic across seeds (canonical convention).
    HERAMBA R5: per-seed RCMKNContextModel (SSL+VQ K=8+joint; frozen
                config byte-identical to rcmkn_haptics_seed42/config.json),
                H = occupancy-weighted regime heterogeneity (min_occ
                0.01), X_R5 = [G 4998 || H 4998], same Ridge/alpha grid,
                train-only fit -> val diagnostic; train+val refit -> ONE
                test evaluation.

NO FALLBACK (task directive): there is NO validation-stage model
switching. Both arms are final; per-seed deltas = HERAMBA_test -
MR_test as measured. No selection, no clamping, no max() anywhere.

No test-driven tuning; R5 frozen; MR frozen.
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
OUT = os.path.join(ROOT, "results", "electricdevices_mr_r5")

SEEDS = [42, 43, 44]
N_FEATURES, N_GLOBAL, N_HET = 9996, 4998, 4998
MINIROCKET_SEED = 42
ALPHAS = np.logspace(-4, 4, 20)

ARCHIVE = os.path.join("data", "kaggle", "_ucrarchive_2018",
                       "UCRArchive_2018", "UCRArchive_2018")
DS = "ElectricDevices"
EXPECTED = {"train": 7587, "val": 1339, "test": 7711, "T": 96}
N_CLASSES = 7


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro", zero_division=0)


def accuracy(y, p):
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y, p))


def sha16(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(
        a, dtype=np.int64).tobytes()).hexdigest()[:16]


def znorm(X):
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


def load_data():
    """Canonical UCR TSVs + stratified 15% of train as val (seed 42).

    Mirrors drtn_conditioned_minirocket_context3_seed42.load_kaggle_ucr
    (same split rule, label remap, sha recording).
    """
    from sklearn.model_selection import train_test_split
    d_dir = os.path.join(ROOT, ARCHIVE, DS)
    tr = np.loadtxt(os.path.join(d_dir, f"{DS}_TRAIN.tsv"),
                    delimiter="\t", ndmin=2)
    te = np.loadtxt(os.path.join(d_dir, f"{DS}_TEST.tsv"),
                    delimiter="\t", ndmin=2)
    ytr, Xtr = tr[:, 0].astype(int), tr[:, 1:].astype(np.float32)
    yte, Xte = te[:, 0].astype(int), te[:, 1:].astype(np.float32)

    labels = sorted(set(ytr.tolist()) | set(yte.tolist()))
    assert labels == list(range(1, 8)), f"label set {labels}"
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)
    lut = {lab: i for i, lab in enumerate(labels)}
    map_fn = np.vectorize(lut.get)
    return {
        "Xtr": Xtr, "ytr": map_fn(ytr), "Xva": Xva, "yva": map_fn(yva),
        "Xte": Xte, "yte": map_fn(yte),
        "n_classes": len(labels), "L": int(Xtr.shape[1]),
        "val_source": "stratified_15pct_of_train_seed42",
        "source_path": os.path.relpath(d_dir, ROOT),
    }


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    log(f"ELECTRICDEVICES -- MR vs HERAMBA R5 (NO FALLBACK) -- "
        f"seeds={SEEDS} device={device}")
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, ppv_from_activations)
    from experiments.rcmkn_haptics_seed42.runner import (
        set_seed, train_context_model, extract_context_regimes)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    from sklearn.linear_model import RidgeClassifierCV
    from aeon.transformations.collection.convolution_based import MiniRocket

    data = load_data()
    ytr, yva, yte = data["ytr"], data["yva"], data["yte"]
    got = {"train": len(data["Xtr"]), "val": len(data["Xva"]),
           "test": len(data["Xte"]), "T": int(data["Xtr"].shape[1])}
    assert got == EXPECTED, f"split identity FAILED: {got}"
    assert data["n_classes"] == N_CLASSES
    log(f"=== {DS}: {got} classes={N_CLASSES} "
        f"val_source={data['val_source']} ===")

    Xtr_z, Xva_z, Xte_z = (znorm(data["Xtr"]), znorm(data["Xva"]),
                           znorm(data["Xte"]))
    Xtrva_z = np.vstack([Xtr_z, Xva_z])
    n_tr = EXPECTED["train"]
    ytrva = np.concatenate([ytr, yva])

    # fixed MiniRocket (canonical: random_state=42, fit on train only)
    set_seed(MINIROCKET_SEED)
    extractor = MiniRocket(random_state=MINIROCKET_SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES
    assert not np.isnan(F_trva).any() and not np.isnan(F_te).any()
    assert not np.isinf(F_trva).any() and not np.isinf(F_te).any()

    # extractor-identity audit (raw PPV vs aeon)
    mr_id, valid = 0.0, None
    for c0 in range(0, len(Xtrva_z), 32):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 32])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 32]))))
        del act
    assert mr_id < 1e-5, f"extractor identity FAILED: {mr_id}"
    valid_het = valid[N_GLOBAL:]
    log(f"  [AUDIT] extractor identity max|diff|={mr_id:.2e}")

    # ---------------- MR arm (frozen control, deterministic) ----------------
    t0 = time.time()
    mr_val = macro_f1(yva, RidgeClassifierCV(alphas=ALPHAS).fit(
        F_trva[:n_tr], ytr).predict(F_trva[n_tr:]))
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(F_trva, ytrva)
    mr_pred = clf.predict(F_te).astype(np.int64)
    mr_test = macro_f1(yte, mr_pred)
    mr_alpha = float(clf.alpha_)
    mr_res = {"val_macro_f1": round(float(mr_val), 4),
              "test_macro_f1": round(float(mr_test), 4),
              "selected_alpha": mr_alpha,
              "accuracy": round(accuracy(yte, mr_pred), 4),
              "unique_pred_classes": int(len(np.unique(mr_pred))),
              "pred_class_balance": {int(c): int((mr_pred == c).sum())
                                     for c in np.unique(mr_pred)},
              "runtime_s": round(time.time() - t0, 1)}
    log(f"  [MR] val={mr_res['val_macro_f1']} test={mr_res['test_macro_f1']} "
        f"alpha={mr_alpha:.4f} ({mr_res['runtime_s']}s)")

    all_rows = []
    for seed in SEEDS:
        sdir = os.path.join(OUT, f"seed{seed}")
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, "mr_predictions.npy"), mr_pred)
        with open(os.path.join(sdir, "mr_result.json"), "w") as f:
            json.dump(mr_res, f, indent=2)

        # resume guard: a completed seed's result.json is valid --
        # never rerun a valid cell (audit-before-execution rule)
        rj = os.path.join(sdir, "heramba_r5_result.json")
        if os.path.exists(rj):
            with open(rj) as f:
                res = json.load(f)
            all_rows.append({"dataset": DS, "model": "HERAMBA R5", **res})
            log(f"  [R5 seed{seed}] RESUMED from disk: "
                f"test={res['test_macro_f1']} "
                f"delta={res['delta_heramba_minus_mr']:+.4f}")
            continue

        # ---------------- HERAMBA R5 arm (frozen recipe) ----------------
        t0 = time.time()
        set_seed(seed)
        model = RCMKNContextModel(n_classes=N_CLASSES).to(device)
        train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                         device, smoke=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        set_seed(seed)
        regimes_trva = extract_context_regimes(model, Xtrva_z, device,
                                               batch=64)
        set_seed(seed)
        regimes_te = extract_context_regimes(model, Xte_z, device, batch=64)
        set_seed(seed)
        det = bool(np.array_equal(
            extract_context_regimes(model, Xtrva_z[:64], device, batch=64),
            regimes_trva[:64]))
        H_trva = compute_H(extractor, Xtrva_z, regimes_trva, valid_het)
        H_te = compute_H(extractor, Xte_z, regimes_te, valid_het)
        X_trva_r2 = np.hstack([F_trva[:, :N_GLOBAL], H_trva])
        X_te_r2 = np.hstack([F_te[:, :N_GLOBAL], H_te])
        assert not np.isnan(X_trva_r2).any() and not np.isnan(X_te_r2).any()
        assert not np.isinf(X_trva_r2).any() and not np.isinf(X_te_r2).any()

        her_val = macro_f1(yva, RidgeClassifierCV(alphas=ALPHAS).fit(
            X_trva_r2[:n_tr], ytr).predict(X_trva_r2[n_tr:]))
        clf_h = RidgeClassifierCV(alphas=ALPHAS)
        clf_h.fit(X_trva_r2, ytrva)
        her_pred = clf_h.predict(X_te_r2).astype(np.int64)
        her_test = macro_f1(yte, her_pred)
        her_alpha = float(clf_h.alpha_)

        res = {"seed": seed,
               "val_macro_f1": round(float(her_val), 4),
               "test_macro_f1": round(float(her_test), 4),
               "selected_alpha": her_alpha,
               "accuracy": round(accuracy(yte, her_pred), 4),
               "unique_pred_classes": int(len(np.unique(her_pred))),
               "pred_class_balance": {int(c): int((her_pred == c).sum())
                                      for c in np.unique(her_pred)},
               "mr_val_macro_f1": mr_res["val_macro_f1"],
               "mr_test_macro_f1": mr_res["test_macro_f1"],
               "delta_heramba_minus_mr": round(float(her_test - mr_test), 4),
               "regimes_deterministic": det,
               "regime_hash_te": sha16(regimes_te),
               "context_train": train_info,
               "runtime_s": round(time.time() - t0, 1)}
        np.save(os.path.join(sdir, "heramba_r5_predictions.npy"), her_pred)
        with open(os.path.join(sdir, "heramba_r5_result.json"), "w") as f:
            json.dump(res, f, indent=2)
        all_rows.append({"dataset": DS, "model": "HERAMBA R5", **res})
        log(f"  [R5 seed{seed}] val={res['val_macro_f1']} "
            f"test={res['test_macro_f1']} alpha={her_alpha:.4f} "
            f"delta={res['delta_heramba_minus_mr']:+.4f} "
            f"({res['runtime_s']}s)")
        del model, H_trva, H_te
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_rows.append({"dataset": DS, "model": "MiniRocket", "seed": "ALL",
                     **mr_res})
    with open(os.path.join(OUT, "per_run_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    with open(os.path.join(OUT, "gates.json"), "w") as f:
        json.dump({"first_references": {"mr": mr_res["test_macro_f1"],
                                        "heramba_r5_seed42":
                                        all_rows[0]["test_macro_f1"]}},
                  f, indent=2)
    log("\nELECTRICDEVICES RUNS COMPLETE (NO FALLBACK)")


if __name__ == "__main__":
    main()
