"""Wafer / ECG5000_UNBAL / ItalyPowerDemand -- MiniRocket vs HERAMBA R5, 3 seeds.

Scope (task): EXACTLY Wafer, ECG5000 (user-selected canonical variant
ECG5000_UNBAL), ItalyPowerDemand x {MR, HERAMBA R5} x {42,43,44} = 18 cells.

Audit facts (Phase 1, results/wafer_ecg5000_ipd_r5/AUDIT.md):
    Wafer            : UCR TSVs (verified UCRArchive_2018) 1000/6164, T=152,
                       2 classes {-1,+1}; no stored MR/R2 reference ->
                       seed-42 values become the first audited references
                       (repo convention, cf. Phoneme in the transfer line).
    ECG5000_UNBAL    : data/ecg5000_resplit.npz -> 3400/600/1000, T=140,
                       5 classes (load_npz_dataset, stratified 15% @ seed42);
                       canonical M0 = 0.5938 (corrected_negative_retest,
                       val 0.8224, alpha 11.2884, tol 0.0011) and
                       canonical R2 = 0.5894 (rcmkn_ssl_context_transfer_
                       seed42, val 0.8170, tol 0.011).
    ItalyPowerDemand : UCR TSVs (sha-identical to data/kaggle copy) 67/1029,
                       T=24, 2 classes; stratified 15% of train -> 56/11/1029;
                       canonical M0 = 0.9650 (val 1.0, tol 0.0011) and
                       canonical R2 = 0.9592 (val 1.0, tol 0.011).

Frozen recipe (identical to the audited rcmkn/gunpoint_phoneme_forda lines):
    MR        : aeon MiniRocket(random_state=42, fit TRAIN only) -> 9996
                features; RidgeClassifierCV(logspace(-4,4,20)) train-only fit
                -> val diagnostic; train+val refit -> ONE test evaluation.
                Deterministic across seeds (canonical convention); seed-42
                gated vs the stored references.
    HERAMBA R5: per-seed RCMKNContextModel (SSL+VQ K=8+joint; frozen config
                byte-identical to rcmkn_haptics_seed42/config.json and the
                stored transfer/important2 configs), H = occupancy-weighted
                regime heterogeneity (min_occ 0.01), X_R5 = [G 4998 || H
                4998], same Ridge/alpha grid, train-only fit -> val
                diagnostic; train+val refit -> ONE test evaluation. Seed-42
                gated vs the canonical R2 (Wafer: first reference).

FALLBACK PROTOCOL (task-defined, validation-stage only, identical to
gunpoint_phoneme_forda_r5):
    final arm for a seed = HERAMBA candidate iff its TRAIN-ONLY-fit
    validation Macro-F1 >= MR's train-only-fit validation Macro-F1 on the
    SAME frozen features/split; else the identical MR model. Selection uses
    validation ONLY; the final selected model is evaluated on test exactly
    ONCE. Raw-HERAMBA test metrics are recorded for audit but are NEVER
    used for selection; no max() over test scores. A negative final delta
    after a correct validation-stage selection is reported as-is (never
    clamped).

No test-driven tuning; R5 frozen; MR frozen; existing artifacts untouched.
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
OUT = os.path.join(ROOT, "results", "wafer_ecg5000_ipd_r5")

SEEDS = [42, 43, 44]
N_FEATURES, N_GLOBAL, N_HET = 9996, 4998, 4998
MINIROCKET_SEED = 42
ALPHAS = np.logspace(-4, 4, 20)

ARCHIVE = os.path.join("data", "kaggle", "_ucrarchive_2018",
                       "UCRArchive_2018", "UCRArchive_2018")

SPEC = {
    "Wafer": {
        "loader": "ucr", "n_classes": 2,
        "expected": {"train": 850, "val": 150, "test": 6164, "T": 152},
        "mr_ref": None, "mr_tol": 0.0011,
        "r2_ref": None, "r2_tol": 0.011,
    },
    "ECG5000_UNBAL": {
        "loader": "npz", "n_classes": 5,
        "expected": {"train": 3400, "val": 600, "test": 1000, "T": 140},
        "mr_ref": 0.5938, "mr_tol": 0.0011,
        "r2_ref": 0.5894, "r2_tol": 0.011,
    },
    "ItalyPowerDemand": {
        "loader": "ucr", "n_classes": 2,
        "expected": {"train": 56, "val": 11, "test": 1029, "T": 24},
        "mr_ref": 0.9650, "mr_tol": 0.0011,
        "r2_ref": 0.9592, "r2_tol": 0.011,
    },
}


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


def load_ucr(ds_name):
    """Canonical UCR TSVs + stratified 15% of train as val (seed 42).

    Mirrors drtn_conditioned_minirocket_context3_seed42.load_kaggle_ucr
    exactly (same split rule, same label remap, same sha recording).
    """
    from sklearn.model_selection import train_test_split
    d_dir = os.path.join(ROOT, ARCHIVE, ds_name)
    tr = np.loadtxt(os.path.join(d_dir, f"{ds_name}_TRAIN.tsv"),
                    delimiter="\t", ndmin=2)
    te = np.loadtxt(os.path.join(d_dir, f"{ds_name}_TEST.tsv"),
                    delimiter="\t", ndmin=2)
    ytr, Xtr = tr[:, 0].astype(int), tr[:, 1:].astype(np.float32)
    yte, Xte = te[:, 0].astype(int), te[:, 1:].astype(np.float32)

    labels = sorted(set(ytr.tolist()) | set(yte.tolist()))
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)
    lut = {lab: i for i, lab in enumerate(labels)}
    map_fn = np.vectorize(lut.get)
    return {
        "name": ds_name, "source_path": os.path.relpath(d_dir, ROOT),
        "Xtr": Xtr, "ytr": map_fn(ytr), "Xva": Xva, "yva": map_fn(yva),
        "Xte": Xte, "yte": map_fn(yte),
        "n_classes": len(labels), "L": int(Xtr.shape[1]),
        "label_map": {str(k): v for k, v in lut.items()},
        "val_source": "stratified_15pct_of_train_seed42",
    }


def load_data(ds_name):
    if SPEC[ds_name]["loader"] == "npz":
        from experiments.drtn_conditioned_minirocket_transfer_seed42.runner \
            import load_npz_dataset
        d = load_npz_dataset(ds_name)
    else:
        d = load_ucr(ds_name)
    out = {k: d[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte",
                             "n_classes", "val_source")}
    out["source_path"] = d.get("source_path", d.get("provenance", "?"))
    return out


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

    n_tr = spec["expected"]["train"]
    Xtr_z, Xva_z = Xtrva_z[:n_tr], Xtrva_z[n_tr:]
    t0 = time.time()

    # ---------------- MR arm (frozen control) ----------------
    mr_val = macro_f1(yva, RidgeClassifierCV(alphas=ALPHAS).fit(
        F_trva[:n_tr], ytr).predict(F_trva[n_tr:]))
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(F_trva, np.concatenate([ytr, yva]))
    mr_pred = clf.predict(F_te).astype(np.int64)
    mr_test = macro_f1(yte, mr_pred)
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
    assert not np.isinf(X_trva_r2).any() and not np.isinf(X_te_r2).any()

    # raw-HERAMBA validation (train-only fit) -- fallback criterion input
    her_val = macro_f1(yva, RidgeClassifierCV(alphas=ALPHAS).fit(
        X_trva_r2[:n_tr], ytr).predict(X_trva_r2[n_tr:]))
    # raw-HERAMBA refit test (audit only -- NEVER used for selection)
    clf_h = RidgeClassifierCV(alphas=ALPHAS)
    clf_h.fit(X_trva_r2, np.concatenate([ytr, yva]))
    her_pred = clf_h.predict(X_te_r2).astype(np.int64)
    her_test = macro_f1(yte, her_pred)
    her_alpha = float(clf_h.alpha_)

    # ---------------- VALIDATION-STAGE FALLBACK ----------------
    select_heramba = bool(her_val >= mr_val)
    final_test = her_test if select_heramba else mr_test
    final_pred = her_pred if select_heramba else mr_pred
    final_alpha = her_alpha if select_heramba else mr_alpha
    del model, H_trva, H_te
    if device.type == "cuda":
        torch.cuda.empty_cache()

    res = {
        "seed": seed,
        "mr_val": round(float(mr_val), 4), "mr_test": round(float(mr_test), 4),
        "mr_alpha": mr_alpha, "mr_acc": round(accuracy(yte, mr_pred), 4),
        "her_val": round(float(her_val), 4),
        "her_test": round(float(her_test), 4), "her_alpha": her_alpha,
        "her_acc": round(accuracy(yte, her_pred), 4),
        "selected": "HERAMBA" if select_heramba else "MiniRocket",
        "fallback_used": not select_heramba,
        "final_test": round(float(final_test), 4),
        "final_alpha": final_alpha,
        "final_acc": round(accuracy(yte, final_pred), 4),
        "final_unique_pred_classes": int(len(np.unique(final_pred))),
        "final_pred_class_balance": {int(c): int((final_pred == c).sum())
                                     for c in np.unique(final_pred)},
        "regimes_deterministic": det, "regime_hash_te": sha16(regimes_te),
        "context_train": train_info,
        "runtime_s": round(time.time() - t0, 1),
    }
    return res, final_pred, her_pred, mr_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    log(f"WAFER/ECG5000_UNBAL/IPD -- MR vs HERAMBA R5 -- seeds={SEEDS} "
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
        log(f"\n=== {ds_name}: {got} classes={spec['n_classes']} "
            f"val_source={data['val_source']} ===")

        Xtr_z, Xva_z, Xte_z = (znorm(data["Xtr"]), znorm(data["Xva"]),
                               znorm(data["Xte"]))
        Xtrva_z = np.vstack([Xtr_z, Xva_z])
        set_seed(MINIROCKET_SEED)
        extractor = MiniRocket(random_state=MINIROCKET_SEED, n_jobs=-1)
        extractor.fit(Xtr_z[:, None, :].astype(np.float32))
        F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
        F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
        assert F_trva.shape[1] == N_FEATURES
        assert not np.isnan(F_trva).any() and not np.isnan(F_te).any()
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
            sdir = os.path.join(ds_dir, f"seed{seed}")
            os.makedirs(sdir, exist_ok=True)
            # resume guard: a completed seed's result.json is valid --
            # never rerun a valid cell (audit-before-execution rule)
            rj = os.path.join(sdir, "result.json")
            if os.path.exists(rj):
                with open(rj) as f:
                    res = json.load(f)
                all_rows.append({"dataset": ds_name, **res})
                log(f"  [seed{seed}] RESUMED from disk: "
                    f"selected={res['selected']} final={res['final_test']}")
                continue
            res, final_pred, her_pred, mr_pred = run_seed(
                ds_name, spec, seed, data, extractor, valid_het, Xtrva_z,
                Xte_z, F_trva, F_te, ytr, yva, yte, device)
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

        # ---- seed-42 gates ----
        s42 = next(r for r in all_rows
                   if r["dataset"] == ds_name and r["seed"] == 42)
        gate_log = {}
        if spec["mr_ref"] is not None:
            ok = abs(s42["mr_test"] - spec["mr_ref"]) <= spec["mr_tol"]
            gate_log["mr_gate"] = {"reference": spec["mr_ref"],
                                   "observed": s42["mr_test"],
                                   "pass": bool(ok)}
            log(f"  [MR GATE {ds_name}] {s42['mr_test']} vs "
                f"{spec['mr_ref']} -> {'PASS' if ok else 'FAIL'}")
            assert ok, f"MR gate FAILED on {ds_name}"
        else:
            gate_log["mr_gate"] = {"first_reference": s42["mr_test"]}
            log(f"  [MR REF {ds_name}] first audited reference "
                f"{s42['mr_test']}")
        if spec["r2_ref"] is not None:
            ok = abs(s42["her_test"] - spec["r2_ref"]) <= spec["r2_tol"]
            gate_log["r2_gate"] = {"reference": spec["r2_ref"],
                                   "observed": s42["her_test"],
                                   "pass": bool(ok)}
            log(f"  [R5 GATE {ds_name}] raw HERAMBA {s42['her_test']} vs "
                f"canonical R2 {spec['r2_ref']} -> {'PASS' if ok else 'FAIL'}")
            assert ok, f"R5 gate FAILED on {ds_name}"
        else:
            gate_log["r2_gate"] = {"first_reference": s42["her_test"]}
            log(f"  [R5 REF {ds_name}] first audited reference "
                f"{s42['her_test']}")
        with open(os.path.join(ds_dir, "gates.json"), "w") as f:
            json.dump(gate_log, f, indent=2)

    with open(os.path.join(OUT, "per_run_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    log("\nWAFER/ECG5000/IPD RUNS COMPLETE")


if __name__ == "__main__":
    main()
