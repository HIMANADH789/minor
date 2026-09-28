"""Runner: R2 architecture on the remaining canonical kaggle-UCR datasets
(ItalyPowerDemand, FordA) — seed 42.

Exactly the audited rcmkn_r2_gunpoint_seed42 pattern:
    load_kaggle_ucr (canonical split + stratified 15% val, seed 42)
    -> per-sample z-norm -> MiniRocket(random_state=42, fit TRAIN)
    -> M0 = full 9996-feature Ridge (reproduction gate vs the context3
       canonical reference, tol 0.0011)
    -> R2 context model (exact Haptics schedule: SSL -> HardVQ K=8,
       train-only SSL, val early-stop) -> frozen -> hard regimes
    -> [G || H] RidgeClassifierCV on train+val -> ONE official test eval
No DRTN-conditioned variant is produced (that branch is retired; R2 is the
final architecture). C1/C2 controls are included for the ablation table.
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

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    compute_regime_heterogeneity,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations, independent_heterogeneity_recompute,
    ppv_from_activations,
)
from experiments.drtn_conditioned_minirocket_context3_seed42.core import (
    DATASET_SPECS, load_kaggle_ucr, znorm,
)
from experiments.rcmkn_haptics_seed42.config import (
    SEED, N_FEATURES, N_GLOBAL, N_HET, K_CODES, ENCODER, VQ, JOINT,
)
from experiments.rcmkn_haptics_seed42.model import (
    RCMKNContextModel, parameter_report)
from experiments.rcmkn_haptics_seed42.runner import (
    train_context_model, extract_context_regimes, set_seed)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "results", "rcmkn_r2_kaggle_context2_seed42")
DATASETS = ["ItalyPowerDemand", "FordA"]
ALPHAS = np.logspace(-4, 4, 20)
M0_TOL = 0.0011

# context3 canonical M0 references (same loader/split/protocol)
M0_REF = {"ItalyPowerDemand": 0.9650, "FordA": 0.9499}

EXPECTED = {ds: {"train": DATASET_SPECS[ds]["train"] -
                 int(np.ceil(0.15 * DATASET_SPECS[ds]["train"])),
                 "val": int(np.ceil(0.15 * DATASET_SPECS[ds]["train"])),
                 "test": DATASET_SPECS[ds]["test"],
                 "T": DATASET_SPECS[ds]["T"],
                 "n_classes": DATASET_SPECS[ds]["n_classes"]}
            for ds in DATASETS}


def log(m):
    print(m, flush=True)


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def hetero_block(extractor, X_z, regimes, valid_het, chunk=32):
    N = len(X_z)
    H = np.empty((N, N_HET), dtype=np.float64)
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, _ = compute_raw_activations(extractor, X_z[c0:c1])
        H[c0:c1] = compute_regime_heterogeneity(act[:, N_GLOBAL:], valid_het,
                                                regimes[c0:c1])
        del act
    return H


def run_variant(name, blocks, yva, yte, ytrva, n_classes):
    F_tr = np.hstack([b[0] for b in blocks.values()])
    F_va = np.hstack([b[1] for b in blocks.values()])
    F_te = np.hstack([b[2] for b in blocks.values()])
    assert F_tr.shape[1] == F_va.shape[1] == F_te.shape[1] == N_FEATURES
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(F_tr, ytrva)
    pred_va = ridge.predict(F_va)
    pred_te = ridge.predict(F_te)
    return {
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "test_macro_f1": round(macro_f1(yte, pred_te), 4),
        "accuracy": round(float(accuracy_score(yte, pred_te)), 4),
        "selected_alpha": float(ridge.alpha_),
        "feature_dim": int(F_tr.shape[1]),
        "class_f1s": [round(float(x), 4) for x in f1_score(
            yte, pred_te, average=None, zero_division=0,
            labels=list(range(n_classes)))],
    }, pred_te


def run_dataset(ds_name, device, smoke=False):
    t0 = time.time()
    ds_dir = os.path.join(OUT_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)
    log(f"\n{'=' * 74}\n  R2 — {ds_name}\n{'=' * 74}")
    audits = {}

    data = load_kaggle_ucr(ds_name, seed=SEED)
    Xtr, ytr, Xva, yva, Xte, yte = (data["Xtr"], data["ytr"], data["Xva"],
                                    data["yva"], data["Xte"], data["yte"])
    ytrva = np.concatenate([ytr, yva])
    got = {"train": len(Xtr), "val": len(Xva), "test": len(Xte),
           "T": data["L"], "n_classes": data["n_classes"]}
    assert got == EXPECTED[ds_name], f"AUDIT1 {ds_name}: {got}"
    audits["audit1_config"] = {"expected": EXPECTED[ds_name],
                               "actual": got, "pass": True}
    audits["provenance"] = {"source_path": data["source_path"],
                            "val_source": data["val_source"]}
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z])

    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES
    G_trva, G_te = F_trva[:, :N_GLOBAL], F_te[:, :N_GLOBAL]
    G_va = G_trva[len(Xtr):]

    valid = None
    mr_id = 0.0
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5, f"extractor identity {mr_id}"
    valid_het = valid[N_GLOBAL:]
    audits["audit2_extractor_identity_maxdiff"] = float(mr_id)

    # M0 + reproduction gate
    results, preds = {}, {}
    results["M0"], preds["M0"] = run_variant(
        "M0", {"all": (F_trva, F_trva[len(Xtr):], F_te)},
        yva, yte, ytrva, data["n_classes"])
    gate = abs(results["M0"]["test_macro_f1"] - M0_REF[ds_name]) <= M0_TOL
    audits["m0_gate"] = {"reference": M0_REF[ds_name],
                         "observed": results["M0"]["test_macro_f1"],
                         "pass": bool(gate)}
    log(f"  [M0 GATE] {results['M0']['test_macro_f1']:.4f} vs "
        f"{M0_REF[ds_name]} -> {'PASS' if gate else 'FAIL'}")
    assert gate, f"M0 gate failed on {ds_name}"

    # R2 context model (exact Haptics schedule)
    set_seed(SEED)
    model = RCMKNContextModel(n_classes=data["n_classes"]).to(device)
    params = parameter_report(model)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=smoke)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.save({"model_state": model.state_dict(),
                "config": {"encoder": ENCODER, "vq": VQ, "joint": JOINT},
                "train_info": train_info, "params": params},
               os.path.join(ds_dir, "context_model_seed42.pt"))

    with torch.no_grad():
        xs = torch.from_numpy(Xtr_z[:1])[:, None, :].to(device)
        z1 = model.encoder(xs)
        x2 = xs.clone()
        x2[:, :, got["T"] // 2:] = torch.randn_like(x2[:, :, got["T"] // 2:])
        caus = float((z1[:, :got["T"] // 2] -
                      model.encoder(x2)[:, :got["T"] // 2]).abs().max())
    assert caus == 0.0
    audits["audit3_causality"] = float(caus)

    set_seed(SEED)
    reg_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    set_seed(SEED)
    reg_te = extract_context_regimes(model, Xte_z, device, batch=32)
    set_seed(SEED)
    assert np.array_equal(
        extract_context_regimes(model, Xtrva_z, device, batch=32), reg_trva)
    audits["audit4_regimes_deterministic"] = True

    H_trva = hetero_block(extractor, Xtrva_z, reg_trva, valid_het)
    H_te = hetero_block(extractor, Xte_z, reg_te, valid_het)
    H_va = H_trva[len(Xtr):]

    smp, val = compute_raw_activations(extractor, Xtrva_z[0:1])
    rec = 0.0
    for m in np.linspace(0, N_HET - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[0, N_GLOBAL:][m], val[N_GLOBAL:][m],
            reg_trva[0].astype(np.int64), K=K_CODES)
        rec = max(rec, abs(ref - float(H_trva[0, m])))
    assert rec <= 1e-6
    audits["audit5_H_recompute_maxdiff"] = float(rec)

    log("  [R2] SSL context + hard-VQ heterogeneity")
    results["R2"], preds["R2"] = run_variant(
        "R2", {"global": (G_trva, G_va, G_te), "het": (H_trva, H_va, H_te)},
        yva, yte, ytrva, data["n_classes"])
    results["R2"]["context_train"] = train_info
    results["R2"]["params"] = params

    deltas = {f"R2-{k}": round(results["R2"]["test_macro_f1"] -
                               results[k]["test_macro_f1"], 4)
              for k in ("M0",)}

    os.makedirs(os.path.join(ds_dir, "predictions"), exist_ok=True)
    with open(os.path.join(ds_dir, "predictions", f"{ds_name}_seed42.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "true_class", "M0", "R2"])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), int(preds["M0"][i]),
                        int(preds["R2"][i])])

    res = {
        "dataset": ds_name, "seed": SEED, "T": got["T"],
        "split": {"train": got["train"], "val": got["val"],
                  "test": got["test"]},
        "n_classes": data["n_classes"], "n_features": N_FEATURES,
        "K_regimes": K_CODES,
        "results": results, "deltas": deltas,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ds_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(ds_dir, "audits.json"), "w") as f:
        json.dump(audits, f, indent=2)
    log(f"  [{ds_name}] done: M0={results['M0']['test_macro_f1']} "
        f"R2={results['R2']['test_macro_f1']} "
        f"({time.time() - t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    all_res = {ds: run_dataset(ds, device, args.smoke) for ds in
               args.datasets}
    with open(os.path.join(OUT_DIR, "per_dataset_results.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    log("\nRESULTS (test Macro-F1, seed 42)")
    for ds, r in all_res.items():
        log(f"  {ds}: M0={r['results']['M0']['test_macro_f1']} "
            f"R2={r['results']['R2']['test_macro_f1']}")


if __name__ == "__main__":
    main()
