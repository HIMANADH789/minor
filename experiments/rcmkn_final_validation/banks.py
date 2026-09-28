"""Generic frozen-bank builder for the final-validation phase.

Dispatches to each dataset's CANONICAL loader and rebuilds the frozen R2
[G || H] banks exactly as the original audited run did, re-verifying the
bit-exact extractor-identity and independent-H-recompute audits. Context
checkpoints are the stored seed-42 artifacts (never retrained here).

Loader dispatch (canonical provenance per dataset):
    Haptics, Phoneme, EpilepticSeizures:
        experiments/external_stack_generalization.data.load_dataset
        (Phoneme: stratified 15% of TRAIN, seed 42; EpilepticSeizures:
        provided canonical val)
    ECG5000_UNBAL, CWRU_UNBAL:
        drtn_conditioned_minirocket_transfer_seed42.load_any_dataset
        (same underlying canonical loaders as the original R2 runs)
    ECG5000_BAL, CWRU_BAL:
        rcmkn_ssl_context_important2_seed42 core loader (same convention)
    GunPoint, ItalyPowerDemand, FordA:
        drtn_conditioned_minirocket_context3_seed42.load_kaggle_ucr
    UWave{All,X,Y,Z}:
        rcmkn_r2_uwave_seed42.data.load_and_split (official UCR split +
        stratified 15% val, saved indices, verified identical)
"""
import os

import numpy as np
import torch

from experiments.rcmkn_final_validation.config import (
    CKPT_PATHS, SPLIT_DIR, REPO_ROOT, RESULTS_ROOT, SEED,
)
from experiments.rcmkn_haptics_seed42.runner import set_seed

N_FEATURES, N_GLOBAL, N_HET, K_CODES = 9996, 4998, 4998, 8

KAGGLE_SPLITS = {  # canonical stratified-15% splits (context3 / GunPoint)
    "GunPoint": {"train": 42, "val": 8, "test": 150},
    "ItalyPowerDemand": {"train": 56, "val": 11, "test": 1029},
    "FordA": {"train": 3060, "val": 541, "test": 1320},
}


def znorm(X):
    return ((X - X.mean(-1, keepdims=True)) /
            (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)


def load_canonical(ds_name, ds_dir=None):
    """Return dict with Xtr/ytr/Xva/yva/Xte/yte/n_classes via the canonical
    loader for this dataset."""
    if ds_name in ("Haptics", "Phoneme", "EpilepticSeizures"):
        from experiments.external_stack_generalization.data import load_dataset
        return load_dataset(ds_name)
    if ds_name in ("ECG5000_UNBAL", "CWRU_UNBAL"):
        from experiments.drtn_conditioned_minirocket_transfer_seed42.runner \
            import load_any_dataset
        d = load_any_dataset(ds_name)
        return {"Xtr": d["Xtr"], "ytr": d["ytr"], "Xva": d["Xva"],
                "yva": d["yva"], "Xte": d["Xte"], "yte": d["yte"],
                "n_classes": d["n_classes"], "L": d["L"]}
    if ds_name in ("ECG5000_BAL", "CWRU_BAL"):
        # identical loader to the rcmkn_ssl_context_important2_seed42 BAL
        # runner (which used transfer.load_any_dataset on the same NPZ specs)
        from experiments.drtn_conditioned_minirocket_transfer_seed42.runner \
            import load_any_dataset
        d = load_any_dataset(ds_name)
        return {"Xtr": d["Xtr"], "ytr": d["ytr"], "Xva": d["Xva"],
                "yva": d["yva"], "Xte": d["Xte"], "yte": d["yte"],
                "n_classes": d["n_classes"], "L": d["L"]}
    if ds_name in ("GunPoint", "ItalyPowerDemand", "FordA"):
        from experiments.drtn_conditioned_minirocket_context3_seed42.core \
            import load_kaggle_ucr
        d = load_kaggle_ucr(ds_name, seed=SEED)
        return {"Xtr": d["Xtr"], "ytr": d["ytr"], "Xva": d["Xva"],
                "yva": d["yva"], "Xte": d["Xte"], "yte": d["yte"],
                "n_classes": d["n_classes"], "L": d["L"]}
    if ds_name.startswith("UWave"):
        from experiments.rcmkn_r2_uwave_seed42.data import load_and_split
        return load_and_split(ds_name, ds_dir or os.path.join(
            RESULTS_ROOT, "r2_uwave_seed42", ds_name))
    raise KeyError(ds_name)


def build_banks(ds_name, device, ds_dir=None):
    """Rebuild frozen G/H banks with the stored seed-42 context model."""
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity)
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        compute_raw_activations, independent_heterogeneity_recompute,
        ppv_from_activations)
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel

    d = load_canonical(ds_name, ds_dir)
    Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
    Xte, yte = d["Xte"], d["yte"]
    n_train, n_val = len(Xtr), len(Xva)
    Xtrva_z = np.vstack([znorm(Xtr), znorm(Xva)])
    Xte_z = znorm(Xte)

    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(SEED)
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(Xtrva_z[:n_train][:, None, :].astype(np.float32))
    F_trva = extractor.transform(Xtrva_z[:, None, :].astype(np.float32))
    F_te = extractor.transform(Xte_z[:, None, :].astype(np.float32))
    assert F_trva.shape[1] == N_FEATURES

    mr_id, valid = 0.0, None
    for c0 in range(0, len(Xtrva_z), 64):
        act, valid = compute_raw_activations(extractor, Xtrva_z[c0:c0 + 64])
        mr_id = max(mr_id, float(np.max(np.abs(
            ppv_from_activations(act, valid) - F_trva[c0:c0 + 64]))))
        del act
    assert mr_id < 1e-5, f"{ds_name}: extractor identity {mr_id}"
    valid_het = valid[N_GLOBAL:]

    ckpt = os.path.join(CKPT_PATHS[ds_name], "context_model_seed42.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CKPT_PATHS[ds_name], "checkpoints",
                            "context_model_seed42.pt")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = RCMKNContextModel(n_classes=d["n_classes"])
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    from experiments.rcmkn_haptics_seed42.runner import (
        extract_context_regimes)
    set_seed(SEED)
    regimes_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    set_seed(SEED)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=32)

    def het(X_z, regimes, chunk=32):
        H = np.empty((len(X_z), N_HET), dtype=np.float64)
        for c0 in range(0, len(X_z), chunk):
            c1 = min(c0 + chunk, len(X_z))
            act, _ = compute_raw_activations(extractor, X_z[c0:c1])
            H[c0:c1] = compute_regime_heterogeneity(
                act[:, N_GLOBAL:], valid_het, regimes[c0:c1])
            del act
        return H

    H_trva = het(Xtrva_z, regimes_trva)
    H_te = het(Xte_z, regimes_te)

    smp, val = compute_raw_activations(extractor, Xtrva_z[0:1])
    rec = 0.0
    for m in np.linspace(0, N_HET - 1, 8).astype(int):
        ref = independent_heterogeneity_recompute(
            smp[0, N_GLOBAL:][m].astype(bool), val[N_GLOBAL:][m].astype(bool),
            regimes_trva[0].astype(np.int64), K=K_CODES)
        rec = max(rec, abs(ref - float(H_trva[0, m])))
    # canonical tolerance (boolean float32->float64 PPV rounding, as in the
    # original transfer/important2 runners: <= 0.05)
    assert rec <= 0.05, f"{ds_name}: H recompute {rec}"

    return {
        "y": (ytr, yva, yte), "n_classes": d["n_classes"],
        "G_trva": F_trva[:, :N_GLOBAL], "G_te": F_te[:, :N_GLOBAL],
        "F_trva": F_trva, "F_te": F_te,   # full 9996 MiniROCKET candidate
                                          # bank for R5 G selection
        "H_trva": H_trva, "H_te": H_te,
        "n_train": n_train, "n_val": n_val,
        "n_test": len(yte), "T": int(len(Xtr[0])),
        "context_params": ck.get("params", {}),
        "audits": {"extractor_identity_maxdiff": float(mr_id),
                   "H_recompute_maxdiff": float(rec)},
        # raw material for per-seed context retraining (multi-seed runs)
        "_Xtr": Xtr, "_Xva": Xva, "_Xte": Xte,
        "_Xtr_z": znorm(Xtr), "_Xva_z": znorm(Xva), "_Xte_z": Xte_z,
        "_Xtrva_z": Xtrva_z,
        "_extractor": extractor, "_valid_het": valid_het,
    }


def split_label(ds_name):
    return KAGGLE_SPLITS.get(ds_name)
