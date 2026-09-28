"""Core pipeline for the R2 Haptics 3-seed robustness experiment.

Every stage imports the AUDITED R2 implementations directly (no copies):
    MiniROCKET raw activations: drtn_conditioned_minirocket_transfer_seed42.core
    Heterogeneity H:            drtn_conditioned_minirocket_haptics_3seed.runner
    SSL encoder / VQ / model:   rcmkn_haptics_seed42 (config, model, vq)
    Ridge protocol:             RidgeClassifierCV(alphas=logspace(-4,4,20)),
                                fit on train+val, evaluated once per seed.

Seed policy (spec section 5):
    * MiniROCKET extractor: random_state=42 for ALL seeds (fixed bank)
    * outer seed s in {42,43,44}: SSL encoder init/training, VQ init/training,
      and every other stochastic R2 training component
"""
import hashlib
import time

import numpy as np
import torch

from experiments.external_stack_generalization.data import load_dataset
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations,
    ppv_from_activations,
    independent_heterogeneity_recompute,
)
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    compute_regime_heterogeneity,
)
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel, parameter_report
from experiments.rcmkn_haptics_seed42.runner import (
    train_context_model, extract_context_regimes, macro_f1, set_seed,
)
from experiments.rcmkn_haptics_seed42.vq import occupancy_stats
from experiments.rcmkn_r2_haptics_3seed.config import (
    ALPHAS, DATASET, M0_REF, MINIROCKET_SEED, N_FEATURES, N_GLOBAL, N_HET,
)


def log(msg):
    print(msg, flush=True)


def sha16(a):
    """Stable content hash of an integer array (provenance / identity)."""
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()
                          ).hexdigest()[:16]


def znorm(X):
    """Per-sample z-normalization -- the canonical R2 preprocessing."""
    mu = X.mean(axis=-1, keepdims=True)
    sig = X.std(axis=-1, keepdims=True) + 1e-8
    return ((X - mu) / sig).astype(np.float32)


# ---------------------------------------------------------------------------
# Data + fixed MiniROCKET
# ---------------------------------------------------------------------------
def load_data():
    """Canonical Haptics split (AUDIT 1/2). Never reshuffled."""
    data = load_dataset(DATASET)
    expected = {"train": 132, "val": 23, "test": 308, "T": 1092,
                "n_classes": 5}
    got = {"train": len(data["Xtr"]), "val": len(data["Xva"]),
           "test": len(data["Xte"]), "T": data["L"],
           "n_classes": data["n_classes"]}
    assert got == expected, f"AUDIT 1 FAILED: dataset identity {got} != {expected}"
    return data


def verify_znorm_invariant(data):
    """AUDIT 3: per-sample z-normalization matches the canonical R2 transform
    and is deterministic (identical across calls / seeds)."""
    X = data["Xtr"][:3]
    z1 = znorm(X)
    z2 = znorm(X.copy())
    assert np.array_equal(z1, z2), "AUDIT 3 FAILED: znorm not deterministic"
    mu = z1.mean(axis=-1)
    sd = z1.std(axis=-1)
    assert np.all(np.abs(mu) < 1e-4) and np.all(np.abs(sd - 1) < 1e-3), \
        "AUDIT 3 FAILED: znorm output is not per-sample standardized"
    # must match the audited transfer-seed42 znorm exactly
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as tr
    z_ref = tr.znorm(X)
    assert np.array_equal(z1, z_ref), "AUDIT 3 FAILED: znorm differs from R2"
    return {"pass": True, "max_abs_mean": float(np.abs(mu).max()),
            "max_abs_std_err": float(np.abs(sd - 1).max()),
            "identical_to_r2_znorm": True}


def build_fixed_extractor(Xtr_z):
    """Canonical MiniROCKET, random_state=42, fit on TRAIN only.

    Identical construction for every outer seed (spec section 3).
    """
    from aeon.transformations.collection.convolution_based import MiniRocket
    set_seed(MINIROCKET_SEED)
    extractor = MiniRocket(random_state=MINIROCKET_SEED, n_jobs=-1)
    extractor.fit(Xtr_z[:, None, :].astype(np.float32))
    return extractor


def compute_global_block(extractor, X_z):
    """G = canonical aeon transform, first 4998 kernels (R2's exact block)."""
    F = extractor.transform(X_z[:, None, :].astype(np.float32))
    assert F.shape[1] == N_FEATURES, f"expected {N_FEATURES}, got {F.shape[1]}"
    assert not np.isnan(F).any() and not np.isinf(F).any()
    return F[:, :N_GLOBAL]


def stack_trva(Xtr_z, Xva_z):
    return np.vstack([Xtr_z, Xva_z]).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-seed R2 context pipeline (the ONLY thing that varies with the seed)
# ---------------------------------------------------------------------------
def run_context_for_seed(seed, Xtr_z, ytr, Xva_z, yva, device):
    """Train the exact R2 SSL+VQ context model for one outer seed.

    Uses the audited train_context_model unchanged (same architecture, same
    SSL masking, same joint fine-tune, same early stopping). Only the seed
    differs. Returns (model, train_info, regimes_trva, regimes_te).
    """
    set_seed(seed)
    model = RCMKNContextModel(n_classes=5).to(device)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Xtrva_z = stack_trva(Xtr_z, Xva_z)
    set_seed(seed)
    regimes_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    return model, train_info, Xtrva_z, regimes_trva


# ---------------------------------------------------------------------------
# Heterogeneity block (audited formula; identical across seeds given regimes)
# ---------------------------------------------------------------------------
def compute_H(extractor, X_z, regimes, valid_het, chunk=64):
    """Chunked audited H (N, 4998) via compute_regime_heterogeneity."""
    N = X_z.shape[0]
    H = np.empty((N, N_HET), dtype=np.float64)
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, _ = compute_raw_activations(extractor, X_z[c0:c1])
        H[c0:c1] = compute_regime_heterogeneity(
            act[:, N_GLOBAL:], valid_het, regimes[c0:c1])
        del act
    return H


def compute_valid_het(extractor, X_probe):
    """Per-feature valid-region mask from the audited raw extractor."""
    _, valid = compute_raw_activations(extractor, X_probe)
    return valid[N_GLOBAL:]


def h_block_diagnostics(H):
    H = np.asarray(H, dtype=np.float64)
    return {"mean": float(H.mean()), "std": float(H.std()),
            "min": float(H.min()), "max": float(H.max()),
            "zero_fraction": float((H == 0).mean())}


# ---------------------------------------------------------------------------
# Ridge protocol (identical to validated R2; train+val fit, one test eval)
# ---------------------------------------------------------------------------
def ridge_fit_eval(G_trva, H_trva, G_te, H_te, n_train, yva, yte, ytrva,
                   n_classes):
    """Canonical R2 Ridge protocol: fit on train+val rows, predict val and
    test. The validation rows are the LAST n_val rows of the trainva block
    (canonical split ordering), so yva is the val label vector."""
    F_tr = np.hstack([G_trva, H_trva])
    F_te = np.hstack([G_te, H_te])
    assert F_tr.shape[1] == F_te.shape[1] == N_FEATURES
    assert F_tr.shape[0] == len(ytrva), \
        f"train rows {F_tr.shape[0]} != labels {len(ytrva)}"
    assert F_te.shape[0] == len(yte), \
        f"test rows {F_te.shape[0]} != labels {len(yte)}"
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import accuracy_score, f1_score
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(F_tr, ytrva)                      # train + validation
    F_va = F_tr[n_train:]                       # validation rows
    pred_va = ridge.predict(F_va)
    pred_te = ridge.predict(F_te)               # OFFICIAL test eval: once
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


# re-export so the runner can use a single import surface
__all__ = ["parameter_report"]
