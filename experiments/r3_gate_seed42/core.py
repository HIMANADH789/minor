"""R3 core: frozen-R2 context + audited G/H features for two datasets.

Scope (user-narrowed): Haptics + ECG5000_BAL, seed 42.
Haptics     = R2's strongest win (R2 0.5500 vs M0 0.4974) -> gate should OPEN
ECG5000_BAL = R2's clear harm   (R2 0.6089 vs M0 0.6553) -> gate should CLOSE

Everything frozen is imported from the audited implementations; stages 1-3
are never retrained or reconfigured here.
"""

import hashlib
import os

import numpy as np
import torch

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    compute_regime_heterogeneity,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations, ppv_from_activations,
)
import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as transfer

from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
from experiments.rcmkn_haptics_seed42.runner import (
    extract_context_regimes, macro_f1, set_seed,
)
from experiments.rcmkn_haptics_seed42.vq import occupancy_stats

SEED = 42
K_CODES = 8
N_G = 4998
N_H = 4998
N_TOTAL = 9996
MIN_OCCUPANCY = 0.01
THETA_INIT = -2.0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "r3_gate_seed42")

DATASETS = {
    "Haptics": {
        "ckpt": os.path.join(RESULTS_ROOT, "rcmkn_haptics_seed42",
                             "context_model_seed42.pt"),
        "ckpt_hash": "2dbb0cf4f3db3df8",
        "expected": {"train": 132, "val": 23, "test": 308, "T": 1092,
                     "n_classes": 5},
        "M0": 0.4974, "R2": 0.5500,
    },
    "ECG5000_BAL": {
        "ckpt": os.path.join(RESULTS_ROOT,
                             "rcmkn_ssl_context_important2_seed42",
                             "ECG5000_BAL", "context_model_seed42.pt"),
        "ckpt_hash": "f2eda1b2ff0bbe99",
        "expected": {"train": 5226, "val": 923, "test": 1000, "T": 140,
                     "n_classes": 5},
        "M0": 0.6553, "R2": 0.6089,
    },
}


def log(msg):
    print(msg, flush=True)


def sha16(a):
    if isinstance(a, (bytes, bytearray)):
        return hashlib.sha256(a).hexdigest()[:16]
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ #
# data + frozen context                                               #
# ------------------------------------------------------------------ #
def load_frozen_context(ds_name, device):
    """Canonical split + frozen audited R2 context (never trained here)."""
    info = DATASETS[ds_name]
    data = transfer.load_any_dataset(ds_name)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    znorm = transfer.znorm
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z]).astype(np.float32)
    ytrva = np.concatenate([ytr, yva])

    model = RCMKNContextModel(n_classes=info["expected"]["n_classes"])
    ck = torch.load(info["ckpt"], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    set_seed(SEED)
    regimes_trva = extract_context_regimes(model, Xtrva_z, device, batch=32)
    regimes_te = extract_context_regimes(model, Xte_z, device, batch=32)
    return {
        "data": data, "model": model,
        "Xtrva_z": Xtrva_z, "Xte_z": Xte_z,
        "ytrva": ytrva, "yva": yva, "yte": yte,
        "ytr": ytr, "n_train": len(Xtr),
        "regimes_trva": regimes_trva, "regimes_te": regimes_te,
        "T": int(Xtr.shape[1]), "n_classes": int(data["n_classes"]),
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "info": info,
    }


def fit_extractor(ctx):
    """Canonical MiniRocket, fit on TRAIN rows of the z-normed trainva."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(ctx["Xtrva_z"][:ctx["n_train"]][:, None, :].astype(np.float32))
    return extractor


def compute_G_H(extractor, X_z, regimes, chunk=16):
    """Chunked G (N, 4998) and H (N, 4998) via the audited implementations.

    G uses the first 4998 global kernels (R2's exact global selection);
    H the audited heterogeneity on the het block.
    """
    N = X_z.shape[0]
    G = np.empty((N, N_G), dtype=np.float32)
    H = np.empty((N, N_H), dtype=np.float64)
    valid = None
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, valid = compute_raw_activations(extractor, X_z[c0:c1])
        G[c0:c1] = ppv_from_activations(act, valid)[:, :N_G]
        H[c0:c1] = compute_regime_heterogeneity(
            act[:, N_G:], valid[N_G:], regimes[c0:c1])
        del act
    return G, H, valid


def vq_diagnostics(ctx):
    return {"codes_trva": occupancy_stats(ctx["regimes_trva"], K=K_CODES),
            "codes_te": occupancy_stats(ctx["regimes_te"], K=K_CODES)}


def h_block_diagnostics(H):
    """Track-17 style frozen-context H stats."""
    H = np.asarray(H, dtype=np.float64)
    return {
        "mean": float(H.mean()), "std": float(H.std()),
        "min": float(H.min()), "max": float(H.max()),
        "frac_zero_entries": float((H == 0).mean()),
    }
