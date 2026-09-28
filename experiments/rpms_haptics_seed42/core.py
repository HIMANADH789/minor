"""RPMS core: equal-budget three-branch representation on Haptics (seed 42).

Branches (3332 + 3332 + 3332 = 9996 features, allocation FIXED by rule):
  G      -- canonical global PPV, first 3332 kernels in canonical order
  H      -- audited regime PPV heterogeneity, first 3332 het features
  HydraH -- regime-conditional Hydra win dispersion, first 3332 Hydra units

Deterministic allocation rule (pre-declared, no validation or test input):
  Branch 1: kernels 0..3331 of the canonical 9996-feature MiniRocket layout
            (features [f0, f0+1, ...] = (dilation, kernel, bias) flat order;
            kernel subset = first 3332 feature indices, of which kernels
            0..~792 are used; documented mapping below).
  Branch 2: het features 0..3331 (same canonical order, het block).
  Branch 3: Hydra units 0..3331 (dilation-major, then diff, group, kernel).

The temporal context is the FROZEN validated R2 checkpoint
(results/rcmkn_haptics_seed42/context_model_seed42.pt) -- identical encoder,
VQ, and regime extraction as the R2 experiment (extraction re-run here; the
run's regime audit gate verifies byte-identity with the saved audit arrays,
making the frozen-context requirement checkable, not assumed).
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    compute_regime_heterogeneity,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations, ppv_from_activations,
    independent_heterogeneity_recompute,
)
import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as transfer

from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
from experiments.rcmkn_haptics_seed42.runner import (
    extract_context_regimes, macro_f1, set_seed,
)
from experiments.rcmkn_haptics_seed42.vq import occupancy_stats
from experiments.rpms_haptics_seed42 import hydra_stats, information, regime_stats

SEED = 42
K_CODES = 8
T_EXPECTED = 1092
N_BRANCH = 3332
N_TOTAL = 9996
MIN_OCCUPANCY = 0.01
ALPHAS = np.logspace(-4, 4, 20)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "rpms_haptics_seed42")
R2_CKPT = os.path.join(RESULTS_ROOT, "rcmkn_haptics_seed42",
                       "context_model_seed42.pt")

EXPECTED = {"train": 132, "val": 23, "test": 308, "T": 1092, "n_classes": 5}
REFS = {"M0": 0.4974, "R2": 0.5500}


def log(msg):
    print(msg, flush=True)


def sha16(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ #
# data + frozen context                                               #
# ------------------------------------------------------------------ #
def load_all(device):
    data = transfer.load_any_dataset("Haptics")
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    znorm = transfer.znorm
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    Xtrva_z = np.vstack([Xtr_z, Xva_z]).astype(np.float32)
    ytrva = np.concatenate([ytr, yva])

    model = RCMKNContextModel(n_classes=EXPECTED["n_classes"])
    ck = torch.load(R2_CKPT, map_location="cpu", weights_only=False)
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
        "n_train": len(Xtr), "regimes_trva": regimes_trva,
        "regimes_te": regimes_te,
        "T": int(Xtr.shape[1]), "n_classes": int(data["n_classes"]),
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
    }


# ------------------------------------------------------------------ #
# Branch 1 + 2 shared raw-activation pass (chunked)                   #
# ------------------------------------------------------------------ #
def compute_G_H(extractor, X_z, regimes, chunk=16):
    """Chunked G (N, 9996) + H on the het block (N, 4998)."""
    N, T = X_z.shape
    G = np.empty((N, 9996), dtype=np.float32)
    H = np.empty((N, 4998), dtype=np.float64)
    valid = None
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, valid = compute_raw_activations(extractor, X_z[c0:c1])
        G[c0:c1] = ppv_from_activations(act, valid)
        H[c0:c1] = compute_regime_heterogeneity(
            act[:, 4998:], valid[4998:], regimes[c0:c1])
        del act
    return G, H, valid


# ------------------------------------------------------------------ #
# Branch 3                                                            #
# ------------------------------------------------------------------ #
def compute_HydraH(X_z, regimes, bank, batch=16):
    c_kr, n_ir, n_v = hydra_stats.hydra_regime_counts(bank, X_z, regimes,
                                                      batch=batch)
    H, contrib = hydra_stats.hydra_h_from_counts(c_kr, n_ir, n_v,
                                                 X_z.shape[1])
    return H, contrib, (c_kr, n_ir, n_v)


# ------------------------------------------------------------------ #
# Ridge protocol                                                      #
# ------------------------------------------------------------------ #
def fit_ridge(F_tr, ytrva, F_va, yva, F_te=None, yte=None):
    from sklearn.linear_model import RidgeClassifierCV
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(F_tr, ytrva)
    pred_va = ridge.predict(F_va)
    out = {
        "val_macro_f1": round(macro_f1(yva, pred_va), 4),
        "selected_alpha": float(ridge.alpha_),
        "dim": int(F_tr.shape[1]),
    }
    pred_te = None
    if F_te is not None:
        pred_te = ridge.predict(F_te)
        out["test_macro_f1"] = round(macro_f1(yte, pred_te), 4)
    return out, pred_va, pred_te
