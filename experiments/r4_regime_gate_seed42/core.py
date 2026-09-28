"""R4 core: frozen R2 pipeline + the EXACT per-regime H decomposition.

H_{m,k} = q_k (PPV_{m,k} - PPV_m)^2 with q_k the audited RENORMALIZED,
min-occupancy-masked regime weight -- so that sum_k H_{m,k} == H_m^(R2)
exactly (Audit 8).  The decomposition mirrors heterogeneity_features
(padding groups, valid region, min_count, fallback, renormalization);
the audited function itself still computes the official H used by the
GH control, and the two are cross-checked per run.
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
    _padding_groups,
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
# spec (R4 differentiable-Ridge update) section 25 artifact namespace
OUT_DIR = os.path.join(RESULTS_ROOT,
                       "r4_differentiable_ridge_regime_gate_seed42")

DATASETS = {
    "Haptics": {
        "ckpt": os.path.join(RESULTS_ROOT, "rcmkn_haptics_seed42",
                             "context_model_seed42.pt"),
        "ckpt_hash": "2dbb0cf4f3db3df8",
        "expected": {"train": 132, "val": 23, "test": 308, "T": 1092,
                     "n_classes": 5},
        # frozen seed-42 references (user-supplied; section 20)
        "M0": 0.4974, "R2": 0.5500,
        # R2-selected Ridge alpha (audit: results/rcmkn_haptics_seed42/
        # report.json /results/R2/selected_alpha)
        "alpha_r2": 4.281332398719396,
        "alpha_source": "results/rcmkn_haptics_seed42/report.json"
                        ":/results/R2/selected_alpha",
    },
    "ECG5000_BAL": {
        "ckpt": os.path.join(RESULTS_ROOT,
                             "rcmkn_ssl_context_important2_seed42",
                             "ECG5000_BAL", "context_model_seed42.pt"),
        "ckpt_hash": "f2eda1b2ff0bbe99",
        "expected": {"train": 5226, "val": 923, "test": 1000, "T": 140,
                     "n_classes": 5},
        # user-frozen reference 0.6409; NOTE: the NaN-fix rerun of the
        # official ECG5000_BAL trainer (rcmkn_ssl_context_important2
        # rerun, R0 gate 0.6748 exact) reported R2 = 0.6089 -- both are
        # recorded in the report; tables use the user-frozen value.
        "M0": 0.6553, "R2": 0.6409,
        # R2-selected Ridge alpha (audit: results/
        # rcmkn_ssl_context_important2_seed42/ECG5000_BAL/result.json
        # /results/R2/selected_alpha)
        "alpha_r2": 1.623776739188721,
        "alpha_source":
            "results/rcmkn_ssl_context_important2_seed42/"
            "ECG5000_BAL/result.json:/results/R2/selected_alpha",
    },
}


def log(msg):
    print(msg, flush=True)


def sha16(a):
    if isinstance(a, (bytes, bytearray)):
        return hashlib.sha256(a).hexdigest()[:16]
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def load_frozen_context(ds_name, device):
    """Canonical split + frozen audited R2 context (identical to R3)."""
    info = DATASETS[ds_name]
    data = transfer.load_any_dataset(ds_name)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    Xtr_z, Xva_z, Xte_z = transfer.znorm(Xtr), transfer.znorm(Xva), \
        transfer.znorm(Xte)
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
        "n_train": len(Xtr),
        "regimes_trva": regimes_trva, "regimes_te": regimes_te,
        "T": int(Xtr.shape[1]), "n_classes": int(data["n_classes"]),
        "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "info": info,
    }


def fit_extractor(ctx):
    from aeon.transformations.collection.convolution_based import MiniRocket
    extractor = MiniRocket(random_state=SEED, n_jobs=-1)
    extractor.fit(ctx["Xtrva_z"][:ctx["n_train"]][:, None, :].astype(np.float32))
    return extractor


def compute_H_contributions(act, valid, regimes, K=K_CODES,
                            min_occupancy=MIN_OCCUPANCY):
    """Per-regime contributions H_{m,k} = q_k (PPV_{m,k} - PPV_m)^2.

    Mirrors heterogeneity_features EXACTLY (padding groups, valid region,
    min_count selection, fallback, renormalized q) but returns the
    UNAGGREGATED per-regime terms.

    act: (N, F, T) bool; valid: (F, T) bool; regimes: (N, T) int
    Returns contrib: (N, F, K) float32  (H_m = contrib.sum(axis=2)).
    """
    n_samples, n_features, T = act.shape
    min_count = int(np.ceil(min_occupancy * T))
    groups = _padding_groups(valid)

    contrib = np.zeros((n_samples, n_features, K), dtype=np.float32)
    for i in range(n_samples):
        for p, feats in groups:
            hi = T - p if p > 0 else T
            Av = act[i, feats, p:hi].astype(np.float32)      # (Fp, n_valid)
            rv = regimes[i, p:hi]
            n_valid = Av.shape[1]
            onehot = np.zeros((K, n_valid), dtype=np.float32)
            onehot[rv, np.arange(n_valid)] = 1.0
            counts = onehot.sum(axis=1)                      # (K,)
            sel = counts >= min_count
            if not sel.any():
                sel[np.argmax(counts)] = True
            w = np.where(sel, counts / n_valid, 0.0)
            w = w / w.sum()
            ppv_k = (onehot @ Av.T) / np.maximum(counts, 1)[:, None]
            ppv_g = Av.mean(axis=1)
            dev = ppv_k - ppv_g[None, :]
            contrib[i, feats, :] = (w[:, None] * (dev ** 2)
                                    * sel[:, None]).T.astype(np.float32)
    return contrib


def compute_G_H_Hk(extractor, X_z, regimes, chunk=16):
    """Chunked G (N, 4998), audited H (N, 4998), contributions (N, 4998, 8)."""
    N, T = X_z.shape
    G = np.empty((N, N_G), dtype=np.float32)
    H = np.empty((N, N_H), dtype=np.float64)
    Hk = np.empty((N, N_H, K_CODES), dtype=np.float32)
    valid = None
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, valid = compute_raw_activations(extractor, X_z[c0:c1])
        G[c0:c1] = ppv_from_activations(act, valid)[:, :N_G]
        H[c0:c1] = compute_regime_heterogeneity(
            act[:, N_G:], valid[N_G:], regimes[c0:c1])
        Hk[c0:c1] = compute_H_contributions(
            act[:, N_G:], valid[N_G:], regimes[c0:c1])
        del act
    return G, H, Hk, valid


def regime_statistics(regimes, Hk, T, K=K_CODES, n_train=None):
    """Required regime diagnostics (spec section 17), train+val scope."""
    occ = np.bincount(regimes.ravel(), minlength=K)
    total = occ.sum()
    per_sample = np.stack([(regimes == k).any(axis=1) for k in range(K)])
    stats = []
    Hk_sum = Hk.sum(axis=0)                        # (F, K)
    grand = Hk_sum.sum()
    mean_Hk = Hk.mean(axis=(0, 1))                 # (K,) mean over N, F
    for k in range(K):
        stats.append({
            "regime": k,
            "occupancy": int(occ[k]),
            "occupancy_fraction": float(occ[k] / max(total, 1)),
            "samples_containing": int(per_sample[k].sum()),
            "mean_H_contribution": float(mean_Hk[k]),
            "fraction_of_H_contribution": float(
                Hk_sum[:, k].sum() / max(grand, 1e-30)),
        })
    return stats


def vq_diagnostics(ctx):
    return {"codes_trva": occupancy_stats(ctx["regimes_trva"], K=K_CODES),
            "codes_te": occupancy_stats(ctx["regimes_te"], K=K_CODES)}


def h_block_diagnostics(H):
    H = np.asarray(H, dtype=np.float64)
    return {"mean": float(H.mean()), "std": float(H.std()),
            "min": float(H.min()), "max": float(H.max()),
            "frac_zero_entries": float((H == 0).mean())}
