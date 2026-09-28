"""Stage 1/4/5 fixed-kernel feature computation for RCMKN.

    * canonical MiniROCKET + bit-exact raw activation extractor (reused)
    * audited valid-region heterogeneity H_m (reused, formula unchanged)
    * Hydra competitive pooling: count_max / count_min from aeon's
      _HydraInternal (the exact HydraTransformer core) applied to the
      SAME z-normalized input

Memory: everything chunked over samples; no act.copy() of full tensors.
"""
import numpy as np
import torch

from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations, ppv_from_activations,
)
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    compute_regime_heterogeneity,
)


def compute_heterogeneity_chunked(extractor, X_z, regimes, valid_het,
                                  chunk=128, act_h=None, N_GLOBAL=4998):
    """Audited H_m over (N, T) regimes in bounded sample chunks.

    regimes: (N, T) int array (actual, random, or shuffled - same function).
    Returns (N, 4998) float64.
    """
    act_h = act_h or (lambda a: a[:, N_GLOBAL:, :])
    N = len(X_z)
    H = np.empty((N, N_GLOBAL), dtype=np.float64)
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        act, _ = compute_raw_activations(extractor, X_z[c0:c1])
        H[c0:c1] = compute_regime_heterogeneity(
            act_h(act), valid_het, regimes[c0:c1])
        del act
    return H


def compute_hydra_features(X_z, T, k=8, g=16, batch=64, seed=42):
    """Hydra competitive/count features via aeon's _HydraInternal.

    X_z: (N, T) float32 z-normalized. Returns (N, F_hydra) float64 plus the
    module (for dimension bookkeeping).

    Determinism: _HydraInternal samples its kernel weights from the torch
    RNG at construction, so the seed is RE-APPLIED at every call -> every
    call builds identical weights regardless of prior RNG consumption.
    Computation runs on CPU unconditionally (cuDNN conv forwards are not
    bitwise deterministic; CPU cost is negligible at benchmark scale).
    """
    from aeon.transformations.collection.convolution_based._hydra import (
        _HydraInternal,
    )
    if seed is not None:
        torch.manual_seed(seed)
    hydra = _HydraInternal(T, 1, k=k, g=g)
    out = []
    with torch.no_grad():
        for c0 in range(0, len(X_z), batch):
            xb = torch.from_numpy(
                np.ascontiguousarray(X_z[c0:c0 + batch], dtype=np.float32))[:, None, :]
            out.append(hydra(xb).detach().numpy().astype(np.float64))
    return np.concatenate(out, axis=0), hydra


def scale_hydra_features(H_fit, *H_apply):
    """Canonical Hydra _SparseScaler (aeon), fit on train+val, applied to the
    Hydra block only. The global/heterogeneity blocks keep their audited
    PPV scale untouched. Raw Hydra counts (0..T magnitude) otherwise drown
    the [0,1]-scale PPV features and pin Ridge alpha at the grid maximum
    (observed in smoke: alpha=1e4, R1 test MF1 collapsed).

    Mirrors _SparseScaler exactly: X.clamp(0).sqrt(); epsilon =
    (X==0).mean(0)^4 + 1e-8; transform = ((sqrtX - mu) * mask) / sigma.
    Implemented in numpy (the aeon class requires torch input).
    """
    Xf = np.sqrt(np.clip(H_fit, 0, None))
    epsilon = (H_fit == 0).mean(axis=0) ** 4 + 1e-8
    mu = Xf.mean(axis=0)
    sigma = Xf.std(axis=0) + epsilon
    out = []
    for H in (H_fit,) + H_apply:
        X = np.sqrt(np.clip(H, 0, None))
        out.append(((X - mu) * (H != 0)) / sigma)
    return out if len(H_apply) else out[0]


def hydra_dimension(T, k=8, g=16):
    from aeon.transformations.collection.convolution_based._hydra import (
        _HydraInternal,
    )
    m = _HydraInternal(T, 1, k=k, g=g)
    return int(m.num_dilations * m.divisor * m.h * m.k * 2), {
        "num_dilations": int(m.num_dilations), "divisor": int(m.divisor),
        "h_groups": int(m.h), "k_kernels_per_group": int(m.k),
        "kernels_total": int(m.num_dilations * m.divisor * m.h * m.k),
        "stats_per_kernel": 2,}
