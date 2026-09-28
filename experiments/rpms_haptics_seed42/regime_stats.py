"""Branch 2: regime PPV heterogeneity (validated R2 statistic) plus
per-regime contribution decomposition for information tracking.

The H_m computation itself is delegated to the audited implementation
(heterogeneity_features via compute_regime_heterogeneity) -- never
reimplemented here.  This module adds only the per-regime contribution
decomposition

    contribution_{m,k} = q_k * (PPV_{m,k} - PPV_m)^2

using IDENTICAL grouping/min-occupancy semantics as the audited code
(contiguous padding groups, ceil(0.01*T) regime threshold, renormalized q).
"""

import numpy as np


def _padding_groups(valid):
    """Contiguous feature groups with identical [pad, T-pad) region
    (mirrors the audited heterogeneity_features grouping)."""
    groups = {}
    T = valid.shape[1]
    for f in range(valid.shape[0]):
        row = valid[f]
        if row.all():
            p = 0
        elif row[0]:
            # [something, T-p) pattern: find pad from the tail
            p = int(np.argmax(~row[::-1]))
        else:
            p = int(np.argmax(row))
        groups.setdefault(p, []).append(f)
    return [(p, np.array(f, dtype=int)) for p, f in sorted(groups.items())]


def regime_contributions(act, valid, regimes, K=8, min_occupancy=0.01):
    """Per-feature, per-regime contributions q_k*(PPV_{m,k}-PPV_m)^2.

    Same inputs/semantics as the audited heterogeneity_features.
    Returns (N, F, K) float64 (zero where the regime was excluded).
    """
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        MIN_OCCUPANCY,
    )
    if min_occupancy is None:
        min_occupancy = MIN_OCCUPANCY
    n_samples, n_features, T = act.shape
    min_count = int(np.ceil(min_occupancy * T))
    groups = _padding_groups(valid)

    contrib = np.zeros((n_samples, n_features, K), dtype=np.float64)
    for i in range(n_samples):
        for p, feats in groups:
            hi = T - p if p > 0 else T
            Av = act[i, feats, p:hi].astype(np.float32)
            rv = regimes[i, p:hi]
            n_valid = Av.shape[1]
            onehot = np.zeros((K, n_valid), dtype=np.float32)
            onehot[rv, np.arange(n_valid)] = 1.0
            counts = onehot.sum(axis=1)
            sel = counts >= min_count
            if not sel.any():
                sel[np.argmax(counts)] = True
            w = np.where(sel, counts / n_valid, 0.0)
            w = w / w.sum()
            ppv_k = (onehot @ Av.T) / np.maximum(counts, 1)[:, None]
            ppv_g = Av.mean(axis=1)
            dev = ppv_k - ppv_g[None, :]
            contrib[i, feats] = (w[:, None] * (dev ** 2) * sel[:, None]).T
    return contrib
