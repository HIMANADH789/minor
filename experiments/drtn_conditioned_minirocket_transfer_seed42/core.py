"""
Core feature construction for the DRTN-conditioned MiniROCKET transfer screen.

Generalizes the validated Haptics mechanism to arbitrary-length datasets and
makes the canonical-PPV convention exact:

    * For features whose kernel response is only computed on the interior
      slice [padding : T - padding] (aeon's _padding1 == 1 case), the
      canonical PPV is the mean over that VALID region only. We therefore
      track a per-feature boolean valid mask and compute every activation
      rate over valid positions exclusively. With this convention the
      global PPV block is bit-identical to the aeon MiniRocket transform.

Heterogeneity (unchanged formula, computed over valid positions):

    H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2

    q_k       = n_{k,valid} / n_valid
    PPV_{m,k} = mean of a_m(t) over valid t in regime k
    PPV_m     = mean of a_m(t) over all valid t
    min-occupancy rule: regimes with n_{k,valid} < ceil(0.01 * T) are
    excluded and the weights q_k are renormalized over surviving regimes.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_haptics_seed42.core import (  # noqa: E402,F401
    compute_regime_occupancy_stats,
    create_random_regime_control,
    create_shuffled_regime_control,
    extract_drtn_regimes,
)


# ---------------------------------------------------------------------------
# Raw MiniROCKET responses (numba-jitted, univariate)
# ---------------------------------------------------------------------------
def _compile_raw_response_kernel():
    from numba import njit, prange
    from itertools import combinations

    indices = np.array([_ for _ in combinations(np.arange(9), 3)],
                       dtype=np.int32)

    @njit(fastmath=True, parallel=True, cache=True)
    def _raw_activation_kernel(X, dilations, n_features_per_dilation, biases,
                               indices):
        """Return (act, valid):
        act[i, f, t]   = 1[r_f(t; X_i) > b_f] on valid t, else False
        valid[f, t]    = True where aeon computes the response for feature f
        Feature ordering matches the aeon univariate transform exactly."""
        n_cases, T = X.shape
        n_kernels = len(indices)
        n_dilations = len(dilations)
        n_features = n_kernels * np.sum(n_features_per_dilation)
        act = np.zeros((n_cases, n_features, T), dtype=np.bool_)
        valid = np.zeros((n_features, T), dtype=np.bool_)

        for i in prange(n_cases):
            _X = X[i]
            A = -_X
            G = 3.0 * _X
            f_start = 0
            for j in range(n_dilations):
                _padding0 = j % 2
                dilation = dilations[j]
                padding = (8 * dilation) // 2
                n_feat_this_dil = n_features_per_dilation[j]

                C_alpha = np.zeros(T, dtype=np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, T), dtype=np.float32)
                C_gamma[4] = G

                start = dilation
                end = T - padding
                for gamma_index in range(4):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation
                for gamma_index in range(5, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                for k in range(n_kernels):
                    f_end = f_start + n_feat_this_dil
                    _padding1 = (_padding0 + k) % 2
                    a, b, c = indices[k]
                    C = C_alpha + C_gamma[a] + C_gamma[b] + C_gamma[c]
                    if _padding1 == 0:
                        if i == 0:
                            for t in range(T):
                                valid[f_start:f_start + n_feat_this_dil, t] = True
                        for f in range(n_feat_this_dil):
                            b_f = biases[f_start + f]
                            for t in range(T):
                                if C[t] > b_f:
                                    act[i, f_start + f, t] = True
                    else:
                        if i == 0:
                            for t in range(padding, T - padding):
                                valid[f_start:f_start + n_feat_this_dil, t] = True
                        for f in range(n_feat_this_dil):
                            b_f = biases[f_start + f]
                            for t in range(padding, T - padding):
                                if C[t] > b_f:
                                    act[i, f_start + f, t] = True
                    f_start = f_end
        return act, valid

    return _raw_activation_kernel, indices


_KERNEL_CACHE = {}


def compute_raw_activations(extractor, X):
    """Raw activation indicators + valid masks for every MiniRocket feature.

    Returns
    -------
    act : np.bool_ (n_samples, n_features, T)
    valid : np.bool_ (n_features, T)
    """
    if "kernel" not in _KERNEL_CACHE:
        _KERNEL_CACHE["kernel"], _KERNEL_CACHE["indices"] = \
            _compile_raw_response_kernel()
    kernel, indices = _KERNEL_CACHE["kernel"], _KERNEL_CACHE["indices"]

    X = np.ascontiguousarray(X, dtype=np.float32)
    _, _, dilations, n_features_per_dilation, biases = extractor.parameters
    dilations = np.ascontiguousarray(dilations, dtype=np.int32)
    n_features_per_dilation = np.ascontiguousarray(
        n_features_per_dilation, dtype=np.int32)
    biases = np.ascontiguousarray(biases, dtype=np.float32)
    act, valid = kernel(X, dilations, n_features_per_dilation, biases, indices)
    return act, valid


def ppv_from_activations(act, valid):
    """Canonical PPV: fraction of VALID positions with activation, per feature.

    Matches aeon's transform exactly (mean over the valid slice).
    act: (N, F, T) bool; valid: (F, T) bool -> (N, F) float32
    """
    n_valid = valid.sum(axis=1).astype(np.float64)          # (F,)
    return (act.sum(axis=2) / n_valid[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Regime-heterogeneity features
# ---------------------------------------------------------------------------
MIN_OCCUPANCY = 0.01


def _padding_groups(valid):
    """Group feature indices by their valid-region slice.

    Returns list of (p, idx) where the valid region for features idx is
    [p, T - p). The full-validity group has p == 0.
    """
    T = valid.shape[1]
    groups = {}
    for f in range(valid.shape[0]):
        row = valid[f]
        if row.all():
            p = 0
        else:
            p = int(np.argmax(row))          # first True
            assert row[p:T - p].all() and not row[:p].any() \
                and not row[T - p:].any(), "non-contiguous valid region"
        groups.setdefault(p, []).append(f)
    return [(p, np.asarray(idx, dtype=np.int64)) for p, idx in sorted(groups.items())]


def heterogeneity_features(act, valid, regimes, K=8,
                           min_occupancy=MIN_OCCUPANCY):
    """Regime-heterogeneity features per the validated formula, computed
    over each feature's valid region.

    act: (N, F, T) bool; valid: (F, T) bool; regimes: (N, T) int
    Returns H: (N, F) float64.
    """
    n_samples, n_features, T = act.shape
    min_count = int(np.ceil(min_occupancy * T))
    groups = _padding_groups(valid)

    H = np.zeros((n_samples, n_features), dtype=np.float64)
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
            ppv_k = (onehot @ Av.T) / np.maximum(counts, 1)[:, None]  # (K, Fp)
            ppv_g = Av.mean(axis=1)                                   # (Fp,)
            dev = ppv_k - ppv_g[None, :]
            H[i, feats] = (w[:, None] * (dev ** 2) * sel[:, None]).sum(axis=0)
    return H


def independent_heterogeneity_recompute(act_single, valid_single,
                                        regimes_single, K=8,
                                        min_occupancy=MIN_OCCUPANCY):
    """Reference O(T) recomputation for one sample over all its features.

    act_single: (T,) bool for ONE feature of ONE sample
    valid_single: (T,) bool valid mask for that feature
    regimes_single: (T,) regime ids
    Returns scalar H_m.
    """
    T = len(act_single)
    min_count = int(np.ceil(min_occupancy * T))
    vm = np.asarray(valid_single, dtype=bool)
    a = np.asarray(act_single, dtype=bool)[vm]
    r = np.asarray(regimes_single)[vm]
    n_valid = len(a)
    if n_valid == 0:
        return 0.0
    ppv_global = a.mean()
    total_w, H = 0.0, 0.0
    terms = []
    for k in np.unique(r):
        mask = r == k
        n_k = int(mask.sum())
        if n_k < min_count:
            continue
        q_k = n_k / n_valid
        ppv_k = a[mask].mean()
        terms.append(q_k * (ppv_k - ppv_global) ** 2)
        total_w += q_k
    if total_w == 0:
        return 0.0
    # renormalized weights (q_k / total valid weight) to match the vectorized
    # implementation's exclusion of sub-occupancy regimes
    return float(sum(terms) / total_w)


def dataset_stats(H):
    return {
        "mean": float(H.mean()),
        "std": float(H.std()),
        "nonzero_fraction": float((H != 0).mean()),
        "max": float(H.max()),
        "exactly_zero_fraction": float((H == 0).mean()),
    }
