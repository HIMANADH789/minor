"""TURS-KTM: Kernel activation-Timing Moments for aeon MiniROCKET (1.2.0, univariate).

Background. For a fitted aeon ``MiniRocket`` the transform computes, for each
dilation d_j and each of the 84 fixed kernel index-combinations k, a response

    C_{j,k}(t) = C_alpha(t) + C_gamma[a](t) + C_gamma[b](t) + C_gamma[c](t)

shared across all biases of that (j, k) combo, where C_alpha is the negative
9-tap box sum at offsets {0, +/-d, +/-2d, +/-3d, +/-4d} and C_gamma[g] = 3*X at
the same tap offsets. Each MiniROCKET feature m = (j, k, q) is

    PPV_m = mean_t 1[ C_{j,k}(t) > b_q ]

evaluated over ALL n_timepoints t when the padding parity (_padding0 + k) % 2
is 0, and over the interior slice t in [4d, T - 4d) when the parity is 1.

KTM reuses the *same* activation set {t : C(t) > b} that this PPV thresholding
already defines, and summarizes WHERE in time those activations occur:

    x_i = tau_i / T_e        # normalized positions in [0, 1)
    mu  = (1/k) sum_i x_i
    W   = (1/k) sum_i | x_(i) - i/(k+1) |   # sorted x_(i), i = 1..k

Normalization convention (documented, deliberate): tau_i are positions counted
from the START OF THE EVALUATED WINDOW and T_e is the number of evaluated
positions. For parity-0 features the window is the full signal, so T_e = T and
tau_i are absolute; for parity-1 features the window is the interior slice
[4d, T - 4d), so T_e = T - 8d and tau_i are window-relative. This keeps x_i in
[0, 1) for every feature and keeps W's interpretation (distance to
Uniform[0, 1] over the evaluated window) independent of dilation-induced
window offsets.

with W the closed-form empirical 1D Wasserstein-1 distance from the activation
positions to Uniform[0, 1]. KTM adds no learned parameters and runs no extra
convolution pass that is not already implied by the PPV computation: the same
response array C is reused.

Two implementations live here:

* ``ktm_transform`` -- a numba kernel mirroring ``_static_transform_uni``
  (same feature ordering, same padding), returning (PPV, mu, W) jointly.
* ``ktm_transform_reference`` -- a slower, deliberately naive numpy
  implementation built from an explicit tap list (no aeon slicing tricks);
  used only to cross-validate the fast path in unit tests.

Edge-case conventions (deterministic, documented):
    k = 0 activations : PPV = 0, mu = 0, W = 0
    k = 1 activation  : mu = x_1, W = |x_1 - 1/2|

All three outputs are aligned column-for-column with ``transformer.transform``
output of the same fitted MiniRocket.
"""

from itertools import combinations

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def feature_owners(transformer):
    """Map feature index -> (dilation_value, dilation_idx, kernel_idx, bias_idx).

    Returns arrays ``dil_val, dil_idx, kern_idx`` of length F (bias index is the
    feature index itself).
    """
    _, _, dilations, n_features_per_dilation, biases = transformer.parameters
    dilations = np.asarray(dilations)
    nfpd = np.asarray(n_features_per_dilation)
    dil_idx = np.concatenate(
        [np.full(84 * int(nf), j, dtype=np.int64)
         for j, nf in enumerate(nfpd)])
    kern_idx = np.concatenate(
        [np.repeat(np.arange(84, dtype=np.int64), int(nf))
         for nf in nfpd])
    return dilations[dil_idx], dil_idx, kern_idx


def _combo_indices():
    """The 84 (a, b, c) tap triples, identical to ``MiniRocket._indices``."""
    return np.array(list(combinations(np.arange(9), 3)), dtype=np.int32)


# ---------------------------------------------------------------------------
# fast numba path (mirrors aeon _static_transform_uni)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _activation_stats(C, b, T):
    """Count, (mu, W) of activations {t : C[t] > b} on positions [0, T).

    Positions are collected in increasing order so the buffer is already
    sorted, matching the tau_(1) < ... < tau_(k) definition. ``T`` is the
    normalization length (number of evaluated positions; x = tau / T).
    """
    n = C.shape[0]
    count = 0
    tau_sum = 0
    # two-pass-free: first pass count, then buffer; numba-friendly
    for t in range(n):
        if C[t] > b:
            count += 1
            tau_sum += t
    if count == 0:
        return 0, 0.0, 0.0
    taus = np.empty(count, dtype=np.int64)
    idx = 0
    for t in range(n):
        if C[t] > b:
            taus[idx] = t
            idx += 1
    mu = 0.0
    for i in range(count):
        mu += taus[i]
    mu = (mu / T) / count
    w = 0.0
    for i in range(count):
        w += abs(taus[i] / T - (i + 1) / (count + 1))
    w /= count
    return count, mu, w


@njit(cache=True, fastmath=True, parallel=True)
def _ktm_static_uni(X, dilations, n_features_per_dilation, biases, indices,
                    want_counts):
    """Per-sample (PPV, mu, W[, counts]) mirroring _static_transform_uni."""
    n_cases, n_timepoints = X.shape
    n_kernels = indices.shape[0]
    n_dilations = dilations.shape[0]
    f_total = n_kernels * int(n_features_per_dilation.sum())
    PPV = np.zeros((n_cases, f_total), dtype=np.float32)
    MU = np.zeros((n_cases, f_total), dtype=np.float32)
    W = np.zeros((n_cases, f_total), dtype=np.float32)
    CT = np.zeros((n_cases, f_total), dtype=np.int64) if want_counts else \
        np.zeros((1, 1), dtype=np.int64)

    for i in prange(n_cases):
        _X = X[i]
        A = -_X
        G = 3 * _X
        f_start = 0
        for j in range(n_dilations):
            _padding0 = j % 2
            dilation = dilations[j]
            padding = (8 * dilation) // 2
            n_features = n_features_per_dilation[j]
            C_alpha = np.zeros(n_timepoints, dtype=np.float32)
            C_alpha[:] = A
            C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
            C_gamma[4] = G
            start = dilation
            end = n_timepoints - padding
            for gamma_index in range(4):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]
                end += dilation
            for gamma_index in range(5, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]
                start += dilation
            for k in range(n_kernels):
                f_end = f_start + n_features
                _padding1 = (_padding0 + k) % 2
                a, b, c = indices[k, 0], indices[k, 1], indices[k, 2]
                C = C_alpha + C_gamma[a] + C_gamma[b] + C_gamma[c]
                if _padding1 == 0:
                    for f in range(n_features):
                        cnt, mu, w = _activation_stats(
                            C, biases[f_start + f], n_timepoints)
                        PPV[i, f_start + f] = cnt / n_timepoints
                        MU[i, f_start + f] = mu
                        W[i, f_start + f] = w
                        if want_counts:
                            CT[i, f_start + f] = cnt
                else:
                    Cw = C[padding:-padding]
                    Tw = n_timepoints - 2 * padding
                    for f in range(n_features):
                        cnt, mu, w = _activation_stats(
                            Cw, biases[f_start + f], Tw)
                        PPV[i, f_start + f] = cnt / Tw
                        MU[i, f_start + f] = mu
                        W[i, f_start + f] = w
                        if want_counts:
                            CT[i, f_start + f] = cnt
                f_start = f_end
    return PPV, MU, W, CT


def ktm_transform(X, transformer, return_counts=False):
    """(PPV, mu, W) per MiniROCKET feature for univariate X, fast numba path.

    Parameters
    ----------
    X : np.ndarray [n_cases, n_timepoints] or [n_cases, 1, n_timepoints],
        float32 per-sample z-normalized (same preprocessing as the benchmark).
    transformer : fitted aeon MiniRocket.
    return_counts : also return the activation count per (sample, feature).

    Returns
    -------
    ppv, mu, W : float32 arrays [n_cases, F] aligned with
        ``transformer.transform`` columns; ``counts`` int64 when requested.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        if X.shape[1] != 1:
            raise ValueError("ktm_transform supports univariate input only")
        X = X[:, 0, :]
    pp = transformer.parameters
    _, _, dilations, n_features_per_dilation, biases = pp
    dilations = np.asarray(dilations, dtype=np.int32)
    nfpd = np.asarray(n_features_per_dilation, dtype=np.int32)
    biases = np.asarray(biases, dtype=np.float32)
    indices = _combo_indices()
    ppv, mu, w, ct = _ktm_static_uni(
        X, dilations, nfpd, biases, indices, bool(return_counts))
    if return_counts:
        return ppv, mu, w, ct
    return ppv, mu, w


# ---------------------------------------------------------------------------
# independent reference path (explicit taps; deliberately naive)
# ---------------------------------------------------------------------------


def _tap_offsets(d):
    """Absolute tap offsets of the 9-tap kernel at dilation d: {0, +/-d, ...4d}."""
    return np.array([g * d for g in range(-4, 5)], dtype=np.int64)


def ktm_transform_reference(X, transformer):
    """Slow explicit-tap reference for (PPV, mu, W); unit-test cross-check only.

    Builds every response from raw numpy indexing with modulo-free tap
    arithmetic: taps falling outside [0, T) are treated as ABSENT (their
    contribution is skipped), which reproduces aeon's slice-add construction
    (C_alpha[0:4d] contains only the non-negative-offset taps etc.). Padding
    parity and evaluation windows match ``_static_transform_uni``.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = X[:, 0, :]
    n, T = X.shape
    _, _, dilations, n_features_per_dilation, biases = transformer.parameters
    dilations = np.asarray(dilations, dtype=np.int32)
    nfpd = np.asarray(n_features_per_dilation, dtype=np.int32)
    biases = np.asarray(biases, dtype=np.float32)
    indices = _combo_indices()
    f_total = 84 * int(nfpd.sum())

    ppv = np.zeros((n, f_total), dtype=np.float64)
    mu = np.zeros((n, f_total), dtype=np.float64)
    W = np.zeros((n, f_total), dtype=np.float64)

    for i in range(n):
        x = X[i]
        f_start = 0
        for j in range(len(dilations)):
            d = int(dilations[j])
            offs = _tap_offsets(d)
            nfeat = int(nfpd[j])
            # gamma value arrays per tap g in 0..8 (gamma row index)
            # gamma[g] evaluated at t is 3*x[t + (g-4)*d] if in range else 0
            # alpha evaluated at t is -sum over all 9 taps x[t + s] (in range)
            # build via explicit tap indexing
            tap_vals = np.zeros((9, T), dtype=np.float64)
            valid = np.zeros((9, T), dtype=bool)
            for g in range(9):
                s = (g - 4) * d
                if s >= 0:
                    valid[g, : T - s] = True
                    tap_vals[g, : T - s] = 3.0 * x[s:]
                elif s < 0:
                    valid[g, -s:] = True
                    tap_vals[g, -s:] = 3.0 * x[: T + s]
            alpha = -(tap_vals.sum(axis=0) / 3.0)  # -sum of x over valid taps
            for k in range(84):
                f_end = f_start + nfeat
                a, b_, c = int(indices[k, 0]), int(indices[k, 1]), int(indices[k, 2])
                C = alpha + tap_vals[a] + tap_vals[b_] + tap_vals[c]
                if (j % 2 + k) % 2 == 1:
                    pad = 4 * d
                    C = C[pad: T - pad]
                Te = C.shape[0]
                for f in range(nfeat):
                    b = biases[f_start + f]
                    mask = C > b
                    cnt = int(mask.sum())
                    ppv[i, f_start + f] = cnt / Te
                    if cnt == 0:
                        continue
                    taus = np.nonzero(mask)[0]  # window-relative positions
                    xs = taus / Te
                    mu[i, f_start + f] = xs.mean()
                    W[i, f_start + f] = np.mean(
                        np.abs(xs - np.arange(1, cnt + 1) / (cnt + 1)))
                f_start = f_end
    return ppv, mu, W


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def activation_count_summary(counts, limit=64):
    """Histogram summary of activation counts over (sample, feature) pairs."""
    c = np.asarray(counts).ravel()
    return {
        "n_pairs": int(c.size),
        "zero": int((c == 0).sum()),
        "one": int((c == 1).sum()),
        "multi": int((c > 1).sum()),
        "zero_frac": float((c == 0).mean()),
        "one_frac": float((c == 1).mean()),
        "multi_frac": float((c > 1).mean()),
        "mean": float(c.mean()),
        "median": float(np.median(c)),
        "max": int(c.max()),
    }


def block_stats(block):
    """min/max/mean/std/median/near-zero-variance for a feature block [N, F]."""
    b = np.asarray(block, dtype=np.float64)
    fmin = b.min(axis=0)
    fmax = b.max(axis=0)
    fmean = b.mean(axis=0)
    fstd = b.std(axis=0)
    fmed = np.median(b, axis=0)
    nzv = int((fstd < 1e-8).sum())
    return {
        "n_features": int(b.shape[1]),
        "min": float(fmin.min()),
        "max": float(fmax.max()),
        "mean": float(fmean.mean()),
        "std": float(fstd.mean()),
        "median": float(np.median(fmed)),
        "near_zero_variance_count": nzv,
        "feature_min_mean": float(fmin.mean()),
        "feature_max_mean": float(fmax.mean()),
    }


def standardize_blocks(train_blocks, val_blocks, test_blocks):
    """Block-wise standardization fitted on the given fit-rows only.

    Parameters/Returns
    ------------------
    train_blocks : list of [N_tr, F_b] arrays
    val_blocks, test_blocks : lists of matching arrays (may be None entries)
    Returns lists of standardized arrays; ``None`` entries pass through as
    ``None``. A block column with zero training std is left unscaled (scale 1)
    to avoid division by zero.
    """
    out_tr, out_va, out_te = [], [], []
    for tr, va, te in zip(train_blocks, val_blocks, test_blocks):
        m = tr.mean(axis=0, keepdims=True)
        s = tr.std(axis=0, keepdims=True)
        s = np.where(s < 1e-8, 1.0, s)
        out_tr.append((tr - m) / s)
        out_va.append(None if va is None else (va - m) / s)
        out_te.append(None if te is None else (te - m) / s)
    return out_tr, out_va, out_te


def concatenate_blocks(blocks):
    """np.hstack a list of blocks into one matrix (float64)."""
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks],
                          axis=1)
