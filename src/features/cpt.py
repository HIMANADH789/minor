"""TURS-CPT: Cyclic Phase Timing for aeon MiniROCKET.

Computes intrinsic-cycle-anchored phase features from MiniROCKET kernel
activations. The hypothesis: absolute timing (KTM μ/W) is boundary-dependent,
but cyclic phase relative to the signal's own dominant period is robust.

Features per kernel (after top-K selection):
  C_m = |R_m|        phase concentration ∈ [0, 1]
  Θ_m = angle(R_m)   circular mean phase ∈ [0, 1)

Period estimation is data-driven (autocorrelation primary, FFT fallback),
using NO class labels.
"""
import numpy as np
from numba import njit, prange
from itertools import combinations

# ---------------------------------------------------------------------------
# Period estimation
# ---------------------------------------------------------------------------

def estimate_period_autocorrelation(x, max_lag=None):
    """Estimate dominant period via autocorrelation peak detection.

    Parameters
    ----------
    x : 1-D float array, z-normalized signal.
    max_lag : int or None. Maximum lag to search. Default: len(x)//2.

    Returns
    -------
    period : float or None
    method : str ('ACF' or 'FFT' or 'FALLBACK')
    quality : float (0-1, estimated reliability)
    """
    T = len(x)
    if max_lag is None:
        max_lag = T // 2

    if max_lag < 3:
        return None, 'FALLBACK', 0.0

    # Compute autocorrelation via FFT (fast)
    x_centered = x - x.mean()
    fft_x = np.fft.rfft(x_centered)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x), n=T)
    acf = acf_full[1:max_lag + 1] / acf_full[0]  # normalize, skip lag 0

    if len(acf) < 3:
        return None, 'FALLBACK', 0.0

    # Find peaks: local maxima with prominence
    peaks = []
    for i in range(1, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
            # Prominence: minimum drop to either side within window
            w = min(i, len(acf) - 1 - i, max_lag // 4)
            if w < 1:
                w = 1
            left_min = acf[max(0, i - w):i].min() if i > 0 else acf[i]
            right_min = acf[i + 1:min(len(acf), i + w + 1)].min()
            prominence = acf[i] - max(left_min, right_min)
            if prominence > 0.05:  # minimum prominence threshold
                peaks.append((i + 1, acf[i], prominence))  # lag is 1-indexed

    if not peaks:
        return None, 'FALLBACK', 0.0

    # Select first prominent peak (shortest period)
    period, acf_val, prom = peaks[0]
    quality = min(1.0, acf_val * prom * 5)  # rough confidence estimate
    return float(period), 'ACF', quality


def estimate_period_fft(x, min_period=2, max_period=None):
    """Fallback period estimation via dominant FFT frequency.

    Returns
    -------
    period : float or None
    method : str ('FFT' or 'FALLBACK')
    quality : float
    """
    T = len(x)
    if max_period is None:
        max_period = T // 2

    x_centered = x - x.mean()
    fft_mag = np.abs(np.fft.rfft(x_centered))

    # Ignore DC (index 0) and very low frequencies
    if len(fft_mag) < 4:
        return None, 'FALLBACK', 0.0

    # Find dominant frequency in valid range
    valid_start = max(1, int(T / max_period))
    valid_end = min(len(fft_mag), int(T / min_period) + 1)

    if valid_end <= valid_start:
        return None, 'FALLBACK', 0.0

    mag_slice = fft_mag[valid_start:valid_end]
    dom_idx = valid_start + np.argmax(mag_slice)
    dom_mag = fft_mag[dom_idx]

    if dom_idx == 0:
        return None, 'FALLBACK', 0.0

    period = T / dom_idx
    if period < min_period or period > max_period:
        return None, 'FALLBACK', 0.0

    # Quality: relative magnitude of dominant peak
    mean_mag = fft_mag[1:].mean()
    quality = min(1.0, dom_mag / (mean_mag + 1e-8) / 3)
    return float(period), 'FFT', quality


def estimate_period(x, min_period=2, max_lag_ratio=0.5):
    """Combined period estimator: ACF primary, FFT fallback.

    Parameters
    ----------
    x : 1-D float array (z-normalized signal).
    min_period : int, minimum valid period in samples.
    max_lag_ratio : float, max_lag = int(T * max_lag_ratio).

    Returns
    -------
    period : float (always valid, never None)
    method : str ('ACF', 'FFT', 'FALLBACK')
    quality : float
    """
    T = len(x)
    max_lag = int(T * max_lag_ratio)
    max_period = T // 2

    # Try ACF first
    p, m, q = estimate_period_autocorrelation(x, max_lag)
    if p is not None and min_period <= p <= max_period:
        return p, m, q

    # Try FFT fallback
    p, m, q = estimate_period_fft(x, min_period, max_period)
    if p is not None and min_period <= p <= max_period:
        return p, m, q

    # Final fallback: use a reasonable default
    # For ECG-like signals, ~50-70 samples; for CWRU-like, ~200-500
    # Use the median of autocorrelation zero-crossings as heuristic
    x_centered = x - x.mean()
    signs = np.sign(x_centered)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) >= 2:
        # Estimate period as 2 * mean half-cycle length
        half_cycles = np.diff(crossings)
        p = float(2 * np.median(half_cycles))
        if min_period <= p <= max_period:
            return p, 'FALLBACK', 0.3

    # Ultimate fallback: T/4 (arbitrary but bounded)
    return float(T // 4), 'FALLBACK', 0.1


# ---------------------------------------------------------------------------
# Phase computation (numba-accelerated)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _cpt_from_taus(taus, T, P):
    """Compute (concentration, phase_mean) from sorted activation positions.

    Parameters
    ----------
    taus : int64 array, sorted activation positions (0-based).
    T : int, signal length (for normalization context).
    P : float, estimated period.

    Returns
    -------
    concentration : float64, |R| ∈ [0, 1]
    phase_mean : float64, normalized angle ∈ [0, 1)
    """
    k = taus.shape[0]
    if k == 0:
        return 0.0, 0.0
    if k == 1:
        return 1.0, (taus[0] % P) / P

    sin_sum = 0.0
    cos_sum = 0.0
    for i in range(k):
        phi = (taus[i] % P) / P
        angle = 2.0 * np.pi * phi
        sin_sum += np.sin(angle)
        cos_sum += np.cos(angle)

    sin_mean = sin_sum / k
    cos_mean = cos_sum / k
    R = np.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    theta = np.arctan2(sin_mean, cos_mean)
    theta_norm = (theta % (2.0 * np.pi)) / (2.0 * np.pi)

    return R, theta_norm


@njit(cache=True, fastmath=True, parallel=True)
def _cpt_transform_uni(X, dilations, n_features_per_dilation, biases,
                       indices, periods, want_counts):
    """Per-sample (concentration, phase_mean) for all features."""
    n_cases, n_timepoints = X.shape
    n_kernels = indices.shape[0]
    n_dilations = dilations.shape[0]
    f_total = n_kernels * int(n_features_per_dilation.sum())

    CONC = np.zeros((n_cases, f_total), dtype=np.float32)
    PHASE = np.zeros((n_cases, f_total), dtype=np.float32)
    CT = np.zeros((n_cases, f_total), dtype=np.int64) if want_counts else \
        np.zeros((1, 1), dtype=np.int64)

    for i in prange(n_cases):
        _X = X[i]
        A = -_X
        G = 3 * _X
        P = periods[i]
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
            for gi in range(4):
                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gi, -end:] = G[:end]
                end += dilation
            for gi in range(5, 9):
                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gi, :-start] = G[start:]
                start += dilation
            for k in range(n_kernels):
                f_end = f_start + n_features
                _padding1 = (_padding0 + k) % 2
                a, b, c = indices[k, 0], indices[k, 1], indices[k, 2]
                C = C_alpha + C_gamma[a] + C_gamma[b] + C_gamma[c]
                if _padding1 == 0:
                    Cw = C
                    Tw = n_timepoints
                else:
                    Cw = C[padding:-padding]
                    Tw = n_timepoints - 2 * padding
                # Collect activations
                cnt = 0
                for t in range(Tw):
                    if Cw[t] > biases[f_start + 0]:  # placeholder
                        cnt += 1
                # Actually we need per-feature bias loop
                for f in range(n_features):
                    bias = biases[f_start + f]
                    cnt = 0
                    for t in range(Tw):
                        if Cw[t] > bias:
                            cnt += 1
                    if cnt == 0:
                        CONC[i, f_start + f] = 0.0
                        PHASE[i, f_start + f] = 0.0
                    else:
                        taus = np.empty(cnt, dtype=np.int64)
                        idx = 0
                        for t in range(Tw):
                            if Cw[t] > bias:
                                taus[idx] = t
                                idx += 1
                        conc, phase = _cpt_from_taus(taus, Tw, P)
                        CONC[i, f_start + f] = conc
                        PHASE[i, f_start + f] = phase
                    if want_counts:
                        CT[i, f_start + f] = cnt
                f_start = f_end
    return CONC, PHASE, CT


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _combo_indices():
    return np.array(list(combinations(np.arange(9), 3)), dtype=np.int32)


def cpt_transform(X, transformer, periods, top_k_mask=None,
                  return_counts=False):
    """Compute CPT features (concentration, phase) for all features.

    Parameters
    ----------
    X : np.ndarray [n_cases, n_timepoints] or [n_cases, 1, n_timepoints].
    transformer : fitted aeon MiniRocket.
    periods : np.ndarray [n_cases], estimated period per sample.
    top_k_mask : bool array [F] or None. If given, only compute for True features.
        Other features are set to 0.
    return_counts : bool, also return activation counts.

    Returns
    -------
    conc, phase : float32 arrays [n_cases, F]
    counts : int64 array if requested
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        if X.shape[1] != 1:
            raise ValueError("cpt_transform supports univariate input only")
        X = X[:, 0, :]
    periods = np.asarray(periods, dtype=np.float64)

    pp = transformer.parameters
    _, _, dilations, n_features_per_dilation, biases = pp
    dilations = np.asarray(dilations, dtype=np.int32)
    nfpd = np.asarray(n_features_per_dilation, dtype=np.int32)
    biases = np.asarray(biases, dtype=np.float32)
    indices = _combo_indices()

    conc, phase, ct = _cpt_transform_uni(
        X, dilations, nfpd, biases, indices, periods, bool(return_counts))

    if top_k_mask is not None:
        mask = np.asarray(top_k_mask, dtype=bool)
        conc[:, ~mask] = 0.0
        phase[:, ~mask] = 0.0
        if return_counts:
            ct[:, ~mask] = 0

    if return_counts:
        return conc, phase, ct
    return conc, phase


# ---------------------------------------------------------------------------
# Top-K kernel selection (training-data only)
# ---------------------------------------------------------------------------

def rank_kernels_by_discriminative_power(Z, y, method='f_classif'):
    """Rank MiniROCKET features by discriminative power on training data.

    Parameters
    ----------
    Z : np.ndarray [n_train, F], PPV features.
    y : np.ndarray [n_train], class labels.
    method : str, 'f_classif' (ANOVA F-score).

    Returns
    -------
    ranks : np.ndarray [F], feature indices sorted by descending score.
    scores : np.ndarray [F], discriminative scores.
    """
    from sklearn.feature_selection import f_classif
    scores, _ = f_classif(Z, y)
    scores = np.nan_to_num(scores, nan=0.0)
    ranks = np.argsort(-scores)
    return ranks, scores


def select_top_k_transformers(transformer, ranks, K):
    """Map feature-level ranks to kernel-level selection.

    Each MiniROCKET feature corresponds to a (dilation, kernel, bias) triple.
    We select the top-K *kernels* (not features) based on their best feature score.

    Parameters
    ----------
    transformer : fitted aeon MiniRocket.
    ranks : feature ranks from rank_kernels_by_discriminative_power.
    K : int, number of kernels to select.

    Returns
    -------
    kernel_mask : bool array [84 * n_dilations], True for selected kernels.
    selected_features : bool array [F], True for features of selected kernels.
    """
    _, _, dilations, nfpd, _ = transformer.parameters
    dilations = np.asarray(dilations)
    nfpd = np.asarray(nfpd)
    n_dil = len(dilations)
    n_kernels = 84
    total_kernels = n_kernels * n_dil

    # Map feature index to (dilation_idx, kernel_idx)
    feature_to_kernel = np.empty(len(ranks), dtype=np.int64)
    f_start = 0
    for j in range(n_dil):
        nf = int(nfpd[j])
        for k in range(n_kernels):
            feature_to_kernel[f_start:f_start + nf] = j * n_kernels + k
            f_start += nf

    # For each kernel, take the best feature score
    kernel_scores = np.zeros(total_kernels)
    for f_idx in ranks:  # iterate in order of descending score
        ki = feature_to_kernel[f_idx]
        if kernel_scores[ki] == 0:
            kernel_scores[ki] = 1  # mark as ranked

    # Rank kernels by their best feature's position in the global rank
    kernel_best_rank = np.full(total_kernels, len(ranks))
    for f_idx, ki in enumerate(feature_to_kernel):
        if f_idx < kernel_best_rank[ki]:
            kernel_best_rank[ki] = f_idx

    # Select top-K kernels
    kernel_order = np.argsort(kernel_best_rank)
    selected_kernels = set(kernel_order[:K])

    # Build masks
    kernel_mask = np.zeros(total_kernels, dtype=bool)
    for ki in selected_kernels:
        kernel_mask[ki] = True

    selected_features = np.zeros(len(ranks), dtype=bool)
    f_start = 0
    for j in range(n_dil):
        nf = int(nfpd[j])
        for k in range(n_kernels):
            ki = j * n_kernels + k
            if kernel_mask[ki]:
                selected_features[f_start:f_start + nf] = True
            f_start += nf

    return kernel_mask, selected_features


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def period_summary(periods, methods):
    """Summary statistics for period estimates."""
    periods = np.asarray(periods)
    methods = np.asarray(methods)
    return {
        'mean': float(periods.mean()),
        'median': float(np.median(periods)),
        'std': float(periods.std()),
        'min': float(periods.min()),
        'max': float(periods.max()),
        'acf_frac': float((methods == 'ACF').mean()),
        'fft_frac': float((methods == 'FFT').mean()),
        'fallback_frac': float((methods == 'FALLBACK').mean()),
    }
