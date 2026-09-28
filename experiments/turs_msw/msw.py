"""
TURS-MSW: Multi-Scale Windowed MiniROCKET
=========================================

Controlled representation experiment. The ONLY change vs canonical
MiniROCKET is WHERE the PPV pooling happens:

  M0 (canonical): one global PPV per kernel-bias feature
  M1: global + 2 medium-region PPVs
  M2: global + 4 local-region PPVs
  M3 (TURS-MSW): global + 2 medium + 4 local PPVs   (7 features per feature)
  M4 (diagnostic): medium + local only (no global block)

FAIRNESS (spec sec. 11/41): the kernel bank, biases, dilations, response
arrays and activation masks are EXACTLY the fitted aeon MiniRocket's. The
regional extractor mirrors aeon's `_static_transform_uni` verbatim (same
C_alpha/C_gamma accumulation, same padding1 parity rule) and only replaces
the final mean over the full (valid) response axis by means over fixed
temporal regions of the SAME activation mask.

Deterministic region construction (spec sec. 8/9/17/18), for length T:

  Global  G = [0, T)

  Medium (2 regions, ~50% overlap between neighbours, full coverage):
      w_M = ceil(3T/4)   (stride s_M = T - w_M = floor(T/4))
      M_0 = [0, w_M),    M_1 = [T - w_M, T)
      -> neighbour overlap = 2*w_M - T = ~T/2 (50% of the signal)

  Local (4 equal-width regions, 50% overlap between neighbours):
      w_L = ceil(T/2),   s_L = floor(T/4)
      L_j = [j*s_L, min(j*s_L + w_L, T))  for j = 0..3
      -> neighbour overlap = w_L - s_L = ~T/4 (50% of a window);
         3*s_L + w_L >= T so the four regions cover [0, T).

All boundaries are fractions of T (dataset-agnostic).
For padding1==1 kernels, aeon pools only the valid axis
C[padding:T-padding]; regional windows are mapped onto that axis by
subtracting `padding` (the SAME responses, only re-indexed). A window that
falls entirely inside the padding (possible for small T and large dilation)
yields the constant 0.0 — an honest consequence of aeon's own pooling rule.

Feature layout is SCALE-MAJOR (blocks):
  [ global(F) | medium_0(F) | medium_1(F) | local_0(F) | ... | local_3(F) ]
where F = len(biases) (canonical feature count, e.g. 9996).

No new kernels, no routers/gates/ensembles, 0 trainable neural parameters.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ======================================================================
# Deterministic regions (spec sec. 8/9/17/18)
# ======================================================================
def build_regions(T):
    """Return dict scale -> list of (start, end) index pairs on [0, T)."""
    if T < 9:
        raise ValueError(f"T must be >= 9 for MiniRocket, got {T}")
    regions = {"global": [(0, T)]}

    w_m = int(math.ceil(3 * T / 4))          # medium window length
    regions["medium"] = [(0, w_m), (T - w_m, T)]

    w_l = int(math.ceil(T / 2))              # local window length
    s_l = max(1, int(math.floor(T / 4)))     # local stride
    regions["local"] = [(j * s_l, min(j * s_l + w_l, T)) for j in range(4)]

    for scale, regs in regions.items():
        for (a, b) in regs:
            assert 0 <= a < b <= T, f"empty/invalid window {scale} {(a, b)}"
    assert regions["medium"][0][0] == 0 and regions["medium"][1][1] == T
    assert regions["local"][0][0] == 0
    assert max(b for _, b in regions["local"]) == T
    return regions


def region_slices_for_valid_axis(regions, padding, T):
    """Map original-axis regions onto a padding1==1 valid axis (for tests)."""
    out = {}
    for scale, regs in regions.items():
        sl = []
        for (a, b) in regs:
            va, vb = max(a, padding), min(b, T - padding)
            sl.append((va - padding, vb - padding))
        out[scale] = sl
    return out


# ======================================================================
# Core regional PPV extraction (univariate)
# ======================================================================
def _make_kernel():
    from numba import njit, prange

    @njit(cache=True, fastmath=True, parallel=True)
    def msw_transform_uni(X, dilations, n_features_per_dilation, biases,
                          indices, m_arr, l_arr):
        """Block-major regional PPV features [G | M0 | M1 | L0..L3].

        X: [N, T] float32. Returns [N, F*(1+n_m+n_l)] float32.
        """
        n_cases, n_timepoints = X.shape
        n_dilations = len(dilations)
        F = len(biases)
        n_m = len(m_arr)
        n_l = len(l_arr)
        n_out = F * (1 + n_m + n_l)
        features = np.zeros((n_cases, n_out), dtype=np.float32)
        n_kernels = len(indices)
        max_f = 0
        for j in range(n_dilations):
            if n_features_per_dilation[j] > max_f:
                max_f = n_features_per_dilation[j]
        for i in prange(n_cases):
            _X = X[i]
            A = -_X
            G3 = 3 * _X
            B = np.empty((max_f, n_timepoints), dtype=np.uint8)
            P = np.empty((max_f, n_timepoints + 1), dtype=np.int64)
            f_start = 0
            for j in range(n_dilations):
                _padding0 = j % 2
                dilation = dilations[j]
                padding = (8 * dilation) // 2
                n_features = n_features_per_dilation[j]
                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[4] = G3
                start = dilation
                end = n_timepoints - padding
                for gamma_index in range(4):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G3[:end]
                    end += dilation
                for gamma_index in range(5, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G3[start:]
                    start += dilation
                for k in range(n_kernels):
                    f_end = f_start + n_features
                    _padding1 = (_padding0 + k) % 2
                    a0, b0, c0 = indices[k, 0], indices[k, 1], indices[k, 2]
                    C = C_alpha + C_gamma[a0] + C_gamma[b0] + C_gamma[c0]
                    if _padding1 == 0:
                        base = 0
                        Tv = n_timepoints
                    else:
                        base = padding
                        Tv = n_timepoints - 2 * padding
                    # activation bit rows + prefix sums (per feature)
                    for f in range(n_features):
                        b_f = biases[f_start + f]
                        s = 0
                        P[f, 0] = 0
                        for t in range(Tv):
                            if C[base + t] > b_f:
                                s += 1
                            P[f, t + 1] = s
                    g_denom = Tv
                    # global block
                    for f in range(n_features):
                        features[i, f_start + f] = P[f, Tv] / g_denom
                    # medium blocks
                    for w in range(n_m):
                        aa = m_arr[w, 0] - base
                        bb = m_arr[w, 1] - base
                        if aa < 0:
                            aa = 0
                        if bb > Tv:
                            bb = Tv
                        if aa > Tv:
                            aa = Tv
                        if bb < 0:
                            bb = 0
                        denom = bb - aa
                        for f in range(n_features):
                            if denom > 0:
                                features[i, F + w * F + f_start + f] = \
                                    (P[f, bb] - P[f, aa]) / denom
                            else:
                                features[i, F + w * F + f_start + f] = 0.0
                    # local blocks
                    for w in range(n_l):
                        aa = l_arr[w, 0] - base
                        bb = l_arr[w, 1] - base
                        if aa < 0:
                            aa = 0
                        if bb > Tv:
                            bb = Tv
                        if aa > Tv:
                            aa = Tv
                        if bb < 0:
                            bb = 0
                        denom = bb - aa
                        for f in range(n_features):
                            if denom > 0:
                                features[i, F * (1 + n_m) + w * F
                                         + f_start + f] = \
                                    (P[f, bb] - P[f, aa]) / denom
                            else:
                                features[i, F * (1 + n_m) + w * F
                                         + f_start + f] = 0.0
                    f_start = f_end
        return features

    return msw_transform_uni


_KERNEL = None


def get_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _make_kernel()
    return _KERNEL


def msw_transform_uni(X, parameters, indices, regions):
    """Public wrapper: regional PPV features for a 2D univariate batch.

    X: [N, T] float32; parameters: aeon MiniRocket `.parameters` tuple;
    indices: MiniRocket._indices; regions: dict from build_regions.
    """
    from aeon.transformations.collection.convolution_based._minirocket import (
        _static_fit,  # noqa: F401  (ensures module loaded / caches warm)
    )
    X = np.ascontiguousarray(X, dtype=np.float32)
    (_, _, dilations, n_features_per_dilation, biases) = parameters
    m_arr = np.array(regions["medium"], dtype=np.int64)
    l_arr = np.array(regions["local"], dtype=np.int64)
    idx = np.ascontiguousarray(indices, dtype=np.int32)
    return get_kernel()(X, dilations, n_features_per_dilation, biases,
                        idx, m_arr, l_arr)


# ======================================================================
# Blocks and variants
# ======================================================================
def msw_blocks(features, F, n_medium=2, n_local=4):
    """Return dict scale -> feature block (views, no copies)."""
    G_end = F
    M_end = G_end + n_medium * F
    return {
        "global": features[:, :G_end],
        "medium": features[:, G_end:M_end],
        "local": features[:, M_end:M_end + n_local * F],
    }


def variant_matrix(features, F, n_medium=2, n_local=4):
    """Assemble M0-M4 matrices per spec sec. 15/16 (scale-major blocks)."""
    b = msw_blocks(features, F, n_medium, n_local)
    G, Me, L = b["global"], b["medium"], b["local"]
    return {
        "M0": G,
        "M1": np.concatenate([G, Me], axis=1),
        "M2": np.concatenate([G, L], axis=1),
        "M3": features,
        "M4": np.concatenate([Me, L], axis=1),
    }
