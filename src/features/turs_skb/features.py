"""TURS-SKB feature extraction and utilities.

Thresholding abstraction (fairness rule, spec sec. 6):
Canonical MiniROCKET assigns each kernel 9 biases drawn deterministically
from the golden-ratio quantile sequence, fitted as quantiles of that
kernel's TRAIN-pool responses; features are PPVs (proportion of positions
with response strictly above the bias). We mirror EXACTLY this mechanism
for every structured family:

  biases_k = np.quantile(responses_of_k_on_TRAIN, _quantiles(9))
  PPV_k,b  = mean_t 1[r_k(t) > biases_k[b]]

Same abstraction, same feature semantics, same train-only fitting as
canonical MiniROCKET. No learned thresholds, no extra pooling statistics
(no mu / W / variance / phase), per spec sec. 12.

Response r_k(t) = sum_j k_j x_{t + j d} (dilation d), evaluated at all
t with t + (L-1) d <= T-1. Effective dilation is clipped per kernel to
the largest power of two that fits the signal length - the same rule
canonical MiniROCKET uses to clip its dilation grid to short series.

F5 (Energy) response: B(t) = E_short(t) - lam * E_long(t) with
E_w(t) = sum_j q_j x_{t+j}^2, q_j >= 0. This is a NONLINEAR energy
operator (the signal is squared before the moving weighted average) and
is mathematically distinct from the signed linear convolutions of F1-F4.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.features.turs_skb.kernels import BANK_BUILDERS, _norm, tent_window

# Golden-ratio quantile levels (verbatim mirror of aeon _quantiles).
_PHI = (np.sqrt(5.0) + 1.0) / 2.0
N_BIASES = 9  # canonical MiniROCKET features per kernel


def golden_quantiles(n):
    """[(i * phi) % 1 for i in 1..n] as float32 - identical to aeon."""
    return np.array([(_ * _PHI) % 1 for _ in range(1, n + 1)],
                    dtype=np.float32)


def _max_dilation(L, T):
    """Largest power-of-two dilation d with (L-1) d + 1 <= T (aeon rule)."""
    d = 1
    while d * 2 <= T and (L - 1) * (d * 2) + 1 <= T:
        d *= 2
    return d


def linear_kernel_response(X, k, d):
    """r[i, t] = sum_j k_j X[i, t + j d]; None if the kernel does not fit.

    Kernel taps are STRIDED by the dilation d and the response is evaluated
    at every position t with full support (n_pos = T - (L-1)d) - the same
    response model as canonical MiniROCKET and spec section 6. Implemented
    as a sum of shifted, scalar-scaled slices (BLAS-friendly, cost
    n * L * n_pos independent of d).
    """
    X = np.asarray(X, dtype=np.float32)
    n, T = X.shape
    L = k.size
    d = int(d)
    eff = (L - 1) * d + 1
    if eff > T:
        return None
    n_pos = T - eff + 1
    kf = np.ascontiguousarray(k, dtype=np.float32)
    out = np.empty((n, n_pos), dtype=np.float32)
    chunk = max(1, int(2 ** 22 // max(n_pos * L, 1)))  # ~16 MB work chunks
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        acc = np.zeros((b - a, n_pos), dtype=np.float32)
        for j in range(L):
            acc += kf[j] * X[a:b, j * d: j * d + n_pos]
        out[a:b] = acc
    return out


def energy_kernel_response(X, q_short, q_long, lam):
    """B(t) = E_short(t) - lam * E_long(t); E_w(t) = sum_j q_j x_{t+j}^2.

    NONLINEAR energy operator (see module docstring). Positions with full
    long-window support only.
    """
    X = np.asarray(X, dtype=np.float32)
    Xsq = X * X  # square once; q_j >= 0 windows applied to x^2
    n, T = X.shape
    ws, wl = q_short.size, q_long.size
    if wl > T:
        return None
    n_pos = T - wl + 1
    v_s = sliding_window_view(Xsq, ws, axis=1)[:, :n_pos, :]
    v_l = sliding_window_view(Xsq, wl, axis=1)[:, :n_pos, :]
    qs = np.ascontiguousarray(q_short, dtype=np.float32)
    ql = np.ascontiguousarray(q_long, dtype=np.float32)
    out = np.empty((n, n_pos), dtype=np.float32)
    chunk = max(1, int(2 ** 22 // max(n_pos * wl, 1)))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        out[a:b] = v_s[a:b] @ qs - float(lam) * (v_l[a:b] @ ql)
    return out


class SKBExtractor:
    """Fixed structured-kernel feature extractor (fit once on TRAIN).

    fit(X): builds all family banks, computes each kernel's effective
    dilation for the signal length, pools train responses and stores the 9
    golden-quantile biases per kernel. transform(X) returns
    {family_name: [n, n_kernels*9] float64 PPV matrix}.

    Identical (kernel vector, effective dilation) pairs are computed once
    and shared (deduplication), per spec sec. 13 ("do not duplicate
    identical kernels").
    """

    def __init__(self, families=("morphological", "derivative", "wavelet",
                                 "gabor", "energy")):
        self.families = list(families)
        self.kernels_ = None       # {family: [(kern_or_q, meta, d_eff)]}
        self.biases_ = None        # {family: [np.ndarray(9) per kernel]}
        self.duplicates_ = None    # {family: n_bank_entries_collapsed}
        self.dedup_ = {}           # (kernel_bytes, d_eff) -> response pool

    # ------------------------------------------------------------------
    def _response(self, X, kern, meta, d_eff):
        if meta["kind"] == "energy":
            q_long = tent_window(int(meta["w_long"]))
            return energy_kernel_response(X, kern, q_long, meta["lambda"])
        return linear_kernel_response(X, kern, d_eff)

    def fit(self, X):
        """Fit biases on TRAIN rows; collapse duplicates honestly.

        For each bank entry the dilation is capped to the largest power of
        two that fits the series length. Entries whose (kernel bytes,
        effective dilation) coincide after capping produce IDENTICAL
        features, so they are collapsed (spec sec. 13: document rather than
        duplicate identical kernels). The collapse count per family is
        stored in ``self.duplicates_`` and reported per dataset.
        """
        X = np.asarray(X, dtype=np.float32)
        T = X.shape[1]
        q_levels = golden_quantiles(N_BIASES)
        self.kernels_, self.biases_, self.duplicates_ = {}, {}, {}
        self.dedup_ = {}
        for fam in self.families:
            entries, bias_list = [], []
            seen_keys = set()
            n_dup = 0
            for kern, meta in BANK_BUILDERS[fam]():
                if meta["kind"] == "energy":
                    d_eff = 1  # energy windows carry their own scale
                    key = (b"E", int(meta["w_short"]), float(meta["lambda"]))
                else:
                    d_eff = min(int(meta["dilation"]),
                                _max_dilation(kern.size, T))
                    key = (kern.tobytes(), d_eff)
                if key in seen_keys:
                    n_dup += 1
                    continue
                resp = self._response(X, kern, meta, d_eff)
                if resp is None:  # kernel cannot fit this T; skip
                    n_dup += 1
                    continue
                seen_keys.add(key)
                pool = resp.ravel()
                biases = np.quantile(pool, q_levels).astype(np.float64)
                entries.append((kern, meta, d_eff))
                bias_list.append(biases)
            self.kernels_[fam] = entries
            self.biases_[fam] = bias_list
            self.duplicates_[fam] = n_dup
        self.dedup_ = {}  # free pooled train responses
        return self

    def transform(self, X):
        """{family: [n, sum over kernels of 9]} PPV features (float64)."""
        if self.kernels_ is None:
            raise RuntimeError("SKBExtractor.fit must run before transform")
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        q_levels = golden_quantiles(N_BIASES)
        out = {}
        for fam in self.families:
            cols = []
            for (kern, meta, d_eff), biases in zip(self.kernels_[fam],
                                                   self.biases_[fam]):
                resp = self._response(X, kern, meta, d_eff)
                ppv = (resp[:, :, None] > biases[None, None, :]).mean(axis=1)
                cols.append(ppv.astype(np.float64))
            out[fam] = np.concatenate(cols, axis=1) if cols else \
                np.empty((n, 0))
        return out

    def metadata(self):
        """Per-kernel records: family, kind, grid params, support, d_eff."""
        rows = []
        for fam in self.families:
            for i, (kern, meta, d_eff) in enumerate(self.kernels_[fam]):
                rows.append({"family": fam, "kernel_index": i,
                             "support": int(kern.size),
                             "effective_dilation": int(d_eff),
                             **meta})
        return rows

    def family_dims(self):
        return {fam: len(self.kernels_[fam]) * N_BIASES
                for fam in self.families}


# ============================================================================
# Model assembly: block concatenation + train-only block standardization
# ============================================================================
class BlockStandardizer:
    """Per-block z-standardization; statistics fitted on the fit rows only.

    Blocks with near-zero variance are left unscaled (std -> 1.0), exactly
    like the canonical BlockScaler of run_turs_final.py.
    """

    def __init__(self, block_dims):
        self.block_dims = list(block_dims)

    def fit(self, X):
        b = np.cumsum([0] + list(self.block_dims))
        self.means_, self.stds_ = [], []
        for a, c in zip(b[:-1], b[1:]):
            if c == a:  # empty block (family skipped for this T)
                self.means_.append(None)
                self.stds_.append(None)
                continue
            blk = X[:, a:c]
            m = blk.mean(axis=0, keepdims=True)
            s = blk.std(axis=0, keepdims=True)
            s = np.where(s < 1e-8, 1.0, s)
            self.means_.append(m)
            self.stds_.append(s)
        return self

    def transform(self, X):
        b = np.cumsum([0] + list(self.block_dims))
        outs = []
        for (a, c), m, s in zip(zip(b[:-1], b[1:]), self.means_, self.stds_):
            if c == a:
                outs.append(X[:, a:c])
                continue
            outs.append((X[:, a:c] - m) / s)
        return np.concatenate(outs, axis=1)


def assemble_blocks(block_list):
    """Column-concatenate feature blocks; returns X and per-block dims."""
    dims = [b.shape[1] for b in block_list]
    return np.concatenate(block_list, axis=1), dims


# ============================================================================
# Representation diagnostics
# ============================================================================
def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019): HSIC(X,Y) / sqrt(HSIC(X)HSIC(Y))."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    xy = np.sum((Xc.T @ Yc) ** 2)
    xx = np.sum((Xc.T @ Xc) ** 2)
    yy = np.sum((Yc.T @ Yc) ** 2)
    denom = np.sqrt(xx * yy)
    if denom < 1e-300:
        return float("nan")
    return float(xy / denom)


def mean_abs_cross_correlation(A, B, max_feat=400, seed=42):
    """Mean |Pearson r| between columns of A and B (feature-subsampled)."""
    rng = np.random.default_rng(seed)
    ia = rng.choice(A.shape[1], min(max_feat, A.shape[1]), replace=False)
    ib = rng.choice(B.shape[1], min(max_feat, B.shape[1]), replace=False)
    Ac = A[:, ia] - A[:, ia].mean(axis=0, keepdims=True)
    Bc = B[:, ib] - B[:, ib].mean(axis=0, keepdims=True)
    na = np.sqrt((Ac ** 2).sum(axis=0))
    nb = np.sqrt((Bc ** 2).sum(axis=0))
    ok_a, ok_b = na > 1e-12, nb > 1e-12
    if not ok_a.any() or not ok_b.any():
        return float("nan")
    C = (Ac[:, ok_a].T @ Bc[:, ok_b]) / np.outer(na[ok_a], nb[ok_b])
    return float(np.abs(C).mean())
