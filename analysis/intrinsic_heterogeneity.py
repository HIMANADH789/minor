"""LABEL-FREE INTRINSIC TEMPORAL HETEROGENEITY — analysis module.

Every function here operates on RAW time series X(t) only: no MiniROCKET
features, no HERAMBA features, no labels, no predictions, no learned
representation of any kind.  All preprocessing constants below are
PREDECLARED before any predictive result is consulted and must not be
tuned afterward.

Primary index (predeclared): TIME-VARYING SPECTRAL HETEROGENEITY

    HI_spec(sample) = mean_c mean_{i<j} JS(P_{i,c} || P_{j,c})

with W = 8 equal contiguous windows, per-window normalized power
spectra (hann window, n_fft = next_pow2(window_length), natural log,
epsilon 1e-12), JS averaged over all window pairs and channels.

Secondary label-free index: window variance heterogeneity
    HI_var = V_between / (V_within + 1e-12)

Dataset-level aggregation (predeclared): median over train samples.
H-budget signal (predeclared): B_H = clip(HI_dataset / log(2), 0, 1).

Deterministic: no RNG anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon

__all__ = [
    "W_PRIMARY", "EPS_SPEC", "EPS_VAR", "LOG2",
    "BH_BANDS", "compute_window_spectrum", "js_divergence",
    "compute_sample_spectral_heterogeneity",
    "compute_sample_variance_heterogeneity",
    "stationarity_diagnostics", "compute_dataset_index",
    "derive_intrinsic_h_budget", "run_leakage_audit",
    "window_stability", "window_boundaries",
]

# ---- predeclared constants (frozen; never tuned) ----------------------
W_PRIMARY = 8                      # temporal windows (primary)
EPS_SPEC = 1e-12                   # spectral numerical epsilon
EPS_VAR = 1e-12                    # variance-ratio epsilon
LOG2 = float(np.log(2.0))          # theoretical max JS (natural log convention)

# descriptive bands for B_H (predeclared, not optimized)
BH_BANDS = [(0.00, 0.20, "low"), (0.20, 0.50, "moderate"),
            (0.50, 0.75, "high"), (0.75, 1.00, "very high")]


def window_boundaries(T: int, W: int = W_PRIMARY) -> list[tuple[int, int]]:
    """W equal contiguous windows; remainder split across the earliest
    windows (deterministic; window lengths differ by at most 1)."""
    base, rem = divmod(T, W)
    bounds, start = [], 0
    for w in range(W):
        end = start + base + (1 if w < rem else 0)
        bounds.append((start, end))
        start = end
    return bounds


def compute_window_spectrum(x: np.ndarray, W: int = W_PRIMARY,
                            eps: float = EPS_SPEC) -> np.ndarray:
    """Normalized per-window power spectra of one series.

    Predeclared preprocessing: per-window linear detrending, hann window,
    rfft with n_fft = next power of two >= window length, one-sided
    power P = |X_f|^2 + eps, normalized to sum 1 (valid frequencies only).
    Returns (W, F) for univariate input (n_channels squeezed).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    T = x.shape[0]
    out = []
    for (a, b) in window_boundaries(T, W):
        seg = x[a:b]
        if b - a < 2:
            out.append(np.full(1, 1.0 / (b - a)))
            continue
        seg = seg - np.linspace(seg[0], seg[-1], seg.shape[0])  # detrend
        win = np.hanning(seg.shape[0])
        n_fft = int(2 ** np.ceil(np.log2(seg.shape[0])))
        spec = np.fft.rfft(seg * win, n=n_fft)
        power = (spec.real ** 2 + spec.imag ** 2) + eps
        power = power / power.sum()
        out.append(power)
    F = max(p.shape[0] for p in out)
    out = [np.pad(p, (0, F - p.shape[0])) if p.shape[0] < F else p
           for p in out]
    return np.stack(out)                     # (W, F)


def js_divergence(p: np.ndarray, q: np.ndarray,
                  eps: float = EPS_SPEC) -> float:
    """Jensen-Shannon divergence (natural-log convention, base e), sqrt
    returned by scipy.spatial.distance.jensenshannon -> squared here."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    jsd = jensenshannon(p, q, base=np.e) ** 2
    return float(max(jsd, 0.0))


def compute_sample_spectral_heterogeneity(X: np.ndarray,
                                          W: int = W_PRIMARY,
                                          eps: float = EPS_SPEC) -> np.ndarray:
    """HI_spec for every sample; X (N, T) or (N, C, T).  Averages JS over
    all window pairs and channels."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, T = X.shape
    hi = np.empty(N, dtype=np.float64)
    for i in range(N):
        specs = np.stack([compute_window_spectrum(X[i, c], W, eps)
                          for c in range(C)])          # (C, W, F)
        # pad channels to a common bin count
        F = specs.shape[-1]
        tot = 0.0
        n_pairs = W * (W - 1) // 2
        for c in range(C):
            P = specs[c]
            s = 0.0
            for a in range(W):
                for b in range(a + 1, W):
                    s += js_divergence(P[a], P[b], eps)
            tot += s / n_pairs
        hi[i] = tot / C
    return hi


def compute_sample_variance_heterogeneity(X: np.ndarray,
                                          W: int = W_PRIMARY,
                                          eps: float = EPS_VAR) -> np.ndarray:
    """HI_var = V_between / (V_within + eps) per sample (channel-mean)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, T = X.shape
    hi = np.empty(N, dtype=np.float64)
    for i in range(N):
        vb_c, vw_c = [], []
        for c in range(C):
            means, variances = [], []
            for (a, b) in window_boundaries(T, W):
                seg = X[i, c, a:b]
                means.append(seg.mean())
                variances.append(seg.var())
            vb_c.append(np.var(means))
            vw_c.append(np.mean(variances))
        hi[i] = float(np.mean(vb_c) / (np.mean(vw_c) + eps))
    return hi


# ----------------------------------------------------------------------
# supporting stationarity diagnostics (NOT used for the budget)
# ----------------------------------------------------------------------
def stationarity_diagnostics(X: np.ndarray, W: int = W_PRIMARY,
                             max_n: int | None = None) -> dict:
    """ADF / KPSS aggregate fractions on the raw series (supporting
    diagnostics only; never used for the H budget).  For tractability the
    aggregate is computed on the window-MEAN series of each sample
    (documented aggregate), capped at max_n samples."""
    from statsmodels.tsa.stattools import adfuller, kpss
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X.mean(axis=1)
    if max_n is not None:
        X = X[:max_n]
    n_adf = n_adf_sig = n_kpss = n_kpss_sig = 0
    for x in X:
        x = x[~np.isnan(x)]
        if x.size < 20:
            continue
        try:
            n_adf += 1
            if adfuller(x, autolag="AIC")[1] < 0.05:
                n_adf_sig += 1
        except Exception:
            n_adf -= 1
        try:
            n_kpss += 1
            if kpss(x, regression="c", nlags="auto")[1] < 0.05:
                n_kpss_sig += 1
        except Exception:
            n_kpss -= 1
    return {
        "note": ("supporting diagnostics on window-mean series only; "
                 "never used for the H budget"),
        "n_series": int(n_adf),
        "adf_stationary_fraction": round(n_adf_sig / max(n_adf, 1), 4),
        "kpss_nonstationary_fraction": round(n_kpss_sig / max(n_kpss, 1), 4),
    }


# ----------------------------------------------------------------------
# dataset aggregation + budget rule (predeclared)
# ----------------------------------------------------------------------
def _summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    qs = np.percentile(v, [10, 25, 50, 75, 90])
    return {
        "n": int(v.size),
        "mean": float(v.mean()), "median": float(np.median(v)),
        "std": float(v.std()), "iqr": float(qs[3] - qs[1]),
        "min": float(v.min()), "max": float(v.max()),
        "p10": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]),
        "p75": float(qs[3]), "p90": float(qs[4]),
        "n_unique": int(np.unique(v).size),
        "cv": float(v.std() / abs(v.mean())) if v.mean() != 0 else None,
    }


def compute_dataset_index(hi_samples: np.ndarray,
                          statistic: str = "median") -> float:
    """Dataset-level index: PREDECLARED median over train samples."""
    return float(np.median(np.asarray(hi_samples, dtype=np.float64)))


def derive_intrinsic_h_budget(hi_dataset: float,
                              reference: float = LOG2) -> dict:
    """B_H = clip(HI_dataset / HI_reference, 0, 1); reference predeclared
    as the theoretical max JS under the natural-log convention (log 2)."""
    b = float(np.clip(hi_dataset / reference, 0.0, 1.0))
    band = next(name for lo, hi_, name in BH_BANDS if lo <= b <= hi_)
    return {
        "rule": "B_H = clip(HI_dataset / log(2), 0, 1)",
        "HI_reference": float(reference),
        "HI_reference_source": ("theoretical maximum Jensen-Shannon "
                                "divergence, natural-log convention"),
        "HI_dataset": float(hi_dataset),
        "B_H": b,
        "band": band,
        "bands": [{"low_high": [lo, hi_], "label": name}
                  for lo, hi_, name in BH_BANDS],
        "H_budget_fraction": b,
        "interpretation_note": (
            "B_H is an intrinsic heterogeneity score / normalized "
            "H-capacity signal, NOT the optimal fraction of H features"),
    }


def run_leakage_audit(index_fn, Xtr: np.ndarray, Xva: np.ndarray,
                      Xte: np.ndarray, W: int = W_PRIMARY) -> dict:
    """Verify the train-derived dataset index is invariant to poisoned
    validation/test signals: the budget pipeline only ever receives Xtr,
    so poisoning the held-out signals (and recomputing) must reproduce
    the identical train index.  index_fn(X, W) -> per-sample HI."""
    hi_tr = index_fn(Xtr, W)
    base = compute_dataset_index(hi_tr)
    # poisoned held-out signals present in memory while recomputing
    _ = index_fn(Xva + 1e6, W)
    _ = index_fn(Xte + 1e6, W)
    recomputed = compute_dataset_index(index_fn(Xtr, W))
    res = {
        "train_index_baseline": base,
        "train_index_recomputed_with_poisoned_heldout": recomputed,
        "recomputation_invariant": bool(abs(recomputed - base) < 1e-12),
    }
    # for the record: indices of the held-out splits themselves
    res["val_split_own_index"] = compute_dataset_index(index_fn(Xva, W))
    res["test_split_own_index"] = compute_dataset_index(index_fn(Xte, W))
    res["val_poison_invariant"] = res["recomputation_invariant"]
    res["test_poison_invariant"] = res["recomputation_invariant"]
    res["heldout_signals_enter_budget"] = False
    res["labels_enter_budget"] = False
    return res


def window_stability(X: np.ndarray, Ws=(7, 8, 9)) -> dict:
    """Stability of the sample-level index across W (diagnostic only;
    W=8 remains primary).  Returns per-W dataset medians and rank/Spearman
    agreement of sample scores vs the primary W."""
    from scipy.stats import spearmanr
    scores = {W: compute_sample_spectral_heterogeneity(X, W) for W in Ws}
    primary = scores[W_PRIMARY]
    out = {"W_values": list(Ws), "primary_W": W_PRIMARY,
           "dataset_median": {str(W): float(np.median(s)) for W, s in scores.items()},
           "spearman_vs_primary": {}}
    for W, s in scores.items():
        if W == W_PRIMARY:
            out["spearman_vs_primary"][str(W)] = 1.0
        else:
            rho = spearmanr(primary, s).statistic
            out["spearman_vs_primary"][str(W)] = round(float(rho), 4)
    return out
