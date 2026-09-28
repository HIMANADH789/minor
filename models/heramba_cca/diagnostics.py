"""Effective-rank and variance diagnostics for HERAMBA-CCA."""
from __future__ import annotations

import numpy as np

from models.heramba_cca.cca import effective_rank, numerical_rank


def rank_report(X: np.ndarray, name: str) -> dict:
    """Spectrum + rank diagnostics for one representation matrix."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    if Xc.shape[1] > 4000:                      # repository stride convention
        Xc_s = Xc[:, ::4]
    else:
        Xc_s = Xc
    s = np.linalg.svd(Xc_s, compute_uv=False)
    s = s[s > 1e-12]
    energy = s ** 2
    tot = energy.sum()
    cum = np.cumsum(energy) / max(tot, 1e-300)
    return {
        "name": name,
        "n_samples": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "numerical_rank": numerical_rank(X),
        "effective_rank": round(effective_rank(X), 2),
        "singular_values_top50": [round(float(v), 5) for v in s[:50]],
        "explained_variance_ratio_top50": [round(float(v), 6)
                                           for v in (energy[:50] /
                                                     max(tot, 1e-300))],
        "cumulative_variance_top50": [round(float(v), 6) for v in cum[:50]],
    }


def variance_partition(H: np.ndarray, H_unique: np.ndarray,
                       H_shared_hat: np.ndarray) -> dict:
    """Fraction of H variance shared vs unique (linear, train-fit)."""
    H = np.asarray(H, dtype=np.float64)
    Hu = np.asarray(H_unique, dtype=np.float64)
    Hs = np.asarray(H_shared_hat, dtype=np.float64)
    vh = float(((H - H.mean(0)) ** 2).sum())
    vu = float(((Hu - Hu.mean(0)) ** 2).sum())
    vs = float(((Hs - Hs.mean(0)) ** 2).sum())
    frac_u = vu / max(vh, 1e-300)
    frac_s = max(0.0, 1.0 - frac_u)
    return {
        "H_total_var": round(vh, 6),
        "H_shared_var": round(vs, 6),
        "H_unique_var": round(vu, 6),
        "fraction_H_variance_shared": round(frac_s, 6),
        "fraction_H_variance_unique": round(frac_u, 6),
    }
