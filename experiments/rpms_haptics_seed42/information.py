"""Information tracking (Tracks 1-5, 8) for the RPMS branch decomposition.

All diagnostics use TRAIN/VALIDATION data only.  None of these functions
touches test labels or test features, and none can alter the fixed
3332/3332/3332 allocation (they are read-only over the constructed branch
matrices).

Because all three branches have equal width (3332), between-branch
similarity is well-defined entry-wise (matrix correlation) and per-sample
(profile correlation); sample-level CKA (linear) summarizes redundancy.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score

from experiments.rcmkn_haptics_seed42.runner import macro_f1

ALPHAS = np.logspace(-4, 4, 20)
BRANCHES = ("G", "H", "HydraH")


# ------------------------------------------------------------------ #
# Track 1: basic feature information per branch                       #
# ------------------------------------------------------------------ #
def branch_feature_stats(F, name):
    x = np.asarray(F, dtype=np.float64)
    return {
        "name": name,
        "dim": int(x.shape[1]),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "fraction_near_zero": float((np.abs(x) < 1e-8).mean()),
        "effective_rank": _effective_rank(x),
    }


def _effective_rank(x, energy=0.99):
    """Numerical rank capturing `energy` of the total variance
    (stable on N=155 x 3332 via SVD on the centered matrix)."""
    xc = x - x.mean(0, keepdims=True)
    # SVD of (155, 3332) is cheap
    s = np.linalg.svd(xc, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(e) / tot
    return int(np.searchsorted(c, energy) + 1)


# ------------------------------------------------------------------ #
# Track 2: between-branch redundancy                                  #
# ------------------------------------------------------------------ #
def _matrix_corr(A, B):
    """Pearson correlation over all paired entries."""
    a = A.ravel().astype(np.float64)
    b = B.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _profile_corr(A, B):
    """Mean per-sample Pearson correlation between branch profiles."""
    ac = A - A.mean(1, keepdims=True)
    bc = B - B.mean(1, keepdims=True)
    num = (ac * bc).sum(1)
    den = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1)
    ok = den > 0
    return float((num[ok] / den[ok]).mean()) if ok.any() else 0.0


def _linear_cka(A, B):
    """Sample-level linear CKA (Kornblith et al. 2019)."""
    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    num = np.linalg.norm(Ac.T @ Bc, "fro") ** 2
    den = np.linalg.norm(Ac.T @ Ac, "fro") * np.linalg.norm(Bc.T @ Bc, "fro")
    return float(num / den) if den > 0 else 0.0


def branch_redundancy(branches):
    """branches: dict name -> (N, d) matrix. Returns pairwise similarity."""
    out = {}
    for i, a in enumerate(BRANCHES):
        for b in BRANCHES[i + 1:]:
            A, B = branches[a], branches[b]
            out[f"{a}~{b}"] = {
                "matrix_pearson": _matrix_corr(A, B),
                "profile_pearson_mean": _profile_corr(A, B),
                "linear_cka": _linear_cka(A, B),
            }
    return out


# ------------------------------------------------------------------ #
# Tracks 3-4: incremental gains and leave-one-branch-out              #
# ------------------------------------------------------------------ #
def _fit_val(F, ytrva, n_tr, yva):
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(F[:n_tr], ytrva[:n_tr])
    pred = ridge.predict(F[n_tr:])
    return {
        "val_macro_f1": round(macro_f1(yva, pred), 4),
        "selected_alpha": float(ridge.alpha_),
        "dim": int(F.shape[1]),
    }


def incremental_gains(branches, ytrva, n_tr, yva):
    """Track 3: all single/pair/full combos on VALIDATION ONLY."""
    G, H, X = (branches[b] for b in BRANCHES)
    combos = {
        "G": np.hstack([G]),
        "H": np.hstack([H]),
        "HydraH": np.hstack([X]),
        "G+H": np.hstack([G, H]),
        "G+HydraH": np.hstack([G, X]),
        "H+HydraH": np.hstack([H, X]),
        "G+H+HydraH": np.hstack([G, H, X]),
    }
    return {name: _fit_val(F, ytrva, n_tr, yva) for name, F in combos.items()}


def leave_one_branch_out(branches, ytrva, n_tr, yva):
    """Track 4: full minus each branch (validation only)."""
    G, H, X = (branches[b] for b in BRANCHES)
    combos = {
        "full-minus-G": np.hstack([H, X]),
        "full-minus-H": np.hstack([G, X]),
        "full-minus-HydraH": np.hstack([G, H]),
    }
    return {name: _fit_val(F, ytrva, n_tr, yva) for name, F in combos.items()}


# ------------------------------------------------------------------ #
# Track 5: randomized-branch control (validation only)                #
# ------------------------------------------------------------------ #
def permutation_nulls(branches, ytrva, n_tr, yva, seed=42, n_rep=5):
    """Per-row permutation of each branch (destroys sample-feature alignment
    within the branch, keeps marginals) -> validation Macro-F1 of the
    permuted branch ALONE.  A branch whose solo score collapses to a
    permuting-invariant level carries sample-specific signal."""
    rng = np.random.RandomState(seed)
    out = {}
    for b in BRANCHES:
        F = branches[b]
        base = _fit_val(F, ytrva, n_tr, yva)
        nulls = []
        for rep in range(n_rep):
            Fp = F.copy()
            for j in range(F.shape[1]):
                Fp[:, j] = Fp[rng.permutation(len(Fp)), j]
            nulls.append(_fit_val(Fp, ytrva, n_tr, yva)["val_macro_f1"])
        out[b] = {
            "solo_val_macro_f1": base["val_macro_f1"],
            "permuted_nulls": nulls,
            "null_mean": float(np.mean(nulls)),
            "signal_above_null": float(base["val_macro_f1"] - np.mean(nulls)),
        }
    return out


# ------------------------------------------------------------------ #
# Track 8: context/feature alignment between H and HydraH             #
# ------------------------------------------------------------------ #
def h_vs_hydrah_alignment(H, HydraH):
    """Per-feature correlation between the two context statistics on the
    same (equal-width) branch grids."""
    corr = np.zeros(H.shape[1])
    for j in range(H.shape[1]):
        a, b = H[:, j], HydraH[:, j]
        if a.std() > 0 and b.std() > 0:
            corr[j] = np.corrcoef(a, b)[0, 1]
        else:
            corr[j] = 0.0
    corr = np.nan_to_num(corr)
    return {
        "mean_corr": float(corr.mean()),
        "median_corr": float(np.median(corr)),
        "fraction_strong_corr": float((np.abs(corr) > 0.7).mean()),
        "corr_hist": np.histogram(corr, bins=20, range=(-1, 1))[0].tolist(),
    }


def occupancy_stats(regimes, K=8):
    """VQ health (Track 7) from a hard-regime array (N, T).

    active codes, normalized entropy, perplexity, dominant-code fraction,
    dead-code count.  RPMS reuses the FROZEN R2 context, so no VQ training
    (hence no revival) occurs here -- revival_count is None by design.
    """
    reg = np.asarray(regimes)
    counts = np.bincount(reg.ravel(), minlength=K).astype(np.float64)
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    return {
        "counts": counts.astype(int).tolist(),
        "active_codes": int((counts > 0).sum()),
        "dead_codes": int((counts == 0).sum()),
        "normalized_entropy": ent / float(np.log(K)),
        "perplexity": float(np.exp(ent)),
        "dominant_code_fraction": float(p.max()),
        "revival_count": None,   # frozen context: no VQ training in RPMS
    }
