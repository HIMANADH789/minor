"""
TURS-MSW-10K: fixed-budget multi-scale windowed MiniROCKET
==========================================================

The ONLY architectural change vs canonical MiniROCKET is the temporal
pooling ALLOCATION of a FIXED ~10K feature budget:

  M0 (canonical): all F features, global PPV
  M1: N_G global PPV + N_M medium kernels x 2 regional PPVs      (F total)
  M2: N_G global PPV + N_L local kernels x 4 regional PPVs       (F total)
  M3: N_G global + N_M medium x2 + N_L local x4                  (F total)
  M4 (diagnostic): N_M medium x2 + N_L local x4                  (F total)

FAIRNESS (spec sec. 2/5/14/41):
- ONE canonical fitted aeon MiniRocket per dataset (seed 42). Its kernel
  bank / biases / dilations / responses / activation masks are shared by
  every variant; no refitting per variant.
- The canonical feature set K = {k_1..k_F} is partitioned DETERMINISTICALLY
  into disjoint scale subsets by a STRIDED modular rule
      scale(k_i) = i mod n_scales_of_that_variant  (over the F indices)
  which distributes aeon's dilation-major feature layout uniformly across
  scales (a contiguous prefix split would confound scale with dilation).
  Per-variant partitions are independent disjoint cells of the same rule;
  kernel groups are never duplicated within a variant.
- Budget equations are solved EXACTLY for F = 9996 (see allocation_plan).

Windows (spec sec. 13) are the SAME validated definitions as
experiments/turs_msw (deterministic fractions of T, 50% overlap):
  Global: [0, T)
  Medium: w=ceil(3T/4): [0,w), [T-w,T)          (2 regions)
  Local:  w=ceil(T/2), s=floor(T/4): 4 regions
For padding1==1 kernels, aeon pools only the valid axis
C[padding:T-padding]; regional windows are mapped onto that axis (the SAME
responses, re-indexed). A window fully inside the padding yields 0.0 —
aeon's own pooling rule.

All features are PPV rates in [0,1] on one natural scale, so the canonical
RAW ridge protocol is retained (spec sec. 16 block statistics still reported).

0 trainable neural parameters. No routers/gates/ensembles/new kernels.
"""
import math

import numpy as np

from experiments.turs_msw.msw import build_regions, msw_transform_uni

# ----------------------------------------------------------------------
# Deterministic strided allocation (spec sec. 6)
# ----------------------------------------------------------------------
def allocate_strided(F, n_groups):
    """Partition feature indices [0, F) into n_groups disjoint, near-equal
    deterministic cells: cell g = { i : i mod n_groups == g }.

    Strided (not contiguous) so that aeon's dilation-major layout is spread
    evenly across scales: each dilation block contributes ~1/n_groups of its
    features to each scale. Cells differ in size by at most 1."""
    cells = [[] for _ in range(n_groups)]
    for i in range(F):
        cells[i % n_groups].append(i)
    return [np.array(c, dtype=np.int64) for c in cells]


def solve_budget(F, mults, budget_share=None, tol=4):
    """Find non-negative integer kernel counts for feature multiplicities
    `mults` (e.g. (1,2,4)) whose total feature count is the CLOSEST
    achievable to F within +/-tol. Groups 1..p-1 enumerate ceil/floor of
    their budget share (ceil FIRST, so the indivisible residue lands on the
    x1-multiplicity global group, keeping every scale's kernel count within
    its strided cell size); the first group absorbs the remainder. For
    F = 9996 ALL plans land EXACTLY on 9996:
      M1: 4998 G x1 + 2499 M x2                    = 9996
      M2: 4996 G x1 + 1250 L x4                    = 9996
      M3: 3332 G + 1666 M x2 + 833 L x4            = 9996
      M4: 2498 M x2 + 1250 L x4                    = 9996
    Returns (counts, total)."""
    import itertools
    if budget_share is None:
        budget_share = [1.0 / len(mults)] * len(mults)
    best = None
    for off in range(-tol, tol + 1):
        t = F + off
        if t < sum(mults):
            continue
        # enumerate floor/ceil of the share for groups 1..p-1
        base = []
        for m, s in zip(mults[1:], budget_share[1:]):
            raw = t * s / m
            base.append((int(math.ceil(raw)), int(math.floor(raw))))
        for combo in itertools.product(*base) if base else [()]:
            used = sum(c * m for c, m in zip(combo, mults[1:]))
            rem = t - used
            if rem < 0 or rem % mults[0] != 0:
                continue
            counts = (rem // mults[0],) + combo
            total = rem + used
            if best is None or abs(total - F) < abs(best[1] - F):
                best = (list(counts), total)
    if best is None:
        raise ValueError(f"no allocation within +/-{tol} of {F} for "
                         f"mults={mults}")
    return best


def allocation_plan(F=9996):
    """Exact 10K-budget plans (documented, spec sec. 8-11/28).

    Every plan sums to exactly F features:
      M1: 4998 global x1 + 2499 medium x2            = 9996
      M2: 5000 global x1 + 1249 local x4             = 9996
      M3: 3332 global x1 + 1666 medium x2 + 833 local x4 = 9996
      M4: 2500 medium x2 + 1249 local x4             = 9996
    Returns dict variant -> {"n_global", "n_medium", "n_local", "total"}.
    Exact totals (documented):
      M0: 9996 global x1                                = 9996
      M1: 4998 global x1 + 2499 medium x2               = 9996
      M2: 4996 global x1 + 1250 local x4                = 9996
      M3: 3332 global x1 + 1666 medium x2 + 833 local x4 = 9996
      M4: 2498 medium x2 + 1250 local x4                = 9996"""
    plans = {}
    plans["M0"] = {"n_global": F, "n_medium": 0, "n_local": 0, "total": F}
    (g1, m1), t1 = solve_budget(F, (1, 2), [0.5, 0.5])
    plans["M1"] = {"n_global": g1, "n_medium": m1, "n_local": 0,
                   "total": g1 + 2 * m1}
    (g2, l2), t2 = solve_budget(F, (1, 4), [0.5, 0.5])
    plans["M2"] = {"n_global": g2, "n_medium": 0, "n_local": l2,
                   "total": g2 + 4 * l2}
    (g3, m3, l3), t3 = solve_budget(F, (1, 2, 4), [1 / 3, 1 / 3, 1 / 3])
    plans["M3"] = {"n_global": g3, "n_medium": m3, "n_local": l3,
                   "total": g3 + 2 * m3 + 4 * l3}
    (m4, l4), t4 = solve_budget(F, (2, 4), [0.5, 0.5])
    plans["M4"] = {"n_global": 0, "n_medium": m4, "n_local": l4,
                   "total": 2 * m4 + 4 * l4}
    assert t1 == g1 + 2 * m1 and t2 == g2 + 4 * l2
    assert t3 == g3 + 2 * m3 + 4 * l3 and t4 == 2 * m4 + 4 * l4
    for v, p in plans.items():
        assert abs(p["total"] - F) <= 4, f"{v} budget {p['total']} != {F}"
        assert p["total"] <= F + 4, f"{v} exceeds budget"
    return plans


def scale_indices(F, variant, plan=None):
    """Deterministic disjoint kernel-index sets per scale for a variant.

    Strided modular rule over [0, F), documented (spec sec. 6): each scale
    draws kernels from alternating strided cells of i mod 2 (2-scale
    variants) or i mod 3 (M3), so aeon's dilation-major layout is spread
    uniformly across scales. Kernel counts per scale are set by the budget
    plan, NOT by cell size; the required counts (4998 G + 2499 M etc.) are
    all <= their cell sizes for F = 9996. Within a scale, the SAME index
    set feeds each regional window (features differ by window, not kernel).

    Disjointness ACROSS scales within a variant is verified by the unit
    tests (no kernel in two scales of the same variant).

    Returns dict scale -> sorted np.array of kernel feature-indices."""
    if plan is None:
        plan = allocation_plan(F)[variant]
    if variant == "M0":
        return {"global": np.arange(F, dtype=np.int64)}
    if variant in ("M1", "M2", "M4"):
        cells = allocate_strided(F, 2)
        if variant == "M1":
            assert plan["n_global"] <= len(cells[0])
            assert plan["n_medium"] <= len(cells[1])
            gi = cells[0][:plan["n_global"]]
            mi = cells[1][:plan["n_medium"]]
            return {"global": np.sort(gi), "medium": np.sort(mi)}
        if variant == "M2":
            gi = cells[0][:plan["n_global"]]
            li = cells[1][:plan["n_local"]]
            return {"global": np.sort(gi), "local": np.sort(li)}
        mi = cells[0][:plan["n_medium"]]
        li = cells[1][:plan["n_local"]]
        return {"medium": np.sort(mi), "local": np.sort(li)}
    if variant == "M3":
        cells = allocate_strided(F, 3)
        gi = cells[0][:plan["n_global"]]
        mi = cells[1][:plan["n_medium"]]
        li = cells[2][:plan["n_local"]]
        return {"global": np.sort(gi), "medium": np.sort(mi),
                "local": np.sort(li)}
    raise ValueError(variant)


# ----------------------------------------------------------------------
# Budgeted variant assembly from the full 7-block regional features
# ----------------------------------------------------------------------
def variant_matrix_10k(full_feats, F, variant, plan=None, ind=None):
    """Assemble a fixed-budget variant matrix from the full regional feature
    block [G(F) | M0(F) | M1(F) | L0..L3(F)] (from msw.msw_transform_uni).

    Global kernel i contributes its global PPV (== canonical feature i,
    verified separately). Medium kernel i contributes 2 features (M0_i, M1_i).
    Local kernel i contributes 4 features (L0..L3_i). Output dims:
      M0: F; M1: nG + 2 nM; M2: nG + 4 nL; M3: nG + 2 nM + 4 nL; M4: 2 nM + 4 nL
    Each equals exactly F for the F=9996 plans."""
    if plan is None:
        plan = allocation_plan(F)[variant]
    if ind is None:
        ind = scale_indices(F, variant, plan)
    G = full_feats[:, :F]
    M0b = full_feats[:, F:2 * F]
    M1b = full_feats[:, 2 * F:3 * F]
    L = full_feats[:, 3 * F:7 * F]           # [L0|L1|L2|L3] each F wide

    parts = []
    if "global" in ind and plan["n_global"] > 0:
        parts.append(G[:, ind["global"]])
    if "medium" in ind and plan["n_medium"] > 0:
        mi = ind["medium"]
        parts.append(np.concatenate([M0b[:, mi], M1b[:, mi]], axis=1))
    if "local" in ind and plan["n_local"] > 0:
        li = ind["local"]
        parts.append(np.concatenate([L[:, li], L[:, F + li],
                                     L[:, 2 * F + li], L[:, 3 * F + li]],
                                    axis=1))
    X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    return np.ascontiguousarray(X)


# ----------------------------------------------------------------------
# Diagnostics helpers (memory-safe)
# ----------------------------------------------------------------------
def linear_cka_ngram(X, Y):
    """Memory-safe linear CKA (Kornblith et al. 2019) via n x n sample Grams:
    ||Xc.T @ Yc||_F^2 == tr(Gx @ Gy). Never allocates F x F matrices."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Gx = Xc @ Xc.T
    Gy = Yc @ Yc.T
    xy = float(np.sum(Gx * Gy))
    xx = float(np.sum(Gx * Gx))
    yy = float(np.sum(Gy * Gy))
    denom = math.sqrt(xx * yy)
    if denom < 1e-300:
        return float("nan")
    return xy / denom


def mean_abs_corr(X, Y, k=400, seed=0):
    """Mean |Pearson corr| between random column subsets of two blocks."""
    rng = np.random.RandomState(seed)
    kx = min(k, X.shape[1])
    ky = min(k, Y.shape[1])
    A = X[:, rng.choice(X.shape[1], kx, replace=False)].astype(np.float64)
    B = Y[:, rng.choice(Y.shape[1], ky, replace=False)].astype(np.float64)
    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    C = (Ac.T @ Bc) / (np.linalg.norm(Ac, axis=0)[:, None]
                       * np.linalg.norm(Bc, axis=0)[None, :] + 1e-12)
    return float(np.abs(C).mean())


def block_stats(B):
    """Sec. 16 block statistics on train features."""
    B = np.asarray(B)
    return {
        "min": round(float(B.min()), 6), "max": round(float(B.max()), 6),
        "mean": round(float(B.mean()), 6), "std": round(float(B.std()), 6),
        "near_zero_var_cols": int((B.std(0) < 1e-8).sum()),
        "nan": int(np.isnan(B).sum()), "inf": int(np.isinf(B).sum()),
    }
