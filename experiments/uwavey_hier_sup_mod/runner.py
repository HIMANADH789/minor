"""HIER-SUP-MOD - EB variance-moderated F ranking on the unified
telescoping pool (UWaveY, seed 42).

The next controlled ablation after HIER-SUP-CLEAN.  ONE substantive
change: the supervised ranking statistic.

    HIER-SUP-CLEAN:  raw ANOVA-F  (top_f_select, sklearn f_classif)
    HIER-SUP-MOD:    same ANOVA design (d_between = 7, d_error = N - 8
                     for EVERY candidate) -> candidate residual variances
                     s_j^2 = SSE_j / d_error -> SINGLE Smyth/limma
                     scaled-F EB prior  s^2 ~ s0^2 F(d_e, d0) fitted over
                     the FULL unified pool -> posterior moderated variance
                     s_t^2 = (d0 s0^2 + d_e s^2) / (d0 + d_e) ->
                     moderated F_t = MS_between / s_t^2 -> ONE global
                     top-19,992 ranking (ties by candidate_index).

Representation/classifier unchanged: telescoping pool
C = [D1 | D2 | D4 | D8 | D16] in R^(N x 469,812) (PPV^(0)=0 -> D1 =
PPV^(1)), canonical split, expanded MiniROCKET 84x238, frozen train-only
hierarchy, occupancy-weighted deltas, RidgeClassifierCV.  Explicitly
NOT used anywhere (no rho, no G/H split, no per-level quota, no
energy allocation, no local-fdr, no feature-value shrinkage, no
fallback filter, no occupancy-as-df): the EB machinery here is a
variance prior ONLY.

Safety gates: candidate-layout guard (identical semantic order on
train/val/test) and the production raw-F equivalence gate reconstructing
the HIER-SUP-CLEAN ranking from the same matrix.

All locked components are imported, not reimplemented.
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DS = "UWaveGestureLibraryY"
SEED = 42

B_BASE = 9996
B_HIGH = 19992
G_PART = 4998
H_KERNELS = B_HIGH - G_PART      # 14,994 base hierarchical kernel carriers
KEPT_K = [2, 4, 8, 16]
EDGES = sum(KEPT_K)              # 30 hierarchy edges (children summed)
H_CAND = EDGES * H_KERNELS       # 449,820 deeper telescoping candidates
D1 = B_HIGH                      # 19,992 root increments Delta^(1)
N_CAND = D1 + H_CAND             # 469,812
PPV0 = 0.0
N_CLASSES = 8
DF_BETWEEN = N_CLASSES - 1       # 7 - identical for every candidate
ALPHAS = np.logspace(-4, 4, 20)
MIN_OCCUPANCY = 0.01             # locked chain convention (1% of T)
T_LEN = 315
FALLBACK_MIN_COUNT = int(np.ceil(MIN_OCCUPANCY * T_LEN))   # 4

CAP_RESULTS = os.path.join(ROOT, "results", "uwavey_capacity_scaled", "seed42")
NEST_RESULTS = os.path.join(ROOT, "results", "uwavey_nested_hierarchical", "seed42")
CLEAN_DIR = os.path.join(ROOT, "results", "uwavey_hier_sup_clean", "seed42")
REF_MR_HIGH = 0.7477
GATE_TOL = 0.002
REFS = {"canonical_mr": 0.7543, "mr_high": 0.7477, "r5_high": 0.7772,
        "hier_high": 0.7230, "hier_high_sup": 0.7823,
        "hier_continuous_sup": 0.7599, "hier_sup_clean": 0.7599}

OUT = os.path.join(ROOT, "results", "uwavey_hier_sup_mod", "seed42")

# reused canonical pieces (imported, not reimplemented)
from experiments.uwavey_nested_hierarchical.runner import (   # noqa: E402
    _encoder_latents, compute_activations, load_context_model, load_data,
    ppv_all, ridge_eval, znorm)
from experiments.uwavey_nested_hierarchical.runner import delta_banks  # noqa: E402,E501
from experiments.uwavey_capacity_scaled.runner import (       # noqa: E402
    extractor_facts, fit_minirocket_nk, hier_banks_test, peak_mem_mb)
from experiments.uwavey_hier_high_sup.runner import col_meta  # noqa: E402
from models.nested_regimes.model import (                     # noqa: E402
    assign_levels, build_latent_tree, nesting_errors)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Chunked one-way ANOVA pieces (spec 8).  Design identical for every
# candidate: d_between = C - 1 = 7, d_error = N - C.  No occupancy-as-df.
# ===========================================================================
def anova_pieces_chunk(C_chunk, y):
    """(n, W) candidate slice -> per-candidate SS_between / SSE with the
    one-way class design.  NaN handling mirrors f_classif: constant or
    non-finite columns -> SS_between = 0, SSE = 0 (so s^2 = 0 and the
    raw-F convention below maps them to F = 0, exactly like the
    incumbent top_f_select's nan_to_num)."""
    n = C_chunk.shape[0]
    classes = np.unique(y)
    yv = np.searchsorted(classes, y)                    # 0..C-1
    Cm = len(classes)
    onehot = np.zeros((n, Cm))
    onehot[np.arange(n), yv] = 1.0
    grp = onehot.sum(axis=0)                            # (Cm,)
    tot = C_chunk.sum(axis=0)
    msum = C_chunk.T @ onehot                           # (W, Cm) class sums
    ssq = (C_chunk ** 2).T @ onehot                     # (W, Cm) class sqs
    ssb = ((msum ** 2) / np.maximum(grp, 1.0)[None, :]).sum(axis=1) \
        - tot ** 2 / n
    # stable within-class SSE (avoids grand-total cancellation):
    # SSE = sum_k [ sum_sq_k - sum_k^2 / n_k ]
    sse = (ssq - (msum ** 2) / np.maximum(grp, 1.0)[None, :]).sum(axis=1)
    ssb = np.maximum(ssb, 0.0)
    sse = np.maximum(sse, 0.0)
    bad = ~np.isfinite(C_chunk).all(axis=0)             # NaN/Inf columns
    if bad.any():
        ssb[bad] = 0.0
        sse[bad] = 0.0
    return ssb, sse


def anova_all(C, y, col_chunk=8192):
    """Full-pool ANOVA pieces; returns (ssb, sse) float64 (N_CAND,)."""
    n, W = C.shape
    ssb = np.empty(W, dtype=np.float64)
    sse = np.empty(W, dtype=np.float64)
    for c0 in range(0, W, col_chunk):
        c1 = min(c0 + col_chunk, W)
        b, e = anova_pieces_chunk(C[:, c0:c1], y)
        ssb[c0:c1] = b
        sse[c0:c1] = e
    return ssb, sse


# ===========================================================================
# Smyth/limma EB variance moderation (spec 10-16)
# ===========================================================================
def eb_prior_fit(s2, d_e):
    """Scaled-F prior s^2 ~ s0^2 F(d_e, d0) by Smyth log-moment matching.

    Var[log s^2] = trigamma(d_e/2) + trigamma(d0/2)  -> d0 (bracketed
    Brent);  log s0^2 = mean(log s^2) - digamma(d_e/2) + digamma(d0/2)
    - log(d0/d_e).

    s2 must be strictly positive.  Degenerate fits (Var[log] at/below the
    d0=inf limit) are reported with condition + reason + fallback rule.
    """
    from scipy.optimize import brentq
    from scipy.special import digamma, polygamma
    x = np.asarray(s2, dtype=np.float64)
    n = x.size
    m_log = float(np.mean(np.log(x)))
    v_log = float(np.var(np.log(x), ddof=1)) if n > 1 else 0.0
    tg_e = float(polygamma(1, d_e / 2.0))
    limit = tg_e                                        # d0 -> inf limit
    info = {"n": int(n), "d_e": d_e, "mean_log_s2": m_log,
            "var_log_s2": v_log, "trigamma_d_e_over_2": tg_e,
            "d0_inf_limit": limit}
    if not np.isfinite(v_log) or v_log <= limit + 1e-12:
        d0 = 1e12                                       # limiting behavior
        info.update({"condition": "degenerate_var_log_at_or_below_limit",
                     "reason": "Var[log s^2] <= trigamma(d_e/2); the "
                               "moment equation has no finite d0 solve",
                     "fallback": "d0 = 1e12 (near-infinite prior df; "
                                 "posterior ~ prior, minimal shrinkage)",
                     "degenerate": True})
    else:
        f = lambda d: float(polygamma(1, d / 2.0)) + tg_e - v_log
        lo, hi = 1e-6, 1e12
        if f(lo) < 0:
            d0 = 1e-6                                   # v_log above d0->0
            info.update({"condition": "d0_clamped_low",
                         "reason": "Var[log s^2] above the d0 -> 0 limit",
                         "fallback": "d0 = 1e-6 (near-degenerate prior)",
                         "degenerate": True})
        else:
            d0 = float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12,
                              maxiter=500))
            info["degenerate"] = False
    log_s0sq = m_log - digamma(d_e / 2.0) + digamma(d0 / 2.0) \
        - np.log(d0 / d_e)
    s0sq = float(np.exp(log_s0sq))
    info.update({"d0": d0, "s0_sq": s0sq})
    return d0, s0sq, info


def moderated_scores(ssb, sse, d_e, d0, s0sq, df_b=DF_BETWEEN):
    """Posterior moderated variance and moderated F (spec 14-15).

    s_t^2 = (d0 s0^2 + d_e s^2) / (d0 + d_e);  F_t = MS_between / s_t^2.
    Raw F is returned for the equivalence gate / diagnostics only.
    """
    s2 = sse / d_e
    st2 = (d0 * s0sq + d_e * s2) / (d0 + d_e)
    msb = ssb / df_b
    with np.errstate(divide="ignore", invalid="ignore"):
        f_raw = msb / s2          # nan on 0/0, inf on x/0 (IEEE semantics)
        f_mod = msb / st2         # st2 > 0 always -> finite everywhere
    return s2, st2, f_raw, f_mod


def rank_select(score, b_high):
    """One global ranking: score descending, ties by candidate index.
    Equivalent to the incumbent's argsort(-f, kind='stable') for tied
    scores."""
    return np.lexsort((np.arange(score.size), -score))[:b_high]


def raw_f_gate_convention(f_raw):
    """Exact top_f_select convention: nan_to_num(f, nan=0.0) applied to
    the raw F before ranking (constants -> 0; infinities preserved as
    finfo-max by nan_to_num's defaults)."""
    return np.nan_to_num(f_raw, nan=0.0)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (diagnostic only)."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum.accumulate(
        (p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def select_and_mask_min(counts, min_count):
    """Locked-chain fallback convention (models/nested_regimes): keep
    regimes with count >= min_count, else fall back to the argmax-count
    regime.  Returns (sel (n, K) bool, fallback (n,) bool)."""
    sel = counts >= min_count
    fb = ~sel.any(axis=1)
    if fb.any():
        sel[fb, np.argmax(counts[fb], axis=1)] = True
    return sel, fb


def fallback_diagnostics(reg16_dev):
    """Spec 19: per-level fallback fractions under the locked convention
    (DIAGNOSTIC ONLY - no candidate is filtered)."""
    rows = []
    for li, K in enumerate(KEPT_K):
        coarse = reg16_dev >> (4 - li)                  # level-K ids
        cnt = np.zeros((reg16_dev.shape[0], K), dtype=np.int64)
        for c in range(K):
            cnt[:, c] = (coarse == c).sum(axis=1)
        sel, fb = select_and_mask_min(cnt, FALLBACK_MIN_COUNT)
        rows.append({
            "level": K, "regimes": K,
            "n_samples": int(reg16_dev.shape[0]),
            "min_count": FALLBACK_MIN_COUNT,
            "mean_occupancy_fraction": float((cnt / T_LEN).mean()),
            "fallback_fraction": float(fb.mean()),
            "samples_with_fallback": int(fb.sum()),
            "n_affected_candidates": int(K * H_KERNELS),
            "share_of_pool": float(K * H_KERNELS / H_CAND)})
    return rows


def candidate_occupancy(reg16_dev, level_of_col):
    """Per-CANDIDATE mean branch occupancy (spec 20): for each delta
    candidate (kernel m, child c, level K), the mean over dev samples of
    the child-branch occupancy fraction n_{i,c}/T.  Candidates sharing a
    child share occupancy (fallback is regime-level).  Root candidates
    get occupancy 1.0 (visited by definition)."""
    occ = np.ones(N_CAND, dtype=np.float64)
    for li, K in enumerate(KEPT_K):
        coarse = reg16_dev >> (4 - li)                  # (n_dev,) ids 0..K-1
        frac = np.stack([(coarse == c).mean(axis=1) for c in range(K)],
                        axis=1)                          # (n_dev, K)
        child_mean = frac.mean(axis=0)                   # (K,)
        mask = level_of_col == K
        cols = np.where(mask)[0]
        # within-level block: child-major, kernel-minor (F_H per child)
        child_of_col = (cols - (D1 + sum(k * H_KERNELS
                                         for k in KEPT_K
                                         if k < K))) // H_KERNELS
        occ[cols] = child_mean[child_of_col]
    return occ


def occupancy_diagnostics(reg16_dev, y_dev, f_raw, f_mod, s2, st2,
                          level_of_col):
    """Spec 20/21: occupancy-vs-(variance/F) Spearman (candidate-level)
    and occupancy-vs-class one-way ANOVA with BH per level.  DIAGNOSTIC
    ONLY - nothing here feeds selection."""
    from scipy.stats import f as fdist, spearmanr
    occ = candidate_occupancy(reg16_dev, level_of_col)
    delta_mask = level_of_col != 1
    rows = []
    for lv_name, mask in ([(f"K{k}", level_of_col == k) for k in KEPT_K]
                          + [("delta_pool", delta_mask)]):
        cols = np.where(mask)[0]
        o = occ[cols]
        for name, stat, use_log in (("log_s2", s2, True),
                                    ("F_raw", f_raw, False),
                                    ("log_s2_tilde", st2, True)):
            v = np.log(np.maximum(stat[cols], 1e-300)) if use_log \
                else stat[cols]
            r = spearmanr(o, v)
            rows.append({"level": lv_name,
                         "n_candidates": int(cols.size),
                         "statistic": name,
                         "spearman_rho": float(r.correlation),
                         "p_value": float(r.pvalue)})
    # occupancy-vs-class association per regime (spec 21)
    classes = np.unique(y_dev)
    yv = np.searchsorted(classes, y_dev)
    n_dev = reg16_dev.shape[0]
    assoc_rows = []
    for li, K in enumerate(KEPT_K):
        coarse = reg16_dev >> (4 - li)
        for c in range(K):
            frac = (coarse == c).sum(axis=1).astype(np.float64) / T_LEN
            groups = [frac[yv == g] for g in range(len(classes))]
            if any(len(g) < 2 for g in groups):
                continue
            grand = frac.mean()
            ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
            F = (ssb / (len(classes) - 1)) / \
                (ssw / max(n_dev - len(classes), 1))
            p = float(fdist.sf(F, len(classes) - 1, n_dev - len(classes)))
            assoc_rows.append({"level": K, "regime": c,
                               "occupancy_mean": float(frac.mean()),
                               "F": float(F), "p_value": p})
    # BH per level
    for K in KEPT_K:
        idx = [i for i, r in enumerate(assoc_rows) if r["level"] == K]
        if not idx:
            continue
        p = np.array([assoc_rows[i]["p_value"] for i in idx])
        adj = bh_adjust(p)
        for i, a in zip(idx, adj):
            assoc_rows[i]["p_bh_adj"] = float(a)
    return rows, assoc_rows


def per_sample_level_means(C_dev, level_of_col):
    """Per-sample mean statistic per delta level (figure input); chunked
    so no (n_dev, K*F_H) copy materializes.  Call BEFORE C_dev is freed."""
    out = {}
    n_dev = C_dev.shape[0]
    for K in KEPT_K:
        mask = np.where(level_of_col == K)[0]
        acc = np.zeros(n_dev)
        for c0 in range(0, mask.size, 8192):
            c1 = min(c0 + 8192, mask.size)
            acc += C_dev[:, mask[c0:c1]].mean(axis=1) * (c1 - c0)
        out[K] = acc / mask.size
    return out


def summarize_variance_stats(level_of_col, s2, st2, f_raw, f_mod, sel_mask):
    """Spec 18/35: per-level medians + selected counts (diagnostic only)."""
    rows = []

    def _med_mask(mask, v):
        vv = v[mask]
        vv = vv[np.isfinite(vv)]
        return float(np.median(vv)) if vv.size else 0.0

    for lv in (1, 2, 4, 8, 16):
        m = level_of_col == lv
        safe_fraw = np.where(np.isfinite(f_raw) & (f_raw != 0), f_raw,
                             np.nan)
        rows.append({
            "level": lv,
            "candidate_count": int(m.sum()),
            "median_raw_variance": _med_mask(m, s2),
            "median_moderated_variance": _med_mask(m, st2),
            "median_raw_F": _med_mask(m, f_raw),
            "median_moderated_F": _med_mask(m, f_mod),
            "median_F_ratio": _med_mask(m, f_mod / safe_fraw),
            "selected_count": int(sel_mask[m].sum())})
    return rows


def overlap_stats(sel_raw, sel_mod, n_cand):
    """Spec 22/35: raw vs moderated top-B overlap."""
    a, b = set(sel_raw.tolist()), set(sel_mod.tolist())
    inter = a & b
    return {
        "n_candidates": int(n_cand),
        "budget": int(len(a)),
        "intersection": len(inter),
        "raw_only": len(a - b),
        "moderated_only": len(b - a),
        "jaccard": len(inter) / len(a | b),
        "entering": len(b - a), "leaving": len(a - b)}


# ===========================================================================
def main(smoke=False):
    global OUT
    if smoke:
        OUT = OUT + "_smoke"
    for sub in ("results", "figures", "diagnostics", "predictions"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    t00 = time.time()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}  smoke={smoke}")

    # ---- 1. canonical data, split identity ------------------------------
    d, Xtr, Xva, Xte, ytr, yva, yte = load_data()
    n_tr, n_va, n_te = len(ytr), len(yva), len(yte)
    y_dev = np.concatenate([ytr, yva]).astype(np.int64)
    if smoke:
        Xtr, ytr = Xtr[:200], ytr[:200]
        Xva, yva = Xva[:60], yva[:60]
        Xte, yte = Xte[:200], yte[:200]
        n_tr, n_va, n_te = len(ytr), len(yva), len(yte)
        y_dev = np.concatenate([ytr, yva]).astype(np.int64)
    ntr_ref = np.load(os.path.join(NEST_RESULTS, "train_indices.npy"))
    nva_ref = np.load(os.path.join(NEST_RESULTS, "val_indices.npy"))
    nte_ref = np.load(os.path.join(NEST_RESULTS, "test_indices.npy"))
    split_ok = (len(ntr_ref) == 761 and len(nva_ref) == 135 and
                len(nte_ref) == 3582)
    log(f"data: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}; "
        f"split identity vs stored uwavey_nested indices: {split_ok}")
    assert split_ok or smoke

    # ---- 2. frozen hierarchy (LOCKED; rebuilt deterministically) ---------
    model, _ = load_context_model(device)
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    lat = _encoder_latents(model, Xtr_z, device)
    C, tree_meta = build_latent_tree(lat.reshape(-1, lat.shape[-1]), seed=SEED)
    levels_tr = assign_levels(lat.reshape(-1, lat.shape[-1]), C)
    errs = nesting_errors(levels_tr)
    assert not errs, f"nesting violated: {errs}"
    assert [len(c) for c in C] == [1, 2, 4, 8, 16]
    reg16_tr = levels_tr[4].reshape(n_tr, -1).astype(np.int64)
    del lat
    if not smoke:
        saved = np.load(os.path.join(NEST_RESULTS,
                                     "hierarchy_assignments.npy"))
        assert np.array_equal(reg16_tr, saved), \
            "hierarchy differs from the stored artifact"
        log("hierarchy regression: reg16_tr == stored "
            "hierarchy_assignments.npy (byte-identical)")
    lat_va = _encoder_latents(model, Xva_z, device)
    reg16_va = assign_levels(lat_va.reshape(-1, lat_va.shape[-1]),
                             C)[4].reshape(n_va, -1).astype(np.int64)
    del lat_va
    lat_te = _encoder_latents(model, Xte_z, device)
    reg16_te = assign_levels(lat_te.reshape(-1, lat_te.shape[-1]),
                             C)[4].reshape(n_te, -1).astype(np.int64)
    del lat_te
    del model
    log("tree: levels 1,2,4,8,16 frozen (TRAIN latents only); "
        "val/test transform-only")

    # ---- 3. expanded MiniROCKET (LOCKED) ---------------------------------
    exHK = fit_minirocket_nk(Xtr_z, B_HIGH)
    fHK = extractor_facts(exHK)
    F_H = fHK["total_features"] - G_PART
    assert fHK["physical_kernels"] == 84
    if not smoke:
        stored_dims = json.load(open(os.path.join(
            CAP_RESULTS, "diagnostics", "dimensions.json")))
        assert fHK == stored_dims["extractors"]["expanded"], (fHK,)
        assert fHK["total_features"] == D1 and F_H == H_KERNELS \
            and 30 * F_H == H_CAND
        log(f"expanded extractor facts match stored: {fHK}")
    else:
        log(f"expanded extractor facts (smoke): {fHK}")

    # ---- 4. root increments D1 = PPV^(1) - PPV^(0), FULL width ----------
    GHK_trva = ppv_all(exHK, np.vstack([Xtr_z, Xva_z]))
    GHK_te = ppv_all(exHK, Xte_z)
    assert GHK_trva.shape == (n_tr + n_va, D1)
    assert GHK_te.shape == (n_te, D1)
    D1_dev = GHK_trva - PPV0
    D1_te = GHK_te - PPV0
    assert np.array_equal(D1_dev, GHK_trva) and np.array_equal(D1_te, GHK_te)
    log(f"root increments D1: {D1_dev.shape} - Delta^(1) == PPV^(1) "
        f"exactly (PPV^(0) = {PPV0})")

    # ---- 5. MR-HIGH reproduction gate (regression assertion) -------------
    mr_gate, _ = ridge_eval(D1_dev, y_dev, D1_te, yte)
    gate_diff = abs(mr_gate["macro_f1"] - REF_MR_HIGH)
    log(f"[gate MR-HIGH reproduction] = {mr_gate['macro_f1']} "
        f"(stored {REF_MR_HIGH}, diff {gate_diff:.4f})")
    assert smoke or gate_diff <= GATE_TOL, gate_diff

    # ---- 6. deeper telescoping pool (train+val) --------------------------
    reg16_dev = np.vstack([reg16_tr, reg16_va])
    X_dev = np.vstack([Xtr_z, Xva_z])
    pool_parts = []
    sig_chunk = 32
    for c0 in range(0, n_tr + n_va, sig_chunk):
        c1 = min(c0 + sig_chunk, n_tr + n_va)
        act, valid = compute_activations(exHK, X_dev[c0:c1])
        pool_parts.append(delta_banks(act[:, G_PART:], valid[G_PART:],
                                      reg16_dev[c0:c1], KEPT_K,
                                      delta_cols=None, sample_chunk=96))
        del act
    Hcand_trva = np.vstack(pool_parts)
    del pool_parts
    assert Hcand_trva.shape == (n_tr + n_va, H_CAND), Hcand_trva.shape
    assert [col_meta(c, F_H)[0] for c in
            (0, 2 * F_H, 6 * F_H, 14 * F_H)] == [2, 4, 8, 16]
    log(f"deeper telescoping pool: {Hcand_trva.shape} "
        f"({EDGES} edges x {F_H} kernels)")

    # ---- 7. UNIFIED telescoping pool + LAYOUT GUARD ----------------------
    C_dev = np.hstack([D1_dev, Hcand_trva])
    del D1_dev, Hcand_trva
    assert C_dev.shape == (n_tr + n_va, N_CAND), C_dev.shape
    assert np.isfinite(C_dev).all()
    level_of_col = np.full(N_CAND, -1, dtype=np.int64)
    level_of_col[:D1] = 1
    for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
        level_of_col[D1 + cbefore * F_H:D1 + (cbefore + K) * F_H] = K
    assert (level_of_col == 1).sum() == D1
    for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
        assert (level_of_col[D1:] == K).sum() == K * F_H
    log(f"unified pool: {C_dev.shape} (D1 {D1} + deltas {H_CAND} "
        f"= {N_CAND}); layout guard OK [D1|D2|D4|D8|D16]")

    # per-sample level means for the occupancy figures (before C_dev freed)
    pslm = per_sample_level_means(C_dev, level_of_col)

    # ---- 8. ANOVA pieces (design locked for EVERY candidate) -------------
    n_dev = n_tr + n_va
    d_e = n_dev - N_CLASSES
    assert DF_BETWEEN == 7 and d_e == n_dev - 8
    ssb, sse = anova_all(C_dev, y_dev)
    msb = ssb / DF_BETWEEN
    s2m = sse / d_e
    from sklearn.feature_selection import f_classif
    f_ref, _ = f_classif(C_dev, y_dev)
    f_mine = msb / s2m
    ok = np.isfinite(f_ref)
    agree = np.allclose(f_mine[ok], f_ref[ok], rtol=1e-7, atol=1e-12)
    f_gate_chk = raw_f_gate_convention(f_mine)
    nan_handled = np.array_equal(
        f_gate_chk[~ok], np.nan_to_num(f_ref[~ok], nan=0.0))
    assert agree and nan_handled
    log(f"ANOVA pieces validated against f_classif: {int(ok.sum())} finite "
        f"exact, {int((~ok).sum())} constant handled identically "
        f"(d_between={DF_BETWEEN}, d_error={d_e})")

    # ---- 9. RAW-F EQUIVALENCE GATE (spec 28) -----------------------------
    f_gate = raw_f_gate_convention(f_mine)
    sel_raw = rank_select(f_gate, B_HIGH)
    meta_raw = []
    for c in sel_raw:
        if c < D1:
            meta_raw.append((1, -1, -1, int(c)))
        else:
            K, j, p, kg = col_meta(int(c) - D1, F_H)
            meta_raw.append((K, j, p, kg))
    lv_counts_raw = {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    for (K, j, p, kg) in meta_raw:
        lv_counts_raw[K] += 1
    if smoke:
        eq_gate = {"verdict": "SKIPPED_SMOKE",
                   "reason": "smoke subset differs from the stored full run"}
        log("raw-F equivalence gate: SKIPPED (smoke)")
    else:
        import csv as _csv
        stored_rows = list(_csv.DictReader(open(
            os.path.join(CLEAN_DIR, "diagnostics",
                         "selected_feature_metadata.csv"))))
        stored_seq = [int(r["candidate_index"]) for r in stored_rows]
        stored_map = {}
        for r in stored_rows:
            lv = 1 if r["source_type"] == "root_increment" \
                else int(r["level"])
            stored_map[int(r["candidate_index"])] = (
                lv, int(r["parent"]), int(r["child"]),
                int(r["kernel_id"]), float(r["f_stat"]))
        my_map = {int(c): (1 if K == 1 else K, p, j, kg,
                           float(f_gate[c]) if np.isfinite(f_mine[c])
                           else 0.0)
                  for c, (K, j, p, kg) in zip(sel_raw, meta_raw)}
        set_eq = sorted(my_map) == sorted(stored_map)
        seq_eq = sel_raw.tolist() == stored_seq
        f_eq = set_eq and all(
            np.isclose(my_map[c][4], stored_map[c][4], rtol=1e-5, atol=0.0)
            for c in stored_map if np.isfinite(f_mine[c]))
        sem_eq = set_eq and all(
            my_map[c][:4] == stored_map[c][:4] for c in stored_map)
        stored_json = json.load(open(os.path.join(
            CLEAN_DIR, "results", "hier_sup_clean.json")))
        comp_ref = {int(k): v for k, v
                    in stored_json["selected_composition"].items()}
        comp_eq = lv_counts_raw == comp_ref
        # rebuild the raw-selection representation exactly as HIER-SUP-CLEAN
        order_raw = np.argsort([(-1 if m[0] == 1 else
                                 {2: 0, 4: 1, 8: 2, 16: 3}[m[0]])
                                for m in meta_raw], kind="stable")
        sel_raw_o = sel_raw[order_raw]
        meta_raw_o = [meta_raw[i] for i in order_raw]
        lv_seq = [m[0] for m in meta_raw_o if m[0] != 1]
        assert lv_seq == sorted(lv_seq)
        root_idx_r = np.array([c for c in sel_raw_o if c < D1],
                              dtype=np.int64)
        n_root_r = len(root_idx_r)
        sel_dict_r = {K: np.array([(kg - G_PART, j)
                                   for (Kk, j, p, kg) in meta_raw_o
                                   if Kk == K], dtype=np.int64)
                      for K in KEPT_K}
        Hcand_te_r = hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K,
                                     sel_dict_r)
        Z_te_r = np.empty((n_te, B_HIGH), dtype=np.float64)
        Z_te_r[:, :n_root_r] = D1_te[:, root_idx_r]
        Z_te_r[:, n_root_r:] = Hcand_te_r
        raw_ridge, pred_raw = ridge_eval(C_dev[:, sel_raw_o], y_dev,
                                         Z_te_r, yte)
        stored_pred = np.load(os.path.join(
            CLEAN_DIR, "predictions", "test_predictions.npy"))
        pred_eq = np.array_equal(pred_raw.astype(np.int64), stored_pred)
        f1_eq = abs(raw_ridge["macro_f1"]
                    - stored_json["macro_f1"]) < 1e-9
        alpha_eq = abs(raw_ridge["selected_alpha"]
                       - stored_json["alpha"]) < 1e-9
        del Hcand_te_r, Z_te_r
        eq_gate = {
            "expected": "raw ranking reproduces HIER-SUP-CLEAN",
            "verdict": "PASS" if all([set_eq, f_eq, sem_eq, comp_eq,
                                      pred_eq, alpha_eq, f1_eq])
                       else "FAIL",
            "reference": os.path.relpath(CLEAN_DIR, ROOT),
            "components": {
                "selected_index_set": set_eq,
                "selected_index_sequence": seq_eq,
                "per_candidate_f_stats": f_eq,
                "semantic_metadata": sem_eq,
                "emerged_composition": comp_eq,
                "test_predictions": pred_eq,
                "alpha": alpha_eq, "macro_f1": f1_eq},
            "raw_reproduction_macro_f1": raw_ridge["macro_f1"],
            "n_selected_compared": len(stored_map)}
        log(f"[raw-F equivalence gate vs HIER-SUP-CLEAN] verdict = "
            f"{eq_gate['verdict']}")
        assert eq_gate["verdict"] == "PASS", eq_gate

    # ---- 10. CV diagnostic (fold-internal prior + moderation) ------------
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    for tr_i, va_i in skf.split(C_dev, y_dev):
        ssb_f, sse_f = anova_all(C_dev[tr_i], y_dev[tr_i])
        s2_f = (sse_f / d_e)[np.isfinite(sse_f) & (sse_f > 0)]
        d0_f, s0sq_f, _ = eb_prior_fit(s2_f, d_e)
        _, _, _, f_modv = moderated_scores(ssb_f, sse_f, d_e, d0_f, s0sq_f)
        sel_f = rank_select(f_modv, B_HIGH)
        clf = RidgeClassifierCV(alphas=ALPHAS)
        clf.fit(C_dev[tr_i][:, sel_f], y_dev[tr_i])
        cv_scores.append(f1_score(y_dev[va_i],
                                  clf.predict(C_dev[va_i][:, sel_f]),
                                  average="macro"))
    cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))
    log(f"[CV moderated fold-internal] macro-F1 = {cv_mean:.4f} "
        f"+/- {cv_std:.4f}  (diagnostic only; prior fit on fold-train "
        f"labels only)")

    # ---- 11. EB prior + moderated ranking (THE ONE CHANGE) ---------------
    s2_pos = s2m[np.isfinite(s2m) & (s2m > 0)]
    n_bad_s2 = int(s2m.size - s2_pos.size)
    d0, s0sq, prior_info = eb_prior_fit(s2_pos, d_e)
    log(f"EB prior: d0 = {d0:.6f}, s0^2 = {s0sq:.6e}  "
        f"(fit on {s2_pos.size} positive variances; {n_bad_s2} "
        f"non-positive handled per spec 13)")
    s2_all, st2, f_raw, f_mod = moderated_scores(ssb, sse, d_e, d0, s0sq)
    sel_mod = rank_select(f_mod, B_HIGH)
    meta_mod = []
    for c in sel_mod:
        if c < D1:
            meta_mod.append((1, -1, -1, int(c)))
        else:
            K, j, p, kg = col_meta(int(c) - D1, F_H)
            meta_mod.append((K, j, p, kg))
    lv_counts = {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    for (K, j, p, kg) in meta_mod:
        lv_counts[K] += 1
    assert sum(lv_counts.values()) == B_HIGH
    log(f"moderated selection composition (EMERGED, diagnostic only): "
        f"D1 {lv_counts[1]} ({lv_counts[1] / B_HIGH:.1%}), "
        f"D2/K2 {lv_counts[2]} ({lv_counts[2] / B_HIGH:.1%}), "
        f"D4/K4 {lv_counts[4]} ({lv_counts[4] / B_HIGH:.1%}), "
        f"D8/K8 {lv_counts[8]} ({lv_counts[8] / B_HIGH:.1%}), "
        f"D16/K16 {lv_counts[16]} ({lv_counts[16] / B_HIGH:.1%})")

    # shared column order: D1 first, then deltas level-grouped
    lv_rank = {1: -1, 2: 0, 4: 1, 8: 2, 16: 3}
    order = np.argsort([lv_rank[m[0]] for m in meta_mod], kind="stable")
    sel_mod = sel_mod[order]
    meta_mod = [meta_mod[i] for i in order]
    lv_seq = [m[0] for m in meta_mod if m[0] != 1]
    assert lv_seq == sorted(lv_seq), "shared order not level-grouped"
    sel_mod_mask = np.zeros(N_CAND, dtype=bool)
    sel_mod_mask[sel_mod] = True

    with open(os.path.join(OUT, "diagnostics",
                           "selected_counts_by_level.csv"), "w") as f:
        f.write("level,source_type,pool_columns,selected,"
                "share_of_final,share_of_pool\n")
        f.write(f"1,root_increment,{D1},{lv_counts[1]},"
                f"{lv_counts[1] / B_HIGH:.6f},{lv_counts[1] / D1:.6f}\n")
        for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
            f.write(f"{K},hierarchy_delta,{K * F_H},{lv_counts[K]},"
                    f"{lv_counts[K] / B_HIGH:.6f},"
                    f"{lv_counts[K] / (K * F_H):.6f}\n")

    with open(os.path.join(OUT, "diagnostics",
                           "selected_feature_metadata.csv"), "w") as f:
        f.write("selected_rank,candidate_index,source_type,level,parent,"
                "child,kernel_id,F_raw,F_moderated,s2_raw,s2_moderated\n")
        for rank, (c, m) in enumerate(zip(sel_mod, meta_mod)):
            K, j, p, kg = m
            if K == 1:
                f.write(f"{rank},{c},root_increment,1,-1,-1,{kg},"
                        f"{f_raw[c]:.6g},{f_mod[c]:.6g},{s2_all[c]:.6g},"
                        f"{st2[c]:.6g}\n")
            else:
                f.write(f"{rank},{c},hierarchy_delta,{K},{p},{j},{kg},"
                        f"{f_raw[c]:.6g},{f_mod[c]:.6g},{s2_all[c]:.6g},"
                        f"{st2[c]:.6g}\n")

    # ---- 12. final representation + canonical Ridge ----------------------
    root_idx = np.array([c for c in sel_mod if c < D1], dtype=np.int64)
    n_root_sel = len(root_idx)
    n_delta_sel = B_HIGH - n_root_sel
    assert n_root_sel + n_delta_sel == B_HIGH
    Z_dev = C_dev[:, sel_mod]
    del C_dev
    assert Z_dev.shape == (n_dev, B_HIGH)
    sel_dict = {K: np.array([(kg - G_PART, j)
                             for (Kk, j, p, kg) in meta_mod
                             if Kk == K], dtype=np.int64)
                for K in KEPT_K}
    Hcand_te = hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)
    assert Hcand_te.shape == (n_te, n_delta_sel), Hcand_te.shape
    Z_te = np.empty((n_te, B_HIGH), dtype=np.float64)
    Z_te[:, :n_root_sel] = D1_te[:, root_idx]
    Z_te[:, n_root_sel:] = Hcand_te
    del D1_te, Hcand_te, GHK_te
    assert np.isfinite(Z_dev).all() and np.isfinite(Z_te).all()
    log(f"final representation: {Z_dev.shape} / test {Z_te.shape} "
        f"(shared order: {n_root_sel} root increments + "
        f"{n_delta_sel} deltas)")

    res, pred = ridge_eval(Z_dev, y_dev, Z_te, yte)
    log(f"[HIER-SUP-MOD] = {res['macro_f1']}  "
        f"(alpha={res['selected_alpha']})")

    np.save(os.path.join(OUT, "predictions", "test_predictions.npy"),
            pred.astype(np.int64))
    deltas = {k: round(res["macro_f1"] - v, 4) for k, v in REFS.items()}

    # ---- 13. diagnostics --------------------------------------------------
    var_rows = summarize_variance_stats(level_of_col, s2_all, st2, f_raw,
                                        f_mod, sel_mod_mask)
    with open(os.path.join(OUT, "diagnostics",
                           "variance_statistics.csv"), "w") as f:
        f.write("level,candidate_count,median_raw_variance,"
                "median_moderated_variance,median_raw_F,"
                "median_moderated_F,median_F_ratio,selected_count\n")
        for r in var_rows:
            f.write(",".join(str(r[k]) for k in
                             ("level", "candidate_count",
                              "median_raw_variance",
                              "median_moderated_variance", "median_raw_F",
                              "median_moderated_F", "median_F_ratio",
                              "selected_count")) + "\n")

    with open(os.path.join(OUT, "diagnostics", "variance_prior.json"),
              "w") as f:
        json.dump({"model": "s_j^2 ~ s0^2 * F(d_e, d0) (Smyth/limma)",
                   "fit_scope": "SINGLE prior over the FULL unified pool "
                                "(all levels borrow strength)",
                   "n_dev": n_dev, "n_classes": N_CLASSES,
                   "d_between": DF_BETWEEN, "d_error": d_e,
                   "n_nonpositive_s2": n_bad_s2,
                   "nonpositive_handling":
                       "excluded from the LOG-MOMENT fit only; kept in "
                       "moderation/ranking with st2 = (d0 s0^2)/(d0+d_e)",
                   **prior_info,
                   "moderated_df": d_e + d0}, f, indent=1)

    ov = overlap_stats(sel_raw, sel_mod, N_CAND)
    with open(os.path.join(OUT, "diagnostics",
                           "raw_vs_moderated_selection.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in ov.items():
            f.write(f"{k},{v}\n")
    log(f"raw vs moderated top-B: intersection {ov['intersection']}, "
        f"raw-only {ov['raw_only']}, moderated-only "
        f"{ov['moderated_only']}, Jaccard {ov['jaccard']:.4f}")

    fb_rows = fallback_diagnostics(reg16_dev)
    tiny = s2_all <= 1e-12
    near0 = (s2_all > 1e-12) & (s2_all <= 1e-8)
    with open(os.path.join(OUT, "diagnostics",
                           "fallback_statistics.csv"), "w") as f:
        f.write("level,regime,pool_candidates,fallback_fraction,"
                "samples_with_fallback,n_affected_candidates\n")
        for r in fb_rows:
            f.write(f"{r['level']},ALL,{r['n_affected_candidates']},"
                    f"{r['fallback_fraction']:.6f},"
                    f"{r['samples_with_fallback']},"
                    f"{r['n_affected_candidates']}\n")
        f.write(f"ALL,ALL,{N_CAND},NA,NA,{N_CAND}\n")
        f.write(f"deg_s2_le_1e-12,ALL,NA,NA,NA,{int(tiny.sum())}\n")
        f.write(f"near0_s2_1e-12_to_1e-8,ALL,NA,NA,NA,{int(near0.sum())}\n")
        if tiny.any() or near0.any():
            m = tiny | near0

            def _med(v):
                vv = v[m]
                vv = vv[np.isfinite(vv)]
                return float(np.median(vv)) if vv.size else float("nan")

            f.write(f"median_F_raw_degenerate,ALL,NA,NA,NA,"
                    f"{_med(f_raw):.6g}\n")
            f.write(f"median_F_moderated_degenerate,ALL,NA,NA,NA,"
                    f"{_med(f_mod):.6g}\n")
            f.write(f"selected_moderated_degenerate,ALL,NA,NA,NA,"
                    f"{int(sel_mod_mask[m].sum())}\n")
            f.write(f"selected_raw_degenerate,ALL,NA,NA,NA,"
                    f"{int(np.isin(np.arange(N_CAND), sel_raw)[m].sum())}\n")

    try:
        occ_rows, assoc_rows = occupancy_diagnostics(
            reg16_dev, y_dev, f_raw, f_mod, s2_all, st2, level_of_col)
        with open(os.path.join(OUT, "diagnostics",
                               "occupancy_variance.csv"), "w") as f:
            f.write("level,n_candidates,statistic,spearman_rho,p_value\n")
            for r in occ_rows:
                f.write(f"{r['level']},{r['n_candidates']},"
                        f"{r['statistic']},{r['spearman_rho']:.6g},"
                        f"{r['p_value']:.6g}\n")
        with open(os.path.join(OUT, "diagnostics",
                               "occupancy_class_association.csv"),
                  "w") as f:
            f.write("level,regime,occupancy_mean,F,p_value,p_bh_adj\n")
            for r in assoc_rows:
                f.write(f"{r['level']},{r['regime']},"
                        f"{r['occupancy_mean']:.6g},{r['F']:.6g},"
                        f"{r['p_value']:.6g},"
                        f"{r.get('p_bh_adj', 'NA')}\n")
        n_assoc_sig = sum(1 for r in assoc_rows if r["p_value"] < 0.05)
        n_assoc_bh = sum(1 for r in assoc_rows
                         if r.get("p_bh_adj", 1.0) < 0.05)
        log(f"occupancy-class association: {len(assoc_rows)} regimes "
            f"tested, {n_assoc_sig} p<0.05 raw, {n_assoc_bh} after BH")
    except Exception as e:
        log(f"occupancy diagnostics skipped: {e!r}")
        occ_rows, assoc_rows = [], []
        n_assoc_sig = n_assoc_bh = 0

    # ---- 14. artifacts ----------------------------------------------------
    with open(os.path.join(OUT, "results", "hier_sup_mod.json"), "w") as f:
        json.dump({
            "method": "hier_sup_mod", "label": "HIER-SUP-MOD",
            "dataset": DS, "seed": SEED,
            "base_budget": B_BASE, "high_budget": B_HIGH,
            "kernel_capacity": "84x238", "final_features": B_HIGH,
            "candidate_pool": {"root_D1": D1, "delta": H_CAND,
                               "unified": N_CAND},
            "changed_vs_hier_sup_clean":
                "ranking statistic only: raw ANOVA-F -> EB variance-"
                "moderated F (Smyth scaled-F prior, single pool-wide fit)",
            "anova_design": {"d_between": DF_BETWEEN, "d_error": d_e,
                             "occupancy_as_df": False},
            "eb_prior": {"d0": d0, "s0_sq": s0sq,
                         "moderated_df": d_e + d0,
                         "degenerate": prior_info.get("degenerate", False),
                         "n_nonpositive_s2": n_bad_s2},
            "selected_composition": {str(k): v
                                     for k, v in lv_counts.items()},
            "raw_vs_moderated_overlap": ov,
            "occupancy_class_association": {
                "regimes_tested": len(assoc_rows),
                "significant_raw": n_assoc_sig,
                "significant_bh": n_assoc_bh},
            "cv_moderated_fold_internal": {"mean": cv_mean, "std": cv_std,
                                           "kfold": 5},
            "equivalence_gate": eq_gate,
            "macro_f1": res["macro_f1"], "accuracy": res["accuracy"],
            "alpha": res["selected_alpha"], "deltas_vs": deltas,
            "gate_mr_high": {"ref": REF_MR_HIGH,
                             "reproduced": mr_gate["macro_f1"],
                             "diff": round(gate_diff, 4), "tol": GATE_TOL},
            "runtime_s": round(time.time() - t00, 1),
            "peak_mem_mb": round(peak_mem_mb(), 1), "smoke": smoke,
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics",
                           "equivalence_gate.json"), "w") as f:
        json.dump(eq_gate, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "dimensions.json"),
              "w") as f:
        json.dump({
            "split": {"train": n_tr, "validation": n_va, "test": n_te,
                      "T": int(Xtr.shape[1]),
                      "n_classes": int(len(np.unique(ytr)))},
            "extractors": {"expanded": fHK},
            "anova": {"d_between": DF_BETWEEN, "d_error": d_e,
                      "n_dev": n_dev},
            "pools": {"root_D1": D1, "delta": H_CAND, "edges": EDGES,
                      "kernels_per_edge": F_H, "unified": N_CAND,
                      "column_order": ["D1", "D2", "D4", "D8", "D16"]},
            "prior": {"d0": d0, "s0_sq": s0sq, "moderated_df": d_e + d0},
            "final": {"selected": B_HIGH, "Z_dev": list(Z_dev.shape),
                      "Z_test": list(Z_te.shape),
                      "composition": {str(k): v
                                      for k, v in lv_counts.items()}},
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "leakage_report.json"),
              "w") as f:
        json.dump({
            "minirocket_fit": "train signals only (seed 42)",
            "hierarchy": "TRAIN latents only (frozen seed-42 SSL encoder, "
                         "recursive 2-means); val/test transform-only; "
                         "reg16_tr byte-identical to the stored artifact",
            "anova_and_prior": "train+val labels for the FINAL stage "
                               "(canonical R5 protocol); fold-internal "
                               "prior fit on fold-train labels ONLY",
            "no_test_statistics": "test variances/residuals NEVER used for "
                                  "d0, s0^2, or selection (spec 25/26)",
            "test_labels": "touched only inside ridge_eval for scoring",
            "occupancy_usage": "diagnostics ONLY - never df, weight, "
                               "quota, or filter",
            "fallback_usage": "diagnostics ONLY - no candidate filtered",
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics",
                           "selection_protocol.md"), "w") as f:
        f.write(
            "# Selection protocol - HIER-SUP-MOD\n\n"
            "## Pool (locked, identical to HIER-SUP-CLEAN)\n\n"
            f"C = [D1 | D2 | D4 | D8 | D16] with {D1} root increments "
            f"(Delta^(1) = PPV^(1), PPV^(0) = 0) + {H_CAND} occupancy-"
            f"weighted telescoping deltas = {N_CAND} candidates. One "
            "shared semantic column order for train/val/test (hard "
            "layout guard).\n\n"
            "## Stage 1 - CV diagnostic (never selects)\n\n"
            "5-fold StratifiedKFold(shuffle=True, random_state=42). Per "
            "fold: ANOVA pieces on fold-train rows/labels ONLY -> EB "
            "prior (d0, s0^2) fit on FOLD-TRAIN variances ONLY -> "
            "moderated F -> top-19,992 -> RidgeClassifierCV(ALPHAS) on "
            "fold-train -> macro-F1 on fold-val. Diagnostic only.\n\n"
            "## Stage 2 - final production selection (canonical boundary)\n\n"
            f"ANOVA design on train+val ({n_dev} rows): d_between = 7, "
            f"d_error = {d_e} for EVERY candidate. s_j^2 = SSE_j/"
            f"{d_e}; SINGLE scaled-F EB prior over the full pool: "
            f"d0 = {d0:.6f}, s0^2 = {s0sq:.6e} (Smyth log-moment "
            "matching; no validation tuning, no grid search). Posterior "
            "s_t^2 = (d0 s0^2 + d_e s^2)/(d0 + d_e); ranking F_t = "
            "MS_between/s_t^2 descending, ties by candidate_index; top "
            "19,992. NO raw-F mixing, no FDR threshold, no quotas, no "
            "occupancy in the statistic.\n\n"
            "## Stage 3 - refit + one official test evaluation\n\n"
            "d0, s0^2 and the selected identities are FROZEN before the "
            "test transform; RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) "
            f"on the selected {B_HIGH}-column representation over "
            "train+val; single evaluation on the 3,582-row official test "
            "split.\n\n"
            f"## Emerged composition (diagnostic only)\n\nD1 "
            f"{lv_counts[1]}, D2/K2 {lv_counts[2]}, D4/K4 "
            f"{lv_counts[4]}, D8/K8 {lv_counts[8]}, D16/K16 "
            f"{lv_counts[16]} (sum {sum(lv_counts.values())} = "
            f"{B_HIGH}).\n\n"
            f"Raw vs moderated top-B overlap: intersection "
            f"{ov['intersection']}/{B_HIGH}, Jaccard "
            f"{ov['jaccard']:.4f} (raw-only {ov['raw_only']}, "
            f"moderated-only {ov['moderated_only']}).\n")

    # comparison.csv + figures + summary ------------------------------------
    rows = [("canonical_mr", "Canonical MR", 9996, "84x119", "NA"),
            ("mr_high", "MR-HIGH", 19992, "84x238", "NA"),
            ("r5_high", "R5-HIGH", 19992, "84x238", "14994"),
            ("hier_high", "HIER-HIGH", 19992, "84x238", "449820"),
            ("hier_high_sup", "HIER-HIGH-SUP", 19992, "84x238", "449820"),
            ("hier_continuous_sup", "HIER-CONTINUOUS-SUP", 19992, "84x238",
             "469812"),
            ("hier_sup_clean", "HIER-SUP-CLEAN", 19992, "84x238", "469812"),
            ("hier_sup_mod", "HIER-SUP-MOD", 19992, "84x238", "469812")]
    with open(os.path.join(OUT, "results", "comparison.csv"), "w") as f:
        f.write("method,label,final_features,d1_selected,k2_selected,"
                "k4_selected,k8_selected,k16_selected,candidate_pool,"
                "kernel_capacity,macro_f1,accuracy,alpha,runtime_s,"
                "peak_mem_mb\n")
        for key, label, ff, kcap, pool in rows:
            if key == "hier_sup_mod":
                f.write(f"{key},{label},{ff},{lv_counts[1]},{lv_counts[2]},"
                        f"{lv_counts[4]},{lv_counts[8]},{lv_counts[16]},"
                        f"{pool},{kcap},{res['macro_f1']},{res['accuracy']},"
                        f"{res['selected_alpha']},"
                        f"{round(time.time() - t00, 1)},"
                        f"{round(peak_mem_mb(), 1)}\n")
            else:
                f.write(f"{key},{label},{ff},NA,NA,NA,NA,NA,{pool},{kcap},"
                        f"{REFS[key]},NA,NA,NA,NA\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _fig(name):
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "figures", name), dpi=150)
            plt.close(fig)

        # 1. comparison
        labels = [r[1] for r in rows]
        vals = [REFS.get(r[0]) if r[0] in REFS else res["macro_f1"]
                for r in rows]
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        ax.bar(labels, vals, color=["#9e9e9e"] * 7 + ["#1f77b4"])
        ax.set_ylim(0.70, 0.80)
        ax.set_ylabel("Macro-F1 (official test)")
        ax.set_title("HIER-SUP-MOD - EB-moderated F vs incumbents "
                     "(UWaveY, seed 42)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        _fig("comparison_macro_f1.png")

        # 2. raw vs moderated F (log-log scatter, subsampled)
        rng = np.random.default_rng(0)
        idx = rng.choice(N_CAND, size=min(60000, N_CAND), replace=False)
        fin = np.isfinite(f_raw[idx]) & np.isfinite(f_mod[idx])
        xr, yr = np.maximum(f_raw[idx][fin], 1e-300), f_mod[idx][fin]
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        ax.scatter(xr, yr, s=3, alpha=0.25, c="#1f77b4")
        top = max(float(xr.max()), float(yr.max())) * 1.2 if xr.size else 1.0
        lim = (1e-3, top)
        ax.plot(lim, lim, "r--", lw=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("raw F"); ax.set_ylabel("moderated F")
        ax.set_title("Raw vs moderated F (subsample)")
        ax.legend()
        _fig("raw_vs_moderated_F.png")

        # 3. raw vs moderated variance
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        xs = np.maximum(s2_all[idx], 1e-300)
        ax.scatter(xs, st2[idx], s=3, alpha=0.25, c="#2ca02c")
        lo = max(float(xs.min()), 1e-300)
        hi = max(float(xs.max()), 1e-300)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("raw s^2"); ax.set_ylabel("moderated s^2")
        ax.set_title(f"Variance shrinkage toward s0^2 = {s0sq:.3e} "
                     f"(d0 = {d0:.2f})")
        ax.legend()
        _fig("raw_vs_moderated_variance.png")

        # 4-6. occupancy vs per-sample level statistics
        for K, fname, lbl in (
                (2, "occupancy_vs_residual_variance.png",
                 "mean raw s^2 (log)"),
                (4, "occupancy_vs_raw_F.png", "mean raw F"),
                (8, "occupancy_vs_moderated_F.png", "mean moderated F")):
            coarse = reg16_dev >> (4 - KEPT_K.index(K))
            cnt = np.stack([(coarse == c).sum(axis=1) for c in range(K)],
                           axis=1).astype(np.float64)
            frac = cnt / T_LEN
            selm, _ = select_and_mask_min(cnt.astype(np.int64),
                                          FALLBACK_MIN_COUNT)
            o = np.where(selm, frac, np.inf).min(axis=1)
            v = pslm[K]
            v = np.log(np.maximum(v, 1e-300)) if "variance" in fname else v
            fig, ax = plt.subplots(figsize=(5.6, 4.4))
            ax.scatter(o, v, s=12, alpha=0.5, c="#ff7f0e")
            ax.set_xlabel("min branch occupancy fraction (per sample)")
            ax.set_ylabel(lbl)
            ax.set_title(f"K={K}: occupancy vs {lbl} (per sample)")
            _fig(fname)

        # 7. selected features by level
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        lv = ["D1 (root)", "D2/K2", "D4/K4", "D8/K8", "D16/K16"]
        cnt = [lv_counts[1], lv_counts[2], lv_counts[4], lv_counts[8],
               lv_counts[16]]
        ax.bar(lv, cnt, color="#1f77b4")
        ax.set_ylabel("selected features (of 19,992)")
        ax.set_title("Composition EMERGED from the moderated ranking "
                     "(no quotas)")
        for i, v in enumerate(cnt):
            ax.text(i, v + 30, f"{v}\n({v / B_HIGH:.1%})", ha="center",
                    fontsize=8)
        _fig("selected_features_by_level.png")

        # 8. fallback fraction by level
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        flv = [f"K={r['level']}" for r in fb_rows]
        ffb = [r["fallback_fraction"] for r in fb_rows]
        ax.bar(flv, ffb, color="#d62728")
        ax.set_ylabel("per-sample fallback fraction")
        ax.set_title(f"Fallback rule (count < {FALLBACK_MIN_COUNT} -> "
                     "argmax regime) - DIAGNOSTIC ONLY")
        for i, v in enumerate(ffb):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
        _fig("fallback_fraction_by_level.png")
        log("figures written (8)")
    except Exception as e:  # figures must never kill the science
        log(f"figure generation skipped: {e!r}")

    # summary ---------------------------------------------------------------
    rt = round(time.time() - t00, 1)
    mem = round(peak_mem_mb(), 1)
    with open(os.path.join(OUT, "results", "summary.md"), "w") as f:
        f.write(
            "# HIER-SUP-MOD - summary (UWaveY, seed 42)\n\n"
            f"**Macro-F1 = {res['macro_f1']}** (accuracy "
            f"{res['accuracy']}, alpha {res['selected_alpha']})\n\n"
            "## The one change\n\n"
            "Raw ANOVA-F ranking -> EB variance-moderated F ranking on the "
            "SAME unified 469,812-candidate telescoping pool: per-candidate "
            f"s^2 = SSE/(N-8), SINGLE Smyth scaled-F prior (d0 = {d0:.4f}, "
            f"s0^2 = {s0sq:.4e}), s_t^2 = (d0 s0^2 + d_e s^2)/(d0 + d_e), "
            "F_t = MS_between/s_t^2, one global top-19,992. Used "
            "nowhere (no rho, no G/H, no quotas, no local-fdr, "
            "no occupancy-as-df, no fallback filter).\n\n"
            "## Comparison (saved references; only this model was run)\n\n"
            "| Method | Final | Macro-F1 | Delta vs this |\n|---|---|---|---|\n"
            "| Canonical MR | 9,996 | 0.7543 | "
            f"{round(res['macro_f1'] - REFS['canonical_mr'], 4):+} |\n"
            "| MR-HIGH | 19,992 | 0.7477 | "
            f"{round(res['macro_f1'] - REFS['mr_high'], 4):+} |\n"
            "| R5-HIGH | 19,992 | 0.7772 | "
            f"{round(res['macro_f1'] - REFS['r5_high'], 4):+} |\n"
            "| HIER-HIGH | 19,992 | 0.7230 | "
            f"{round(res['macro_f1'] - REFS['hier_high'], 4):+} |\n"
            "| HIER-HIGH-SUP | 19,992 | 0.7823 | "
            f"{round(res['macro_f1'] - REFS['hier_high_sup'], 4):+} |\n"
            "| HIER-CONTINUOUS-SUP | 19,992 | 0.7599 | "
            f"{round(res['macro_f1'] - REFS['hier_continuous_sup'], 4):+} "
            "|\n"
            "| HIER-SUP-CLEAN | 19,992 | 0.7599 | "
            f"{round(res['macro_f1'] - REFS['hier_sup_clean'], 4):+} |\n"
            f"| **HIER-SUP-MOD** | 19,992 | **{res['macro_f1']}** | NA |\n\n"
            f"## EB prior\n\nd0 = {d0:.6f}, s0^2 = {s0sq:.6e} (single "
            f"pool-wide fit; moderated df {d_e + d0}; {n_bad_s2} "
            "non-positive variances handled per spec 13; degenerate fit: "
            f"{prior_info.get('degenerate', False)}).\n\n"
            f"## Raw vs moderated selection\n\nintersection "
            f"{ov['intersection']}/{B_HIGH}, Jaccard {ov['jaccard']:.4f} "
            f"(raw-only {ov['raw_only']}, moderated-only "
            f"{ov['moderated_only']}; {ov['entering']} entering, "
            f"{ov['leaving']} leaving).\n\n"
            f"## Emerged composition (diagnostic)\n\nD1 {lv_counts[1]} "
            f"({lv_counts[1] / B_HIGH:.1%}), D2/K2 {lv_counts[2]} "
            f"({lv_counts[2] / B_HIGH:.1%}), D4/K4 {lv_counts[4]} "
            f"({lv_counts[4] / B_HIGH:.1%}), D8/K8 {lv_counts[8]} "
            f"({lv_counts[8] / B_HIGH:.1%}), D16/K16 {lv_counts[16]} "
            f"({lv_counts[16] / B_HIGH:.1%}).\n\n"
            f"CV (moderated, fold-internal): {cv_mean:.4f} +/- "
            f"{cv_std:.4f}. Raw-F equivalence gate: "
            f"{eq_gate['verdict']}.\n\n"
            f"Runtime {rt:.0f}s; peak process memory {mem} MB.\n")
    log(f"artifacts complete in {rt:.0f}s  OUT={OUT}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    r = main(smoke=a.smoke)
    print("DONE", r["macro_f1"])
