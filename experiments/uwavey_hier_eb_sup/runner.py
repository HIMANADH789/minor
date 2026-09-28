"""HIER-EB-SUP - empirical-Bayes unified selection (UWaveY, seed 42).

ONE substantive change vs HIER-CONTINUOUS-SUP: the raw ANOVA-F ranking is
replaced by an empirical-Bayes two-groups local-FDR ranking.  The candidate
statistic, the unified pool, the budget, the classifier and every other
component are byte-locked to the accepted experiments.

Pipeline (the ONE change):

    candidate feature
        -> ANOVA F statistic (sklearn f_classif - bit-identical to the
           incumbent top_f_select statistic)
        -> exact ANOVA p-value (same f_classif call, same df/class handling)
        -> z = Phi^{-1}(1 - p)   (one-sided significance coordinate,
                                  NOT an effect direction)
        -> FIVE independent parametric two-groups EB fits
           f_l(z) = pi0_l * N(0,1) + (1-pi0_l) * N(mu1_l, sigma1_l^2)
           (deterministic EM, fixed standard-normal null, mu1_l > 0)
        -> lfdr_l(z) -> posterior signal probability q = 1 - lfdr
        -> ONE global ranking by q (ties: candidate_index ascending)
        -> top 19,992 candidates
        -> canonical Ridge

No per-level quota, no rho, no N_G/N_H, no q multiplication of features,
no moderated-F.  sum(q) per level is an estimated effective signal count -
DIAGNOSTIC ONLY.  The EB model is the parametric two-groups model, NOT
Efron's nonparametric locfdr.

Everything else is imported from the verified helpers, not reimplemented.
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
H_CAND = EDGES * H_KERNELS       # 449,820
ROOTS = B_HIGH                   # 19,992 full root candidates (level 0)
N_CAND = ROOTS + H_CAND          # 469,812
ALPHAS = np.logspace(-4, 4, 20)

# EB hyperparameters - PREDECLARED, never tuned on validation (spec 12)
EB_INIT_PI0 = 0.90
EB_TAIL_FRAC = 0.10
EB_TOL = 1e-8
EB_MAX_ITER = 200
EB_PI0_MIN = 0.5
EB_PI0_MAX = 1.0 - 1e-8
EB_MU1_MIN = 1e-3
EB_SIGMA1_MIN = 1e-3
# machine-safe p clipping before Phi^{-1}: keeps z finite in [-8.22, 37.0]
P_CLIP_LO = 1e-300
P_CLIP_HI = 1.0 - 1e-16

CAP_RESULTS = os.path.join(ROOT, "results", "uwavey_capacity_scaled", "seed42")
NEST_RESULTS = os.path.join(ROOT, "results", "uwavey_nested_hierarchical",
                            "seed42")
REF_MR_HIGH = 0.7477
GATE_TOL = 0.002
REFS = {"canonical_mr": 0.7543, "mr_high": 0.7477, "r5_high": 0.7772,
        "hier_high": 0.7230, "hier_high_sup": 0.7823,
        "hier_continuous_sup": 0.7599}

OUT = os.path.join(ROOT, "results", "uwavey_hier_eb_sup", "seed42")

# reused canonical pieces (imported, not reimplemented)
from experiments.uwavey_nested_hierarchical.runner import (   # noqa: E402
    _encoder_latents, compute_activations, load_context_model, load_data,
    ppv_all, ridge_eval, znorm)
from experiments.uwavey_nested_hierarchical.runner import delta_banks  # noqa
from experiments.uwavey_capacity_scaled.runner import (       # noqa: E402
    extractor_facts, fit_minirocket_nk, hier_banks_test, peak_mem_mb,
    top_f_select)
from experiments.uwavey_hier_high_sup.runner import col_meta  # noqa: E402
from models.nested_regimes.model import (                     # noqa: E402
    assign_levels, build_latent_tree, nesting_errors)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Level structure of the unified pool (canonical level-grouped layout:
# [root | K2 | K4 | K8 | K16]) - asserted against col_meta before selection.
# --------------------------------------------------------------------------
def level_bounds():
    b = [(0, ROOTS, 0)]
    for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
        s = ROOTS + cbefore * H_KERNELS
        b.append((s, s + K * H_KERNELS, K))
    return b


def level_vector(width, n_root=ROOTS):
    """Level label per candidate column for a pool of `width` columns:
    the first n_root columns are roots (level 0); the remainder are delta
    candidates in the canonical level-grouped layout [K2 | K4 | K8 | K16].
    Production pools use the defaults (19,992 roots / 449,820 deltas);
    narrow synthetic pools (tests, n_root=width) are all-root."""
    lv = np.zeros(width, dtype=np.int64)
    n_d = width - n_root
    assert 0 < n_root <= width and n_d >= 0 and n_d % EDGES == 0, \
        (width, n_root)
    if n_d:
        f_h = n_d // EDGES
        for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
            s = n_root + cbefore * f_h
            lv[s:s + K * f_h] = K
    return lv


def level_of_candidates(f_h=H_KERNELS):
    lv = level_vector(N_CAND)
    covered = np.ones(N_CAND, dtype=bool)
    assert covered.all() and (lv == 0).sum() == ROOTS
    for s, e, K in level_bounds()[1:]:
        assert (lv[s:e] == K).all() and (e - s) == K * f_h
    return lv


# --------------------------------------------------------------------------
# The empirical-Bayes model (spec 11-13).  Deterministic EM, fixed N(0,1)
# null, single Gaussian alternative with mu1 > 0.  NOT Efron locfdr.
# --------------------------------------------------------------------------
def _degenerate_fit(n=0):
    """Recorded fallback for an EMPTY level (synthetic narrow pools only;
    every production level is populated).  Parameters respect the clamps.
    """
    return {"pi0": EB_PI0_MAX, "mu1": EB_MU1_MIN, "sigma1": EB_SIGMA1_MIN,
            "lfdr": np.empty(n, dtype=np.float64),
            "q": np.empty(n, dtype=np.float64), "iters": 0,
            "converged": True, "loglik": 0.0, "init_degenerate": True,
            "n": int(n), "final_delta": 0.0}


def eb_two_groups(z):
    """Fit f(z) = pi0*N(0,1) + (1-pi0)*N(mu1, sigma1^2) by deterministic EM.

    Returns pi0, mu1, sigma1, lfdr (clipped [0,1]), q = 1 - lfdr, EM book-
    keeping and the mixture log-likelihood.  No randomness anywhere.
    """
    from scipy.stats import norm
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    if n == 0:
        return _degenerate_fit(0)
    assert np.isfinite(z).all()

    # deterministic upper-tail initialization (spec 12)
    n_tail = max(1, int(round(EB_TAIL_FRAC * n)))
    tail = np.sort(z)[max(0, n - n_tail):]
    mu1 = float(np.mean(tail))
    s1 = float(np.std(tail))
    init_degenerate = not (np.isfinite(mu1) and np.isfinite(s1) and s1 > 0)
    if init_degenerate:
        mu1 = max(float(np.mean(z)), 0.5)
        s1 = max(float(np.std(z)), 0.5)
    mu1 = max(mu1, EB_MU1_MIN)
    s1 = max(s1, EB_SIGMA1_MIN)
    pi0 = EB_INIT_PI0

    f0 = norm.pdf(z)                       # fixed standard-normal null
    converged = False
    iters = 0
    delta = np.inf
    for it in range(EB_MAX_ITER):
        iters = it + 1
        f1 = norm.pdf(z, loc=mu1, scale=s1)
        num = (1.0 - pi0) * f1
        den = num + pi0 * f0
        safe = den > 0
        w = np.where(safe, num / np.where(safe, den, 1.0), 0.0)
        sw = float(w.sum())
        pi0_new = 1.0 - sw / n
        if sw > 0:
            mu1_new = float((w * z).sum() / sw)
            var1_new = float((w * (z - mu1_new) ** 2).sum() / sw)
        else:                              # all-null degenerate: keep alt
            mu1_new, var1_new = mu1, s1 * s1
        pi0_new = float(min(max(pi0_new, EB_PI0_MIN), EB_PI0_MAX))
        mu1_new = max(float(mu1_new), EB_MU1_MIN)
        s1_new = max(float(np.sqrt(max(var1_new, 0.0))), EB_SIGMA1_MIN)
        delta = max(abs(pi0_new - pi0), abs(mu1_new - mu1),
                    abs(s1_new - s1))
        pi0, mu1, s1 = pi0_new, mu1_new, s1_new
        if delta < EB_TOL:
            converged = True
            break

    f1 = norm.pdf(z, loc=mu1, scale=s1)
    den = pi0 * f0 + (1.0 - pi0) * f1
    safe = den > 0
    lfdr = np.where(safe, pi0 * f0 / np.where(safe, den, 1.0), 1.0)
    lfdr = np.clip(lfdr, 0.0, 1.0)
    q = 1.0 - lfdr
    loglik = float(np.log(np.maximum(den, 1e-300)).sum())
    return {"pi0": float(pi0), "mu1": float(mu1), "sigma1": float(s1),
            "lfdr": lfdr, "q": q, "iters": int(iters),
            "converged": bool(converged), "loglik": loglik,
            "init_degenerate": bool(init_degenerate), "n": int(n),
            "final_delta": float(delta)}


# --------------------------------------------------------------------------
# Shared EB scoring path: used by the CV diagnostic (fold-train rows/labels
# ONLY) and by the final production selection (full dev set).  One code path.
# --------------------------------------------------------------------------
def eb_q_scores(C, y, n_root=ROOTS):
    """F, p, z, q (posterior signal probability) and per-level EB params.

    Statistic: sklearn f_classif - the SAME call the incumbent selector
    makes, so F is bit-identical to the HIER-CONTINUOUS-SUP statistic.
    Edge cases (no candidate is discarded): non-finite F or p (constant
    features etc.) -> F=0, p=1; p clipped to [P_CLIP_LO, P_CLIP_HI] before
    inversion; z = norm.isf(p) is then finite in [-8.22, 37.0]; p=1 (null/
    degenerate) maps to z = -8.22, the most-null coordinate.
    """
    from scipy.stats import norm
    from sklearn.feature_selection import f_classif
    F, p = f_classif(C, y)
    F = np.asarray(F, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    bad = ~(np.isfinite(F) & np.isfinite(p))
    F = np.where(bad, 0.0, F)
    p = np.where(bad, 1.0, p)
    p = np.clip(p, P_CLIP_LO, P_CLIP_HI)
    z = norm.isf(p)
    z = np.where(np.isfinite(z), z, float(norm.isf(P_CLIP_HI)))
    lv = level_vector(C.shape[1], n_root)
    q = np.empty(C.shape[1], dtype=np.float64)
    params = {}
    for K in (0, 2, 4, 8, 16):            # FIVE independent EB fits
        m = lv == K
        if not m.any():                    # empty level: recorded fallback
            params[K] = _degenerate_fit(0)
            continue
        fit = eb_two_groups(z[m])
        params[K] = fit
        q[m] = fit["q"]
    assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all()
    return F, p, z, q, params


def eb_select(q, b):
    """ONE global ranking: highest q first, ties by candidate_index asc."""
    cand = np.arange(len(q), dtype=np.int64)
    return np.lexsort((cand, -q))[:b]


def eb_cv_unified(C_dev, y_dev, b_high, kfold=5):
    """Fold-internal EB diagnostic CV (spec 18).

    Per fold: ANOVA F/p on fold-train rows and labels ONLY -> z -> five
    fold-train EB fits -> q -> ONE global top-b selection -> Ridge on
    fold-train -> macro-F1 on the fold.  Fold-validation labels never
    touch the EB fit or the selection.  Diagnostic only.
    """
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(np.zeros(len(y_dev)), y_dev):
        _, _, _, q_f, _ = eb_q_scores(C_dev[tr], y_dev[tr])
        top = eb_select(q_f, b_high)
        ridge = RidgeClassifierCV(alphas=ALPHAS)
        ridge.fit(C_dev[np.ix_(tr, top)], y_dev[tr])
        pred = ridge.predict(C_dev[np.ix_(va, top)])
        scores.append(float(f1_score(y_dev[va], pred, average="macro",
                                     zero_division=0)))
    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------
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
            "hierarchy differs from the stored HIER-HIGH artifact"
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
        assert fHK["total_features"] == ROOTS and F_H == H_KERNELS \
            and 30 * F_H == H_CAND
        log(f"expanded extractor facts match stored: {fHK}")
    else:
        log(f"expanded extractor facts (smoke): {fHK}")

    # ---- 4. FULL root matrix (level 0, no preselection) -------------------
    GHK_trva = ppv_all(exHK, np.vstack([Xtr_z, Xva_z]))
    GHK_te = ppv_all(exHK, Xte_z)
    assert GHK_trva.shape == (n_tr + n_va, ROOTS)
    assert GHK_te.shape == (n_te, ROOTS)
    log(f"root matrix (level 0): {GHK_trva.shape} - full width")

    # ---- 5. MR-HIGH reproduction gate (regression assertion) -------------
    mr_gate, _ = ridge_eval(GHK_trva, y_dev, GHK_te, yte)
    gate_diff = abs(mr_gate["macro_f1"] - REF_MR_HIGH)
    log(f"[gate MR-HIGH reproduction] = {mr_gate['macro_f1']} "
        f"(stored {REF_MR_HIGH}, diff {gate_diff:.4f})")
    assert smoke or gate_diff <= GATE_TOL, gate_diff

    # ---- 6. FULL hierarchical candidate pool, train+val ------------------
    pool_parts = []
    sig_chunk = 32
    for c0 in range(0, n_tr + n_va, sig_chunk):
        c1 = min(c0 + sig_chunk, n_tr + n_va)
        act, valid = compute_activations(
            exHK, np.vstack([Xtr_z, Xva_z])[c0:c1])
        pool_parts.append(delta_banks(act[:, G_PART:], valid[G_PART:],
                                      np.vstack([reg16_tr, reg16_va])[c0:c1],
                                      KEPT_K, delta_cols=None,
                                      sample_chunk=96))
        del act
    Hcand_trva = np.vstack(pool_parts)
    del pool_parts
    assert Hcand_trva.shape == (n_tr + n_va, H_CAND), Hcand_trva.shape
    log(f"hierarchical candidate pool: {Hcand_trva.shape} "
        f"({EDGES} edges x {F_H} kernels)")

    # ---- 7. UNIFIED candidate pool (LOCKED layout) ------------------------
    C_dev = np.hstack([GHK_trva, Hcand_trva])
    del GHK_trva, Hcand_trva
    assert C_dev.shape == (n_tr + n_va, N_CAND), C_dev.shape
    assert np.isfinite(C_dev).all()
    lv = level_of_candidates(F_H)
    # layout assertion: level blocks agree with col_meta on probe columns
    for s, e, K in level_bounds()[1:]:
        for c in (s, e - 1):
            Kc, j, pp, kg = col_meta(int(c) - ROOTS, F_H)
            assert Kc == K and lv[c] == K, (c, Kc, K)
    log(f"unified pool: {C_dev.shape} "
        f"(root {ROOTS} + delta {H_CAND} = {N_CAND}); layout guard passed")

    # ---- 8. CV diagnostic (fold-internal EB, reported only) --------------
    cv_mean, cv_std = eb_cv_unified(C_dev, y_dev, B_HIGH)
    log(f"[CV unified fold-internal EB] macro-F1 = {cv_mean:.4f} "
        f"+/- {cv_std:.4f}  (diagnostic only)")

    # ---- 9. THE ONE CHANGE: EB scoring + ONE global ranking ---------------
    F_stat, p_vals, z_vals, q_scores, eb_params = eb_q_scores(C_dev, y_dev)
    assert np.isfinite(z_vals).all()
    assert (q_scores >= 0).all() and (q_scores <= 1).all()
    sel = eb_select(q_scores, B_HIGH)
    assert len(sel) == B_HIGH
    log(f"EB unified selection: top {B_HIGH} of {N_CAND} by posterior "
        f"signal probability q (ONE global ranking, ties by index; "
        f"no quota)")

    # EB sanity: refit determinism + finite likelihood (spec 20 item 6/8)
    refit_ok = True
    for K in (0, 2, 4, 8, 16):
        m = lv == K
        fit2 = eb_two_groups(z_vals[m])
        refit_ok &= (abs(fit2["pi0"] - eb_params[K]["pi0"]) == 0.0 and
                     abs(fit2["mu1"] - eb_params[K]["mu1"]) == 0.0 and
                     abs(fit2["sigma1"] - eb_params[K]["sigma1"]) == 0.0)
    log(f"EB refit determinism (in-process): {refit_ok}")

    # ---- 10. selected-feature metadata (diagnostic ONLY) ------------------
    meta = []
    for c in sel:
        if c < ROOTS:
            meta.append((0, -1, -1, int(c)))            # root candidate
        else:
            K, j, p, kg = col_meta(int(c) - ROOTS, F_H)
            meta.append((K, j, p, kg))
    lv_counts = {0: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    for m in meta:
        lv_counts[m[0]] += 1
    assert sum(lv_counts.values()) == B_HIGH
    log(f"selected composition (EMERGED, diagnostic only): {lv_counts} "
        f"(root {lv_counts[0] / B_HIGH:.1%}, "
        f"K2 {lv_counts[2] / B_HIGH:.1%}, K4 {lv_counts[4] / B_HIGH:.1%}, "
        f"K8 {lv_counts[8] / B_HIGH:.1%}, K16 {lv_counts[16] / B_HIGH:.1%})")

    # ONE shared column order for dev and test: roots first, then deltas
    # level-grouped (rank order preserved within level) - the hier_banks_test
    # layout; Ridge is permutation-invariant in features.
    lv_rank = {0: -1, 2: 0, 4: 1, 8: 2, 16: 3}
    order = np.argsort([lv_rank[m[0]] for m in meta], kind="stable")
    sel = sel[order]
    meta = [meta[i] for i in order]
    lv_seq = [m[0] for m in meta if m[0] != 0]
    assert lv_seq == sorted(lv_seq), "shared order not level-grouped"
    log("final column order: roots first, then K2/K4/K8/K16 "
        "(selection set unchanged)")

    sel_dict = {K: np.array([(kg - G_PART, j) for (Kk, j, p, kg) in meta
                             if Kk == K], dtype=np.int64).reshape(-1, 2)
                for K in KEPT_K}
    assert all(v.shape == (lv_counts[K], 2) for K, v in sel_dict.items())
    n_delta_sel = B_HIGH - lv_counts[0]

    with open(os.path.join(OUT, "diagnostics",
                           "selected_counts_by_level.csv"), "w") as f:
        f.write("level,source_type,pool_columns,selected,share_of_final,"
                "share_of_pool\n")
        f.write(f"0,root,{ROOTS},{lv_counts[0]},"
                f"{lv_counts[0] / B_HIGH:.6f},{lv_counts[0] / ROOTS:.6f}\n")
        for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
            pool_k = K * F_H
            sel_k = lv_counts[K]
            f.write(f"{K},delta,{pool_k},{sel_k},{sel_k / B_HIGH:.6f},"
                    f"{sel_k / pool_k:.6f}\n")

    with open(os.path.join(OUT, "diagnostics",
                           "selected_feature_metadata.csv"), "w") as f:
        f.write("selected_rank,candidate_index,source_type,level,parent,"
                "child,kernel_id,F_stat,p_value,z_score,pi0_level,"
                "mu1_level,sigma1_level,lfdr,posterior_signal_q\n")
        for rank, (c, m) in enumerate(zip(sel, meta)):
            K, j, p, kg = m
            par = eb_params[K]
            lfdr_c = 1.0 - q_scores[c]
            if K == 0:
                f.write(f"{rank},{c},root,0,NA,NA,{kg},{F_stat[c]:.6g},"
                        f"{p_vals[c]:.6g},{z_vals[c]:.6g},{par['pi0']:.6f},"
                        f"{par['mu1']:.6f},{par['sigma1']:.6f},"
                        f"{lfdr_c:.6g},{q_scores[c]:.6g}\n")
            else:
                f.write(f"{rank},{c},delta,{K},{p},{j},{kg},"
                        f"{F_stat[c]:.6g},{p_vals[c]:.6g},{z_vals[c]:.6g},"
                        f"{par['pi0']:.6f},{par['mu1']:.6f},"
                        f"{par['sigma1']:.6f},{lfdr_c:.6g},"
                        f"{q_scores[c]:.6g}\n")

    # ---- 10b. per-level EB diagnostics (spec 19/20/21) --------------------
    with open(os.path.join(OUT, "diagnostics",
                           "eb_parameters_by_level.csv"), "w") as f:
        f.write("level,candidate_count,pi0,mu1,sigma1,mean_z,median_z,"
                "mean_q,sum_q,q_90,q_95,q_99,lfdr_q10,lfdr_q50,lfdr_q90,"
                "selected_count,selected_fraction,em_iters,converged,"
                "loglik,init_degenerate\n")
        for K in (0, 2, 4, 8, 16):
            m = lv == K
            par = eb_params[K]
            zl, ql, ll = z_vals[m], q_scores[m], par["lfdr"]
            sel_k = lv_counts[K]
            f.write(f"{K},{int(m.sum())},{par['pi0']:.6f},{par['mu1']:.6f},"
                    f"{par['sigma1']:.6f},{np.mean(zl):.6f},"
                    f"{np.median(zl):.6f},{np.mean(ql):.6f},{np.sum(ql):.4f},"
                    f"{np.quantile(ql, 0.90):.6f},{np.quantile(ql, 0.95):.6f},"
                    f"{np.quantile(ql, 0.99):.6f},{np.quantile(ll, 0.10):.6f},"
                    f"{np.quantile(ll, 0.50):.6f},{np.quantile(ll, 0.90):.6f},"
                    f"{sel_k},{sel_k / B_HIGH:.6f},{par['iters']},"
                    f"{par['converged']},{par['loglik']:.4f},"
                    f"{par['init_degenerate']}\n")

    sanity = {
        "pi0_in_[0.5,1)": all(EB_PI0_MIN <= eb_params[K]["pi0"] < 1.0
                              for K in eb_params),
        "mu1_positive": all(eb_params[K]["mu1"] > 0 for K in eb_params),
        "sigma1_positive": all(eb_params[K]["sigma1"] > 0
                               for K in eb_params),
        "q_in_unit_range": bool((q_scores >= 0).all()
                                and (q_scores <= 1).all()),
        "q_finite": bool(np.isfinite(q_scores).all()),
        "lfdr_in_unit_range": all(
            bool((eb_params[K]["lfdr"] >= 0).all()
                 and (eb_params[K]["lfdr"] <= 1).all()) for K in eb_params),
        "loglik_finite": all(np.isfinite(eb_params[K]["loglik"])
                             for K in eb_params),
        "em_converged_or_fallback_recorded": {
            str(K): {"converged": eb_params[K]["converged"],
                     "iters": eb_params[K]["iters"],
                     "init_degenerate": eb_params[K]["init_degenerate"],
                     "final_delta": eb_params[K]["final_delta"]}
            for K in eb_params},
        "refit_deterministic_in_process": bool(refit_ok),
        "parameters_level_specific": {
            str(K): {"pi0": eb_params[K]["pi0"], "mu1": eb_params[K]["mu1"],
                     "sigma1": eb_params[K]["sigma1"]} for K in eb_params},
        "no_validation_or_test_input_to_eb_fit":
            "EB fitted on dev-set (train+val) statistics only for the "
            "final selection; fold-internal CV fits use fold-train only",
    }
    with open(os.path.join(OUT, "diagnostics", "eb_sanity.json"), "w") as f:
        json.dump(sanity, f, indent=1)
    assert sanity["pi0_in_[0.5,1)"] and sanity["mu1_positive"] \
        and sanity["sigma1_positive"] and sanity["q_in_unit_range"] \
        and sanity["q_finite"] and sanity["lfdr_in_unit_range"] \
        and sanity["loglik_finite"] and sanity["refit_deterministic_in_process"]

    # ---- 11. final 19,992-wide representation for BOTH splits -------------
    root_idx = np.array([c for c in sel if c < ROOTS], dtype=np.int64)
    n_root_sel = len(root_idx)
    assert n_root_sel + n_delta_sel == B_HIGH
    Z_dev = C_dev[:, sel]
    del C_dev
    assert Z_dev.shape == (n_tr + n_va, B_HIGH)
    Hcand_te = hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)
    assert Hcand_te.shape == (n_te, n_delta_sel), Hcand_te.shape
    Z_te = np.empty((n_te, B_HIGH), dtype=np.float64)
    Z_te[:, :n_root_sel] = GHK_te[:, root_idx]
    Z_te[:, n_root_sel:] = Hcand_te
    del GHK_te, Hcand_te
    assert np.isfinite(Z_dev).all() and np.isfinite(Z_te).all()
    log(f"final representation: {Z_dev.shape} / test {Z_te.shape} "
        f"(shared order: {n_root_sel} roots + {n_delta_sel} deltas)")

    # ---- 12. canonical Ridge (identical protocol) -------------------------
    res, pred = ridge_eval(Z_dev, y_dev, Z_te, yte)
    log(f"[HIER-EB-SUP] = {res['macro_f1']}  (alpha={res['selected_alpha']})")

    # ---- 13. deltas vs saved references (NOT rerun) -----------------------
    deltas = {k: round(res["macro_f1"] - v, 4) for k, v in REFS.items()}
    np.save(os.path.join(OUT, "predictions", "test_predictions.npy"),
            pred.astype(np.int64))

    # ---- 14. artifacts ----------------------------------------------------
    with open(os.path.join(OUT, "results", "hier_eb_sup.json"), "w") as f:
        json.dump({
            "method": "hier_eb_sup",
            "label": "HIER-EB-SUP",
            "dataset": DS, "seed": SEED,
            "base_budget": B_BASE, "high_budget": B_HIGH,
            "kernel_capacity": "84x238",
            "final_features": B_HIGH,
            "candidate_pool": {"root": ROOTS, "delta": H_CAND,
                               "unified": N_CAND},
            "selection": "empirical-Bayes two-groups local-FDR posterior "
                         "signal probability q = 1 - lfdr; ONE global "
                         "ranking over the unified 469,812-candidate pool "
                         "(train+val labels, canonical R5 final-stage "
                         "protocol); ties by candidate_index; no quota, "
                         "no rho, no q multiplication",
            "eb_model": "parametric two-groups: pi0*N(0,1) + "
                        "(1-pi0)*N(mu1,sigma1^2), five independent "
                        "per-level deterministic-EM fits",
            "changed_vs_hier_continuous_sup":
                "raw ANOVA-F ranking -> per-level EB two-groups lfdr/q "
                "ranking (same F statistic, same pool, same budget)",
            "eb_parameters": {str(K): {kk: eb_params[K][kk]
                                       for kk in ("pi0", "mu1", "sigma1",
                                                  "iters", "converged",
                                                  "loglik")}
                              for K in eb_params},
            "effective_signal_count_sum_q": {
                str(K): float(np.sum(q_scores[lv == K]))
                for K in (0, 2, 4, 8, 16)},
            "selected_composition": lv_counts,
            "cv_unified_fold_internal_eb": {"mean": cv_mean, "std": cv_std,
                                            "kfold": 5},
            "macro_f1": res["macro_f1"], "accuracy": res["accuracy"],
            "alpha": res["selected_alpha"],
            "deltas_vs": deltas,
            "gate_mr_high": {"ref": REF_MR_HIGH,
                             "reproduced": mr_gate["macro_f1"],
                             "diff": round(gate_diff, 4),
                             "tol": GATE_TOL},
            "runtime_s": round(time.time() - t00, 1),
            "peak_mem_mb": round(peak_mem_mb(), 1),
            "smoke": smoke,
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "dimensions.json"), "w") as f:
        json.dump({
            "split": {"train": n_tr, "validation": n_va, "test": n_te,
                      "T": int(Xtr.shape[1]),
                      "n_classes": int(len(np.unique(ytr)))},
            "extractors": {"expanded": fHK},
            "pools": {"root_candidates": ROOTS, "delta_candidates": H_CAND,
                      "edges": EDGES, "kernels_per_edge": F_H,
                      "unified_candidates": N_CAND},
            "final": {"selected": B_HIGH,
                      "Z_dev": list(Z_dev.shape),
                      "Z_test": list(Z_te.shape),
                      "composition": lv_counts},
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics", "leakage_report.json"),
              "w") as f:
        json.dump({
            "minirocket_fit": "train signals only (seed 42)",
            "hierarchy": "TRAIN latents only (frozen seed-42 SSL encoder, "
                         "recursive 2-means); val/test transform-only; "
                         "reg16_tr byte-identical to the stored "
                         "uwavey_nested artifact",
            "delta_pool_and_root_matrix": "frozen transform, label-free",
            "eb_fit_final_selection": "train+val statistics only (canonical "
                                      "R5 final-stage protocol - identical "
                                      "to HIER-HIGH-SUP / "
                                      "HIER-CONTINUOUS-SUP)",
            "eb_fit_cv_diagnostic": "5-fold StratifiedKFold(shuffle=True, "
                                    "random_state=42); per-fold ANOVA-F/p, "
                                    "z, EB fits and selection use "
                                    "fold-train rows and labels ONLY",
            "eb_hyperparameters": "PREDECLARED (init pi0=0.90, tail 10%, "
                                  "tol 1e-8, 200 iters, clamps per spec); "
                                  "no validation tuning",
            "test_labels": "touched only inside ridge_eval for scoring",
            "no_quota": "composition emerged from the global q ranking; "
                        "reported as diagnostics only",
        }, f, indent=1)

    with open(os.path.join(OUT, "diagnostics",
                           "selection_protocol.md"), "w") as f:
        f.write(
            "# Selection protocol - HIER-EB-SUP\n\n"
            "## Unified pool (locked, identical to HIER-CONTINUOUS-SUP)\n\n"
            "C = [G_full || Delta_full] with\n"
            f"- G_full: {ROOTS} root candidates (level 0 of the hierarchy)\n"
            f"- Delta_full: {H_CAND} hierarchical telescoping residuals "
            f"({EDGES} edges x {F_H} kernels; K=2,4,8,16)\n"
            f"- total: {N_CAND} candidates, canonical level-grouped layout "
            "[root | K2 | K4 | K8 | K16] (layout-guarded against col_meta)\n\n"
            "## Statistic (bit-identical to the incumbent)\n\n"
            "F, p = sklearn f_classif(C, y) - the same call top_f_select "
            "makes, so the F statistic and its exact p-values (same df, "
            "same class handling) are identical to the "
            "HIER-CONTINUOUS-SUP selector's statistic. Edge cases: "
            "non-finite F/p (constant features) -> F=0, p=1; p clipped to "
            f"[{P_CLIP_LO:g}, {P_CLIP_HI:g}] before inversion; "
            "z = norm.isf(p) is a one-sided significance coordinate "
            "(NOT an effect direction), finite in [-8.22, 37.0].\n\n"
            "## The empirical-Bayes model (the ONE change)\n\n"
            "Per level l in {0, 2, 4, 8, 16} - FIVE independent fits, "
            "never pooled:\n\n"
            "    f_l(z) = pi0_l * N(0,1) + (1 - pi0_l) * "
            "N(mu1_l, sigma1_l^2),  mu1_l > 0\n\n"
            "fitted by deterministic EM (init pi0=0.90; mu1/sigma1 from "
            "the highest 10% z with degenerate fallback max(mean(z),0.5)/"
            "max(std(z),0.5); tol 1e-8 on max parameter change; 200 "
            "iterations max; clamps pi0 in [0.5, 1-1e-8], sigma1 >= 1e-3, "
            "mu1 >= 1e-3). lfdr = P(null | z) under the fitted mixture; "
            "q = 1 - lfdr. This is the parametric two-groups EB local-FDR "
            "model - NOT Efron's nonparametric locfdr.\n\n"
            "## Stage 1 - CV diagnostic (never selects)\n\n"
            "5-fold StratifiedKFold(shuffle=True, random_state=42) over "
            "train+val (896 rows). Per fold: f_classif on fold-train "
            "rows/labels ONLY -> z -> five fold-train EB fits -> q -> one "
            "global top-19,992 -> RidgeClassifierCV(ALPHAS) on fold-train "
            "-> macro-F1 on the fold. Fold-validation labels never touch "
            "the EB fit or the selection.\n\n"
            "## Stage 2 - final selection (canonical R5 protocol)\n\n"
            "EB scoring on the FULL dev matrix (train+val labels) - the "
            "same supervised boundary R5-HIGH, HIER-HIGH-SUP and "
            "HIER-CONTINUOUS-SUP use for their final selection. ONE "
            "global ranking by q (ties by candidate_index ascending), "
            "top 19,992. NO separate G stage, no rho, no N_G/N_H, no "
            "per-level quota, no q multiplication of features.\n\n"
            "## Stage 3 - refit + one official test evaluation\n\n"
            "RidgeClassifierCV(ALPHAS=logspace(-4,4,20)) fit on the "
            f"selected {B_HIGH}-column representation over train+val; "
            "single evaluation on the official test split.\n\n"
            "## Resulting composition (emerged, diagnostic only)\n\n"
            f"root {lv_counts[0]}, K2 {lv_counts[2]}, K4 {lv_counts[4]}, "
            f"K8 {lv_counts[8]}, K16 {lv_counts[16]} "
            f"(sum {sum(lv_counts.values())} = {B_HIGH}). Estimated "
            "effective signal counts sum(q) per level are in "
            "eb_parameters_by_level.csv and are NOT quotas.\n")

    # comparison.csv + figures + summary ------------------------------------
    rows = [("canonical_mr", "Canonical MR", 9996, "84x119", "NA"),
            ("mr_high", "MR-HIGH", 19992, "84x238", "NA"),
            ("r5_high", "R5-HIGH", 19992, "84x238", "14994"),
            ("hier_high", "HIER-HIGH", 19992, "84x238", "449820"),
            ("hier_high_sup", "HIER-HIGH-SUP", 19992, "84x238", "449820"),
            ("hier_continuous_sup", "HIER-CONTINUOUS-SUP", 19992, "84x238",
             "469812"),
            ("hier_eb_sup", "HIER-EB-SUP", 19992, "84x238", "469812")]
    with open(os.path.join(OUT, "results", "comparison.csv"), "w") as f:
        f.write("method,label,final_features,root_selected,k2_selected,"
                "k4_selected,k8_selected,k16_selected,candidate_pool,"
                "kernel_capacity,macro_f1,accuracy,alpha,runtime_s,"
                "peak_mem_mb\n")
        for key, label, ff, kcap, pool in rows:
            if key == "hier_eb_sup":
                f.write(f"{key},{label},{ff},{lv_counts[0]},{lv_counts[2]},"
                        f"{lv_counts[4]},{lv_counts[8]},{lv_counts[16]},"
                        f"{pool},{kcap},{res['macro_f1']},{res['accuracy']},"
                        f"{res['selected_alpha']},"
                        f"{round(time.time() - t00, 1)},"
                        f"{round(peak_mem_mb(), 1)}\n")
            else:
                ref = REFS[key]
                f.write(f"{key},{label},{ff},NA,NA,NA,NA,NA,{pool},{kcap},"
                        f"{ref},NA,NA,NA,NA\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        # comparison figure
        labels = [r[1] for r in rows]
        vals = [REFS.get(r[0]) if r[0] in REFS else res["macro_f1"]
                for r in rows]
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        cols = ["#9e9e9e"] * 6 + ["#d62728"]
        ax.bar(labels, vals, color=cols)
        ax.set_ylim(0.70, 0.80)
        ax.set_ylabel("Macro-F1 (official test)")
        ax.set_title("HIER-EB-SUP - EB two-groups selection vs incumbents "
                     "(UWaveY, seed 42)")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures",
                                 "comparison_macro_f1.png"), dpi=150)
        plt.close(fig)

        # selected composition
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        lv_names = ["root (0)", "K=2", "K=4", "K=8", "K=16"]
        cnt = [lv_counts[0], lv_counts[2], lv_counts[4], lv_counts[8],
               lv_counts[16]]
        ax.bar(lv_names, cnt, color="#d62728")
        ax.set_ylabel("selected features (of 19,992)")
        ax.set_title("Composition EMERGED from the EB ranking (no quotas)")
        for i, v in enumerate(cnt):
            ax.text(i, v + 30, f"{v}\n({v / B_HIGH:.1%})", ha="center",
                    fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures",
                                 "selected_composition.png"), dpi=150)
        plt.close(fig)

        # effective signal (sum q) vs selected count per level
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        sum_q = [float(np.sum(q_scores[lv == K])) for K in (0, 2, 4, 8, 16)]
        x = np.arange(5)
        ax.bar(x - 0.2, sum_q, 0.4, label="estimated signal sum(q)")
        ax.bar(x + 0.2, cnt, 0.4, label="selected (of 19,992)")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(lv_names)
        ax.set_ylabel("count (log)")
        ax.set_title("EB effective signal vs final selection (distinct "
                     "notions)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures",
                                 "effective_signal_by_level.png"), dpi=150)
        plt.close(fig)

        # per-level EB density figures
        for K, nm in ((0, "level0"), (2, "k2"), (4, "k4"), (8, "k8"),
                      (16, "k16")):
            m = lv == K
            zl = z_vals[m]
            par = eb_params[K]
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            ax.hist(zl, bins=80, density=True, color="#c6dbef",
                    label="empirical z")
            grid = np.linspace(max(-8.0, zl.min()), min(40.0, zl.max()), 400)
            ax.plot(grid, norm.pdf(grid), "k--", lw=1.4, label="null N(0,1)")
            mix = (par["pi0"] * norm.pdf(grid) +
                   (1 - par["pi0"]) * norm.pdf(grid, loc=par["mu1"],
                                               scale=par["sigma1"]))
            ax.plot(grid, mix, "r-", lw=1.6, label="fitted two-groups mix")
            ax.set_yscale("log")
            ax.set_xlabel("z = Phi^-1(1 - p)")
            ax.set_ylabel("density (log)")
            ax.set_title(f"EB two-groups fit - level K={K}  "
                         f"(pi0={par['pi0']:.3f}, mu1={par['mu1']:.2f}, "
                         f"sigma1={par['sigma1']:.2f})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "figures",
                                     f"eb_density_{nm}.png"), dpi=150)
            plt.close(fig)
        log("figures written")
    except Exception as e:  # figures must never kill the science
        log(f"figure generation skipped: {e!r}")

    rt = round(time.time() - t00, 1)
    mem = round(peak_mem_mb(), 1)
    with open(os.path.join(OUT, "results", "summary.md"), "w") as f:
        f.write(
            "# HIER-EB-SUP - summary (UWaveY, seed 42)\n\n"
            f"**Macro-F1 = {res['macro_f1']}** (accuracy "
            f"{res['accuracy']}, alpha {res['selected_alpha']})\n\n"
            "## The one change\n\n"
            "HIER-CONTINUOUS-SUP ranks the unified 469,812-candidate pool "
            "by raw ANOVA-F. HIER-EB-SUP keeps the identical statistic "
            "but converts it to exact p-values, then z = Phi^-1(1-p), "
            "fits FIVE independent parametric two-groups EB models "
            "(pi0*N(0,1) + (1-pi0)*N(mu1,sigma1^2), deterministic EM, "
            "fixed standard-normal null), and ranks all candidates by "
            "posterior signal probability q = 1 - lfdr in ONE global "
            "ranking (ties by candidate_index). Top 19,992 feed the "
            "canonical Ridge. No quota, no rho, no q multiplication.\n\n"
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
            f"| **HIER-EB-SUP** | 19,992 | **{res['macro_f1']}** | NA |\n\n"
            "## EB parameters (per level)\n\n"
            "| level | pi0 | mu1 | sigma1 | EM iters | sum(q) |\n"
            "|---|---|---|---|---|---|\n"
            + "".join(
                f"| {K} | {eb_params[K]['pi0']:.4f} | "
                f"{eb_params[K]['mu1']:.3f} | {eb_params[K]['sigma1']:.3f} "
                f"| {eb_params[K]['iters']} | "
                f"{float(np.sum(q_scores[lv == K])):.1f} |\n"
                for K in (0, 2, 4, 8, 16))
            + "\nsum(q) is the estimated effective signal count - "
            "DIAGNOSTIC ONLY, never a quota.\n\n"
            f"## Emerged composition (diagnostic)\n\nroot {lv_counts[0]} "
            f"({lv_counts[0] / B_HIGH:.1%}), K2 {lv_counts[2]} "
            f"({lv_counts[2] / B_HIGH:.1%}), K4 {lv_counts[4]} "
            f"({lv_counts[4] / B_HIGH:.1%}), K8 {lv_counts[8]} "
            f"({lv_counts[8] / B_HIGH:.1%}), K16 {lv_counts[16]} "
            f"({lv_counts[16] / B_HIGH:.1%}).\n\n"
            f"CV (fold-internal EB over the unified pool): {cv_mean:.4f} "
            f"+/- {cv_std:.4f}.\n\n"
            f"MR-HIGH reproduction gate: {mr_gate['macro_f1']} vs stored "
            f"{REF_MR_HIGH} (diff {gate_diff:.4f}).\n\n"
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
