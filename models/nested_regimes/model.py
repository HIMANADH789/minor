"""NESTED / HIERARCHICAL REGIME-CONDITIONED HETEROGENEITY (proposed).

Replaces the flat K=8 regime-conditioned heterogeneity with ONE nested
binary tree over the frozen RCMKN seed-42 context-model temporal latents.

WHAT IS PARTITIONED (precise, spec section 33)
----------------------------------------------
The tree partitions the FROZEN SSLTemporalEncoder temporal latents
z_t in R^32 (T=1092 per sample), pooled over all TRAIN samples.  The
frozen flat VQ (K=8, HardVQ) is NOT modified and NOT used for the tree;
the hierarchy is a train-only recursive 2-means in the same latent space.

Level ids are BINARY PATH ids: level l (K = 2^l) has ids {0..2^l - 1};
children of level-(l-1) id p are 2p and 2p+1; the level-l ancestor of a
finer id b is b >> (l_fine - l_coarse).  Nesting holds by construction
and is asserted.

MATHEMATICAL CORE (exact, spec sections 1/9/11/12/13)
-----------------------------------------------------
Per sample i, kernel m, padding group g (valid slice [p, T-p)), let
S16 = the VALID timesteps in min-occupancy-selected level-16 regimes
(selection rule identical to the flat bank: regime count over the valid
slice >= ceil(0.01*T); if none survives, the argmax-count regime is
kept).  All levels are evaluated on the SAME support S16.  Because a
selected level-16 regime has >= min_count timesteps, every coarser
regime containing it automatically satisfies the occupancy rule, so
within S16 the only exclusion happens at level 16 and the weight system
is consistent across levels.  With

    n_c   = |S16 intersect regime c|     (per sample, group)
    pi_c  = n_c / |S16|                  (occupancy weight, sums to 1)
    PPV_c = (activation sum in c) / n_c  (undefined iff n_c = 0; such
            regimes are excluded -- no PPV is invented)

the chain is, for level l (K = 2^l), parent p, children c = 2p, 2p+1:

    PPV_p     = sum_c pi_c PPV_c / sum_c pi_c   (over occupied children)
    Delta_c   = PPV_c - PPV_p
    E_l(m,i)  = sum_p sum_{c children(p)} pi_c Delta_c^2
    Var16(m,i)= sum_{c at K=16} pi_c (PPV_c - g)^2,  g = sum_c pi_c PPV_c

Then EXACTLY (telescoping / nested ANOVA):

    sum_c pi_c Delta_c = 0  within every parent          (zero-sum)
    Var16 = E(K=2) + E(K=4) + E(K=8) + E(K=16)           (identity)
    <f_l, f_l'> = 0 for l != l', where the level-l detail FUNCTION is
        f_l(t) = Delta_{c_l(t)} on S16, 0 outside
    (cross terms vanish by the pi-weighted zero-sum of the finer level
     inside each coarser regime + disjoint supports across branches --
     NO Gram-Schmidt, NO CCA).  Computed EXACTLY for ALL kernels via
     joint regime-count matrices, not by subsampling.

The flat K=8 bank used H_m = sum_k q_k (PPV_k - PPV_m)^2 (one number per
kernel); here the same regime-conditioned variation is DECOMPOSED by
resolution.  The hierarchical detail feature bank is Delta^(l) per
kernel/level (the classifier bank), and E_l(m) is the level-energy
diagnostic.

LABEL-FREE NULL (spec sections 14-16)
-------------------------------------
Per permutation: shuffle each train sample's level-16 regime sequence
(per-sample permutation preserving the exact per-sample regime-size
multiset), recompute the identical chain.  S=500, seed 52042.  No
labels, no validation, no test anywhere.

STOPPING RULE (predeclared): keep level l iff real E_l > null 95th
percentile AND Benjamini-Hochberg q_l < 0.05, walking coarse->fine and
stopping at the first failure.  L* = finest kept level (K value).
"""
import json

import numpy as np

SEED = 42
NULL_SEED = 52042
S_PERM = 500
MIN_OCCUPANCY = 0.01
LEVELS = (1, 2, 4, 8, 16)          # K at depth 0..4
N_DET = 4                          # detail levels K=2,4,8,16
FDR_ALPHA = 0.05
IP_PAIRS = ((2, 4), (2, 8), (2, 16), (4, 8), (4, 16), (8, 16))


# ---------------------------------------------------------------------------
# tree construction (train latents only) and frozen assignment
# ---------------------------------------------------------------------------
def build_latent_tree(latents, seed=SEED, max_depth=4):
    """Recursive 2-means over pooled latent vectors (train only).

    Returns (C, meta): C is a list of centroid arrays, C[d] of shape
    (2^d, D) for depth d = 0..max_depth; meta records the exact
    construction.  Children of level-d centroid p are 2p (bit 0) and
    2p+1 (bit 1); the bit-0 child is the one whose centroid has the
    smaller FIRST coordinate (deterministic tie-free ordering rule).
    Nodes with < 2 members fall back to duplicating the parent centroid
    (recorded in meta).  Requires scipy-free sklearn KMeans only.
    """
    from sklearn.cluster import KMeans
    Z = np.ascontiguousarray(latents, dtype=np.float64)
    C = [Z.mean(axis=0)[None, :]]
    fallbacks = 0
    for d in range(max_depth):
        lev = assign_levels(Z, C)[d]                    # (M,) level ids
        child = np.repeat(C[d], 2, axis=0).copy()       # (2^(d+1), D)
        for p in range(2 ** d):
            members = np.where(lev == p)[0]
            if len(members) < 2:
                fallbacks += 1
                continue                                # parent centroid kept
            km = KMeans(n_clusters=2, n_init=10, random_state=seed)
            km.fit(Z[members])
            c0, c1 = km.cluster_centers_
            if c1[0] < c0[0]:                           # deterministic order
                c0, c1 = c1, c0
            child[2 * p], child[2 * p + 1] = c0, c1
        C.append(child)
    meta = {
        "n_latents": int(len(Z)), "latent_dim": int(Z.shape[1]),
        "max_depth": int(max_depth),
        "splitter": "sklearn KMeans(n_clusters=2, n_init=10, "
                    f"random_state={seed}) per node",
        "child_rule": "children of level-d id p are 2p (bit 0), 2p+1 "
                      "(bit 1); bit-0 child has smaller first centroid "
                      "coordinate",
        "level_ids": "binary path ids; ancestor of b at level l is "
                     "b >> (l_fine - l)",
        "singleton_fallbacks": int(fallbacks),
        "input": "pooled TRAIN SSLTemporalEncoder latents (R^32)",
    }
    return C, meta


def assign_levels(Z, C):
    """Frozen-tree assignment of (M, D) latents: list of (M,) int64 level
    ids for depth 0..len(C)-1 (transform only -- never refits).

    Assignment is successive nearest-CHILD from the root, so nesting
    holds by construction: level-d id j's parent is j >> 1 at level d-1.
    """
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    cur = np.zeros(len(Z), dtype=np.int64)
    out = [cur.copy()]
    for d in range(1, len(C)):
        child = C[d]
        new = np.empty(len(Z), dtype=np.int64)
        for p in np.unique(cur):
            idx = np.where(cur == p)[0]
            c0, c1 = child[2 * p], child[2 * p + 1]
            d0 = ((Z[idx] - c0) ** 2).sum(axis=1)
            d1 = ((Z[idx] - c1) ** 2).sum(axis=1)
            new[idx[d0 <= d1]] = 2 * p
            new[idx[d1 < d0]] = 2 * p + 1
        cur = new
        out.append(cur.copy())
    return out


def nesting_errors(levels):
    """Unique-parent invariants (spec section 6)."""
    errs = []
    for lo in range(len(levels) - 1):
        fine, coarse = levels[lo + 1], levels[lo]
        pairs = np.stack([fine, coarse], axis=1)
        if len(np.unique(pairs, axis=0)) != len(np.unique(fine)):
            errs.append((2 ** (lo + 1), 2 ** lo))
    return errs


# ---------------------------------------------------------------------------
# exact ANOVA chain from per-sample regime counts/sums on S16
# ---------------------------------------------------------------------------
def chain_from_counts_sums(cnt, sums, want_deltas=True, want_ips=True):
    """Exact nested chain.  cnt: (n, 16) int64 selected-regime counts on
    S16 (0 = excluded); sums: (n, 16, F) float64 activation sums over the
    SAME support (already zeroed where excluded).

    Returns dict with:
      E        (n, 4, F)  level energies, slots ordered K=2,4,8,16
      var16    (n, F)     weighted conditional variance at K=16
      g        (n, F)     pooled S16 PPV (level-1 mean)
      deltas   dict K -> (n, K, F)  (K=2,4,8,16; regime order 0..K-1)
      cnts     dict K -> (n, K) float64 S16 counts
      zero_sum_max, identity_max_abs/rel, cross_ip (6, F) SIGNED <f_l,f_l'>
      per kernel (caller takes |.| after summing over padding groups)
    """
    n = cnt.shape[0]
    F = sums.shape[2]
    nv = cnt.sum(axis=1).astype(np.float64)              # (n,)
    nvs = np.maximum(nv, 1.0)[:, None]
    pi16 = cnt / nvs
    ppv16 = sums / np.maximum(cnt, 1)[:, :, None]
    g = (pi16[:, :, None] * ppv16).sum(axis=1)           # (n, F)
    var16 = (pi16[:, :, None] * (ppv16 - g[:, None, :]) ** 2).sum(axis=1)

    E = np.empty((n, N_DET, F))
    deltas, cnts = {}, {}
    zs = 0.0
    cross = (np.zeros((len(IP_PAIRS), F)) if want_ips else None)

    for li in range(N_DET):                              # li=0 -> K=2 ...
        K = 2 ** (li + 1)
        # level depth d = li+1; depth-4 ids -> depth-d ids: >> (4 - d)
        idx = np.arange(16) >> (N_DET - 1 - li)          # level ids 0..K-1
        cntK = np.stack([cnt[:, idx == j].sum(axis=1).astype(np.float64)
                         for j in range(K)], axis=1)     # (n, K)
        sumK = np.stack([sums[:, idx == j].sum(axis=1)
                         for j in range(K)], axis=1)     # (n, K, F)
        piK = cntK / nvs
        ppvK = sumK / np.maximum(cntK, 1.0)[:, :, None]
        pc0, pc1 = piK[:, 0::2], piK[:, 1::2]
        den = np.maximum(pc0 + pc1, 1e-300)[:, :, None]
        pp_par = (pc0[:, :, None] * ppvK[:, 0::2]
                  + pc1[:, :, None] * ppvK[:, 1::2]) / den
        d0, d1 = ppvK[:, 0::2] - pp_par, ppvK[:, 1::2] - pp_par
        E[:, li] = (pc0[:, :, None] * d0 ** 2
                    + pc1[:, :, None] * d1 ** 2).sum(axis=1)
        zs = max(zs, float(np.abs(pc0[:, :, None] * d0
                                  + pc1[:, :, None] * d1).max()))
        cnts[K] = cntK       # per-level counts (independent of deltas)
        if want_deltas:
            # interleave: position 2p = even child, 2p+1 = odd child
            dK = np.empty_like(ppvK)
            dK[:, 0::2], dK[:, 1::2] = d0, d1
            dK[cntK <= 0] = 0.0          # excluded regimes: no detail
            deltas[K] = dK

    if want_ips:
        for pi_i, (K_lo, K_hi) in enumerate(IP_PAIRS):
            sh = int(np.log2(K_hi // K_lo))
            anc = np.arange(K_hi) >> sh                   # (K_hi,) -> level lo
            d_lo = deltas[K_lo][:, anc, :]                # (n, K_hi, F)
            cross[pi_i] = (
                cnts[K_hi][:, :, None] * d_lo * deltas[K_hi]).sum(axis=(0, 1))

    ident = np.abs(var16 - E.sum(axis=1))
    denom = max(float(np.abs(var16).max()), 1e-30)
    return {"E": E, "var16": var16, "g": g, "deltas": deltas, "cnts": cnts,
            "zero_sum_max": zs, "identity_max_abs": float(ident.max()),
            "identity_max_rel": float(ident.max() / denom),
            "cross_ip": cross, "n_support": nv}


def select_and_mask(cnt, min_count):
    """Selection at level 16: keep regimes with count >= min_count; if
    none survives keep the argmax-count regime.  Returns boolean (n, 16)."""
    sel = cnt >= min_count
    fb = ~sel.any(axis=1)
    if fb.any():
        sel[fb, np.argmax(cnt[fb], axis=1)] = True
    return sel


def hierarchy_chain(act, valid, reg16, min_occupancy=MIN_OCCUPANCY,
                    want_deltas=True, want_ips=True, kernel_chunk=512,
                    delta_cols=None):
    """Full exact chain for act (n, F, T) bool, valid (F, T) bool, reg16
    (n, T) int level-16 ids in {0..15}.  Handles padding groups and the
    S16 support internally; per-level quantities are SUMMED over groups.

    delta_cols: optional dict K -> array of (kernel, child) column ids;
    when given, only those detail columns are materialized (memory-lean
    frozen-transform banks) while energies are still computed for ALL
    kernels.  Default None keeps the full detail bank (Haptics behavior).

    Returns chain_from_counts_sums output plus:
      E_total (4,)  sum over samples and kernels (stopping-rule input)
      E_sample (n, 4)  per-sample total energies
      groups, min_count, support stats
    """
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        _padding_groups)  # verified shared convention (valid slice [p, T-p))
    n, F, T = act.shape
    min_count = int(np.ceil(min_occupancy * T))
    groups = _padding_groups(valid)
    E = np.zeros((n, N_DET, F))
    var16 = np.zeros((n, F))
    g = np.zeros((n, F))
    zs = 0.0
    ia = ir = 0.0
    cross = np.zeros((len(IP_PAIRS), F)) if want_ips else None
    n_sup = np.zeros(n)
    n_act = np.zeros((n, 16), dtype=np.int64)
    cnt16_full = np.zeros((n, 16))                   # S16 counts (level 16)
    if want_deltas:
        if delta_cols is None:
            deltas_full = {K: np.zeros((n, K, F)) for K in LEVELS[1:]}
        else:
            deltas_full = {K: np.zeros((n, len(delta_cols[K])))
                           for K in LEVELS[1:]}
        edge_acc = {K: np.zeros((K, F)) for K in LEVELS[1:]}
    for p, feats in groups:
        hi = T - p if p > 0 else T
        rv = reg16[:, p:hi]                              # (n, nv)
        onehot = np.zeros((n, 16, rv.shape[1]))
        onehot[np.arange(n)[:, None], rv,
               np.arange(rv.shape[1])[None, :]] = 1.0
        cnt_g = onehot.sum(axis=2).astype(np.int64)
        sel = select_and_mask(cnt_g, min_count)
        cnt_sel = np.where(sel, cnt_g, 0)
        n_act = np.maximum(n_act, (cnt_g * sel) > 0)
        for c0 in range(0, len(feats), kernel_chunk):
            sl = feats[c0:c0 + kernel_chunk]
            Av = act[:, sl, p:hi].astype(np.float64)
            sums = np.einsum("nct,nft->ncf", onehot, Av)
            sums *= sel[:, :, None]                      # S16 support only
            ch = chain_from_counts_sums(cnt_sel, sums,
                                        want_deltas=want_deltas,
                                        want_ips=want_ips)
            E[:, :, sl] += ch["E"]
            var16[:, sl] += ch["var16"]
            g[:, sl] += ch["g"]
            cnt16_full += ch["cnts"][16]             # identical across feats
            zs = max(zs, ch["zero_sum_max"])
            ia = max(ia, ch["identity_max_abs"])
            ir = max(ir, ch["identity_max_rel"])
            if want_deltas:
                if delta_cols is None:
                    for K in LEVELS[1:]:
                        deltas_full[K][:, :, sl] += ch["deltas"][K]
                else:
                    # chunk-local kernel mapping: ch deltas are (n, K,
                    # F_chunk) ordered like sl; delta_cols hold GLOBAL
                    # (kernel, child) ids.
                    pos = {int(g): i for i, g in enumerate(sl)}
                    for K in LEVELS[1:]:
                        cols = np.asarray(delta_cols[K])
                        if cols.size == 0:
                            continue            # empty selection: no-op
                        lm = np.array([pos.get(int(m), -1)
                                       for m in cols[:, 0]],
                                      dtype=np.int64)
                        ok = lm >= 0
                        deltas_full[K][:, ok] += \
                            ch["deltas"][K][:, cols[ok, 1], lm[ok]]
                # per-edge structural energies, EXACT per padding group
                # (the group's own occupancy weights, same convention as
                # E): e_{m,c} accumulates sum_i pi_c Delta_c^2, divided by
                # n at the end.  K=2: the two children alias the single
                # parent edge (both = sum_i E_{i,l=0}).
                for li in range(N_DET):
                    K = LEVELS[1:][li]
                    if K == 2:
                        e2 = ch["E"][:, 0, :].sum(axis=0)
                        edge_acc[2][0, sl] += e2
                        edge_acc[2][1, sl] += e2
                    else:
                        cntK = ch["cnts"][K]
                        piK = cntK / np.maximum(
                            ch["n_support"], 1.0)[:, None]
                        edge_acc[K][:, sl] += (piK[:, :, None]
                                               * ch["deltas"][K]
                                               ** 2).sum(axis=0)
                if want_ips:
                    cross[:, sl] += ch["cross_ip"]
        n_sup += ch["n_support"]                         # once per group
    if want_ips:
        cross = np.abs(cross)
    out = {"E": E, "var16": var16, "g": g, "zero_sum_max": zs,
           "identity_max_abs": ia, "identity_max_rel": ir,
           "n_support": n_sup, "n_active_regimes": n_act,
           "cross_ip": cross, "cnts": {16: cnt16_full},
           "E_total": E.sum(axis=(0, 2)), "E_sample": E.sum(axis=2),
           "groups": [(int(p), len(f)) for p, f in groups],
           "min_count": min_count}
    if want_deltas:
        out["deltas"] = deltas_full
        out["edge_e"] = {K: edge_acc[K] / float(max(n, 1))
                         for K in LEVELS[1:]}
    return out


# ---------------------------------------------------------------------------
# shuffled-regime null (label-free; GPU fast path, CPU fallback)
# ---------------------------------------------------------------------------
def permute_regimes(reg16, rng):
    """Per-sample permutation of the level-16 regime sequence (the null's
    only intervention).  Preserves each sample's regime-size multiset
    exactly; destroys the regime <-> activation temporal alignment."""
    out = np.empty_like(reg16)
    for i in range(reg16.shape[0]):
        out[i] = rng.permutation(reg16[i])
    return out


def shuffle_null(act, valid, reg16, S=S_PERM, seed=NULL_SEED,
                 min_occupancy=MIN_OCCUPANCY, chunk=1024, verbose=0):
    """Null distribution of per-level total energies (S, 4).

    Per permutation: per-sample permutation of the level-16 regime
    sequence (preserves each sample's regime-size multiset exactly),
    then the IDENTICAL estimator (same groups, same selection rule,
    same chain).  Regime occupancy counts are preserved by design.
    """
    import torch
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        _padding_groups)
    n, F, T = act.shape
    min_count = int(np.ceil(min_occupancy * T))
    groups = _padding_groups(valid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    if use_gpu:
        actg = torch.from_numpy(np.ascontiguousarray(
            act.transpose(0, 2, 1)).astype(np.uint8)).to(device)  # (n,T,F)
        ar_n = torch.arange(n, device=device)
        ar_T = torch.arange(T, device=device)
    rng = np.random.default_rng(seed)
    null = np.empty((S, N_DET))
    for s in range(S):
        regp = permute_regimes(reg16, rng)
        if use_gpu:
            rvg = torch.from_numpy(np.ascontiguousarray(regp)).to(device)
            ohg = torch.zeros((n, 16, T), dtype=torch.float32, device=device)
            ohg[ar_n[:, None], rvg, ar_T[None, :]] = 1.0
        acc = np.zeros(N_DET)
        for p, feats in groups:
            hi = T - p if p > 0 else T
            rv = regp[:, p:hi]
            onehot = np.zeros((n, 16, rv.shape[1]))
            onehot[np.arange(n)[:, None], rv,
                   np.arange(rv.shape[1])[None, :]] = 1.0
            cnt_g = onehot.sum(axis=2).astype(np.int64)
            sel = select_and_mask(cnt_g, min_count)
            cnt_sel = np.where(sel, cnt_g, 0)
            selfl = torch.from_numpy(sel.astype(np.float32)).to(
                device)[:, :, None] if use_gpu else sel[:, :, None]
            for c0 in range(0, len(feats), chunk):
                sl = feats[c0:c0 + chunk]
                if use_gpu:
                    oh_sl = ohg[:, :, p:hi] * selfl
                    Ag = actg[:, p:hi, sl].float()           # (n, nv, Fp)
                    sums = torch.einsum("nct,ntf->ncf", oh_sl, Ag)
                    sums = sums.cpu().numpy().astype(np.float64)
                else:
                    Av = act[:, sl, p:hi].astype(np.float64)
                    sums = np.einsum("nct,nft->ncf", onehot, Av)
                sums *= sel[:, :, None]
                ch = chain_from_counts_sums(cnt_sel, sums,
                                            want_deltas=False,
                                            want_ips=False)
                acc += ch["E"].sum(axis=(0, 2))
        null[s] = acc
        if verbose and (s + 1) % verbose == 0:
            print(f"    null {s + 1}/{S}", flush=True)
    if use_gpu:
        del actg
        torch.cuda.empty_cache()
    return null


# ---------------------------------------------------------------------------
# stopping rule (predeclared)
# ---------------------------------------------------------------------------
def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q)."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate(
        (p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.clip(q, 0.0, 1.0)


def stopping_rule(real_E, null, fdr_alpha=FDR_ALPHA):
    """Predeclared rule (spec section 16).  real_E: (4,) level TOTAL
    energies ordered K=2,4,8,16; null: (S, 4).  Returns (L_star, rows)."""
    S = null.shape[0]
    null_mean = null.mean(axis=0)
    null_p95 = np.percentile(null, 95, axis=0)
    pvals = (1 + (null >= real_E[None, :]).sum(axis=0)) / (S + 1)
    q = bh_fdr(pvals)
    rows, L_star = [], 0
    for i, K in enumerate(LEVELS[1:]):
        keep = bool(real_E[i] > null_p95[i] and q[i] < fdr_alpha)
        rows.append({"K": int(K), "real_energy": float(real_E[i]),
                     "null_mean": float(null_mean[i]),
                     "null_std": float(null.std(axis=0)[i]),
                     "null_p95": float(null_p95[i]),
                     "p_raw_plus_one": float(pvals[i]), "q_bh": float(q[i]),
                     "keep": keep})
        if keep:
            L_star = int(K)
        else:
            break
    return L_star, rows


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
