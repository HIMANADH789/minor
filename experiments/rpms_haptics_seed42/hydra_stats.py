"""Branch 3: regime-conditional Hydra competitive dispersion (HydraH).

Uses the project's validated Hydra core, aeon's ``_HydraInternal`` (the
HydraTransformer's exact inner module), in the same re-seeded, CPU,
univariate manner as the audited RCMKN Stage-5 extractor
(experiments/rcmkn_haptics_seed42/kernel_features.py), with one documented
difference: g=64 (h=32 groups per (dilation, diff) pass) instead of g=16 so
the branch can supply the mandated 3332 units (8 dilations x 2 passes x
32 groups x 8 kernels = 4096 >= 3332).

Geometry / alignment (verified against aeon source):
  * dilations d = 2^0..2^7 (T=1092 -> 8 dilations), kernel length 9,
    padding p = 4d.
  * RAW pass input length T: output length L = floor((T + 8d - 9)/d) + 1;
    output position i corresponds to receptive-field CENTER t_in = i*d.
  * DIFF pass input length T-1: L = floor((T - 1 + 8d - 9)/d) + 1; output
    i spans diff element i*d (i.e. X[i*d] -> X[i*d+1]); assigned regime is
    regimes[min(i*d, T-2)] (half-open center; documented approximation).
  * VALID positions: receptive field fully inside the input
        first tap  i*d - 4d >= 0   ->  i >= 4
        last tap   i*d + 4d <= L_in - 1
    i.e. input-center window [4d, L_in - 1 - 4d] -- the exact analogue of
    MiniRocket's per-kernel [padding, T-padding) convention with
    padding = 4d.  aeon itself pools over ALL outputs; RPMS restricts to
    the valid window (audited).

Win-unit definition (Hydra's exact competitive rule):
Hydra's count_max scatter uses max_indices = argmax_k Z[g, k, t] -- the
winning kernel of group g at position t.  RPMS conditions that winner on
the learned VQ regime:

    c_{u,k}(i)   = #{ valid t_in : win_u(i,t) = k }        (global count)
    c_{u,k,r}(i) = #{ valid t_in : win_u(i,t) = k, regime = r }
    W_{u,k}(i)   = c_{u,k}(i) / n_valid(i)                 (win rate)
    W_{u,k,r}(i) = c_{u,k,r}(i) / n_{u,r}(i)               (regime win rate)
    H^Hydra_u(i) = sum_r q_{u,r} * sum_k (W_{u,k,r}(i) - W_{u,k}(i))^2

with q_{u,r} = n_{u,r}/n_valid renormalized over regimes passing the
audited min-occupancy rule (n_{u,r} >= ceil(0.01*T)), zero contribution
otherwise -- identical semantics to the validated H_m.  Deliberate,
documented deviation from Hydra's official feature: RPMS counts HARD wins
(ones), not Hydra's value-weighted count_max scatter (sum of winning
response magnitudes); the competitive structure (winner rule, groups,
dilations) is exactly Hydra's.

Each unit u contributes ONE scalar feature H^Hydra_u (aggregated over the
k winners of its group).  No raw Hydra counts, no SparseScaler, no raw
responses enter the classifier.
"""

import numpy as np
import torch
import torch.nn.functional as F

MIN_OCCUPANCY = 0.01
KERNEL_LEN = 9


def build_hydra_bank(T, k=8, g=64, seed=42):
    """Re-seeded deterministic _HydraInternal bank (CPU, frozen)."""
    from aeon.transformations.collection.convolution_based._hydra import (
        _HydraInternal,
    )
    torch.manual_seed(seed)
    bank = _HydraInternal(T, 1, k=k, g=g)
    for p in bank.parameters():
        p.requires_grad_(False)
    return bank.eval()


def hydra_unit_geometry(bank, T):
    """Per-unit (dilation, diff, valid output indices, input-center map).

    Returns list over units in canonical (dilation, diff, group, kernel)
    order; each entry dict with 'n_valid' and the shared per-(d, j) index
    arrays (deduplicated via cache)."""
    geo = []
    for di in range(bank.num_dilations):
        d = int(bank.dilations[di])
        for dj in range(bank.divisor):
            L_in = T if dj == 0 else T - 1
            L_out = int(np.floor((L_in + 8 * d - KERNEL_LEN) / d)) + 1
            i = np.arange(L_out)
            t_in = i * d
            valid = (t_in >= 4 * d) & (t_in <= L_in - 1 - 4 * d)
            t_map = np.clip(t_in, 0, (T - 1) if dj == 0 else (T - 2))
            for g_i in range(bank.h):
                for k_i in range(bank.k):
                    geo.append({"di": di, "dj": dj, "group": g_i,
                                "kernel": k_i, "d": d, "L_out": L_out,
                                "valid": valid, "t_map": t_map,
                                "n_valid": int(valid.sum())})
    return geo


@torch.no_grad()
def hydra_regime_counts(bank, X_z, regimes, batch=16):
    """Accumulate per-unit winner x regime counts (never materializing
    full winner tensors).

    Returns:
      c_kr : (N, U, K, K) int32  counts c_{u,k,r} (winner k, regime r);
             nonzero only at winner index == the unit's own kernel
      n_ir : (N, U, K)    int32  REGIME OCCUPANCY within each unit's valid
             region (denominator of W_{u,k,r}; NOT the win count)
      n_v  : (N, U)       int32  per-unit valid positions total
    """
    N, T = X_z.shape
    K = 8
    U = bank.num_dilations * bank.divisor * bank.h * bank.k
    c_kr = np.zeros((N, U, K, K), dtype=np.int32)
    n_ir = np.zeros((N, U, K), dtype=np.int32)
    dev = "cpu"
    W = bank.W
    for c0 in range(0, N, batch):
        c1 = min(c0 + batch, N)
        xb = torch.from_numpy(np.ascontiguousarray(
            X_z[c0:c1], dtype=np.float32)).to(dev)
        if xb.ndim == 2:                     # (B, T) -> (B, 1, T) for conv1d
            xb = xb.unsqueeze(1)
        diff_X = torch.diff(xb) if bank.divisor > 1 else None
        u0 = 0
        for di in range(bank.num_dilations):
            d = int(bank.dilations[di]); p = int(bank.paddings[di])
            for dj in range(bank.divisor):
                inp = xb if dj == 0 else diff_X
                Z = F.conv1d(inp, W[di, dj].to(dev), dilation=d, padding=p)
                Z = Z.view(c1 - c0, bank.h, bank.k, -1)
                win = Z.argmax(dim=2).numpy()          # (B, h, L_out)
                B, h, L_out = win.shape
                L_in = T if dj == 0 else T - 1
                i = np.arange(L_out)
                t_in = i * d
                valid = (t_in >= 4 * d) & (t_in <= L_in - 1 - 4 * d)
                t_map = np.clip(t_in, 0, (T - 1) if dj == 0 else (T - 2))
                vi = np.nonzero(valid)[0]
                w_blk = win[:, :, vi]                   # (B, h, Lv)
                r_blk = regimes[c0:c1][:, t_map[vi]]    # (B, Lv)
                B, h, Lv = w_blk.shape
                U_blk = h * bank.k
                # For each (b, group g, position j): the winner w IS the
                # kernel index of the unit that wins -> only unit
                # u = g*K + w gets its winner-dim w incremented at regime
                # r.  Flat c_kr index: ((b*U_blk + g*K + w)*K + w)*K + r.
                b_idx = np.arange(B)[:, None, None]
                g_idx = np.arange(h)[None, :, None]
                idx = ((b_idx * U_blk + g_idx * bank.k + w_blk) * K + w_blk) \
                    * K + r_blk[:, None, :]
                flat = np.bincount(idx.ravel(),
                                   minlength=B * U_blk * K * K)
                c_blk = flat.reshape(B, U_blk, K, K)
                c_kr[c0:c1, u0:u0 + U_blk] = c_blk
                # Denominator of W_{u,k,r}: REGIME OCCUPANCY within the
                # unit's valid region (all units of a (d, j) block share the
                # same valid mask -> same occupancy vector).
                occ_blk = np.stack(
                    [np.bincount(r_blk[b], minlength=K) for b in range(B)])
                n_ir[c0:c1, u0:u0 + U_blk] = occ_blk[:, None, :]
                u0 += U_blk
        assert u0 == U
    n_v = n_ir.sum(axis=2)
    return c_kr, n_ir, n_v


def hydra_h_from_counts(c_kr, n_ir, n_v, T, K=8,
                        min_occupancy=MIN_OCCUPANCY):
    """Exact HydraH per unit from counts (closed form).

    Returns (H (N, U) float64, contributions (N, U, K) float64) where
    contributions[..., r] = q_r * sum_k (W_kr - W_k)^2 (zero for excluded
    regimes)."""
    min_count = int(np.ceil(min_occupancy * T))
    n_v_safe = np.maximum(n_v, 1).astype(np.float64)
    # c_kr axes: (N, U, winner, regime).  W_k = global win rate of winner
    # k = sum over the REGIME axis (3) / n_valid.
    W_k = c_kr.sum(axis=3) / n_v_safe[..., None]           # (N, U, K win)
    # n_ir axes: (N, U, regime).  Insert the new axis BEFORE the regime
    # axis (n_ir[..., None, :]) so regime aligns with c_kr's LAST axis;
    # n_ir[..., None] would broadcast regime against c_kr's WINNER axis
    # (the bug this comment guards against).
    W_kr = c_kr / np.maximum(n_ir, 1)[..., None, :].astype(np.float64)
    # occupancy (N, U, K reg) drives regime selection and q weights
    occ = n_ir.astype(np.float64)                          # (N, U, K)
    q = np.where(occ >= min_count, occ / n_v_safe[..., None], 0.0)
    # fallback: if no regime passes, use the dominant regime only.
    # Renormalization must run for ALL units afterwards (an earlier version
    # renormalized only in the else-branch, so units with excluded regimes
    # got q summing to <1 -- a ~1% systematic H shrinkage the audit caught).
    any_sel = q.sum(axis=2) == 0
    if any_sel.any():
        idx = np.argmax(n_ir, axis=2)          # (N, U)
        q[any_sel] = 0.0
        q[any_sel, idx[any_sel]] = 1.0         # (M, K) row -> argmax regime
    q = q / np.maximum(q.sum(axis=2, keepdims=True), 1e-12)
    # Unit u = (g, k): its own winner index is the k offset within the
    # group -> extract the (winner == own kernel) row per unit.
    K = c_kr.shape[-1]
    U = c_kr.shape[1]
    own = np.arange(U) % K                                  # (U,) own kernel
    dev_own = W_kr[np.arange(c_kr.shape[0])[:, None],
                   np.arange(U)[None, :], own, :]  # (N, U, K reg)
    W_own = W_k[np.arange(c_kr.shape[0])[:, None],
                np.arange(U)[None, :], own]      # (N, U)
    per_reg = (dev_own - W_own[..., None]) ** 2             # (N, U, K reg)
    contrib = q * per_reg
    H = contrib.sum(axis=2)
    return H, contrib


def hydra_h_independent(win_group, regimes_single, d, dj, T, kernel_idx,
                        K=8, min_occupancy=MIN_OCCUPANCY):
    """Reference recompute for ONE unit (group, kernel_idx) of ONE sample.

    win_group: (L_out,) winner series of the unit's GROUP
    Measures dispersion of THIS kernel's win rate across regimes:
        W_m    = #{valid t: win == kernel_idx} / n_valid
        W_{m,r}= #{valid t in r: win == kernel_idx} / occ_r
        H      = sum_r q_r (W_{m,r} - W_m)^2   (min-occupancy, renorm q)
    """
    L_in = T if dj == 0 else T - 1
    L_out = len(win_group)
    i = np.arange(L_out)
    t_in = i * d
    valid = (t_in >= 4 * d) & (t_in <= L_in - 1 - 4 * d)
    t_map = np.clip(t_in, 0, (T - 1) if dj == 0 else (T - 2))
    rv = np.asarray(regimes_single)[t_map[valid]].astype(int)
    wv = (np.asarray(win_group)[valid].astype(int) == kernel_idx)
    min_count = int(np.ceil(min_occupancy * T))
    n_v = len(rv)
    if n_v == 0:
        return 0.0
    occ = np.bincount(rv, minlength=K)
    sel = occ >= min_count
    if not sel.any():
        sel[np.argmax(occ)] = True
    q = np.where(sel, occ / n_v, 0.0)
    q = q / q.sum()
    W_m = wv.mean()
    h = 0.0
    for r in range(K):
        if sel[r] and occ[r] > 0:
            W_mr = wv[rv == r].mean()
            h += q[r] * (W_mr - W_m) ** 2
    return float(h)
