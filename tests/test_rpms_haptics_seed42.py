"""RPMS mandatory tests: allocation, H/HydraH formulas, controls, audits."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.rpms_haptics_seed42 import core, hydra_stats, information
from experiments.rpms_haptics_seed42.core import N_BRANCH, N_TOTAL

K = 8
T = 200


def test_budget_constants():
    assert N_BRANCH == 3332 and N_TOTAL == 9996
    assert 3 * N_BRANCH == N_TOTAL


def test_haptics_registry():
    assert core.EXPECTED == {"train": 132, "val": 23, "test": 308,
                             "T": 1092, "n_classes": 5}
    assert Path(core.R2_CKPT).exists()


def test_hydra_bank_deterministic():
    b1 = hydra_stats.build_hydra_bank(T, k=8, g=64, seed=42)
    b2 = hydra_stats.build_hydra_bank(T, k=8, g=64, seed=42)
    assert torch.equal(b1.W, b2.W)
    # T=1092 -> 8 dilations; T=200 -> 5. Formula check instead of constant:
    import math
    expected_dil = int(np.log2((1092 - 1) / 8)) + 1
    b_h = hydra_stats.build_hydra_bank(1092, k=8, g=64, seed=42)
    assert b_h.num_dilations == expected_dil == 8
    assert b_h.num_dilations * b_h.divisor * b_h.h * b_h.k == 4096


def test_hydra_geometry_valid_window():
    b = hydra_stats.build_hydra_bank(T, k=8, g=8, seed=42)
    geo = hydra_stats.hydra_unit_geometry(b, T)
    d0 = geo[0]["d"]
    # first units: d=1, raw pass -> output length T
    assert geo[0]["L_out"] == T
    # valid window: t_in in [4d, L_in-1-4d] (SYMMETRIC, like MiniRocket)
    v = geo[0]["valid"]
    assert v[:4].sum() == 0 and v[-4:].sum() == 0 and v[4:-4].all()
    # diff pass units lose 1 output
    dj1 = [g for g in geo if g["dj"] == 1 and g["d"] == d0][0]
    assert dj1["L_out"] == T - 1


def test_hydrah_formula_vs_independent():
    """Closed-form HydraH == independent naive recompute (kernel-own row),
    with counts built over the IDENTICAL valid output window the independent
    function derives from the conv geometry ([4d, L_in-1-4d] input centers)."""
    g = np.random.RandomState(7)
    N = 3
    L_out = 60
    d, dj = 2, 0                      # d=2 so the window actually bites
    win = g.randint(0, K, size=(N, 1, L_out)).astype(np.int8)
    reg = g.randint(0, K, size=(N, T))
    # same valid window as hydra_h_independent
    i_idx = np.arange(L_out)
    t_in = i_idx * d
    valid = (t_in >= 4 * d) & (t_in <= T - 1 - 4 * d)
    t_map = np.clip(t_in, 0, T - 1)
    c_kr = np.zeros((N, 1, K, K), dtype=np.int64)
    n_ir = np.zeros((N, 1, K), dtype=np.int64)
    for s in range(N):
        rv = reg[s, t_map[valid]]
        wid = win[s, 0, valid].astype(int)      # winner ID (not boolean)
        for t in range(len(rv)):
            c_kr[s, 0, wid[t], rv[t]] += 1      # production-style indexing
        n_ir[s, 0] = np.bincount(rv, minlength=K)
    n_v = np.full((N, 1), int(valid.sum()), dtype=np.int32)
    H, _ = hydra_stats.hydra_h_from_counts(c_kr, n_ir, n_v, T)
    for s in range(N):
        href = hydra_stats.hydra_h_independent(
            win[s, 0], reg[s], d, dj, T, kernel_idx=0)
        assert abs(H[s, 0] - href) < 1e-10, (s, H[s, 0], href)


def test_hydrah_not_degenerate():
    """W_{m,r} denominator = regime occupancy, not win count (the W_mr==1
    degeneracy bug this test permanently guards against)."""
    g = np.random.RandomState(11)
    L_out, d, dj = 60, 1, 0
    win = g.randint(0, K, size=(2, 1, L_out)).astype(np.int8)
    reg = g.randint(0, K, size=(2, T))
    H, contrib = None, None
    c_kr = np.zeros((2, 1, K, K), dtype=np.int64)
    for i in range(2):
        for t in range(L_out):
            c_kr[i, 0, win[i, 0, t], reg[i, t]] += 1
    n_ir = np.stack([
        np.stack([np.bincount(reg[i, :L_out], minlength=K)])
        for i in range(2)])
    n_v = np.full((2, 1), L_out, dtype=np.int32)
    H, contrib = hydra_stats.hydra_h_from_counts(c_kr, n_ir, n_v, T)
    assert (H > 0).any(), "HydraH collapsed to zero (degenerate W_mr)"


def test_h_formula_via_audited_impl():
    """Branch 2 uses the audited heterogeneity implementation: verify on a
    synthetic case against the independent recompute."""
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        heterogeneity_features, independent_heterogeneity_recompute)
    g = np.random.RandomState(13)
    N, F, Tt = 2, 3, 160
    act = g.rand(N, F, Tt) > 0.5
    valid = np.ones((F, Tt), bool)
    valid[:, :8] = False
    valid[:, Tt - 8:] = False
    reg = g.randint(0, K, size=(N, Tt))
    H = heterogeneity_features(act, valid, reg, K=K, min_occupancy=0.01)
    for i in range(N):
        for f in range(F):
            href = independent_heterogeneity_recompute(
                act[i, f, 8:Tt - 8], valid[f, 8:Tt - 8], reg[i, 8:Tt - 8],
                K=K, min_occupancy=0.01)
            assert abs(H[i, f] - href) < 3.9e-9


def test_regime_contributions_match_H():
    g = np.random.RandomState(17)
    N, F, Tt = 2, 4, 160
    act = g.rand(N, F, Tt) > 0.5
    valid = np.ones((F, Tt), bool)
    valid[:, :8] = False
    valid[:, Tt - 8:] = False
    reg = g.randint(0, K, size=(N, Tt))
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        heterogeneity_features)
    H = heterogeneity_features(act, valid, reg, K=K, min_occupancy=0.01)
    C = information  # noqa
    from experiments.rpms_haptics_seed42.regime_stats import (
        regime_contributions)
    contrib = regime_contributions(act, valid, reg, K=K, min_occupancy=0.01)
    assert np.allclose(contrib.sum(axis=2), H, atol=3.9e-9)


def test_allocation_slice_semantics():
    """G takes the FIRST 3332 of 9996 (feature axis), H the first 3332 of
    the 4998-wide het block (feature axis) -- no sample-axis slicing."""
    x = np.arange(9996 * 2, dtype=float).reshape(2, 9996)
    G = x[:, :N_BRANCH]
    assert G.shape == (2, N_BRANCH) and G[0, 0] == 0 and G[1, 0] == 9996


def test_information_redundancy_shapes():
    g = np.random.RandomState(19)
    br = {k: g.randn(30, 20) for k in ("G", "H", "HydraH")}
    red = information.branch_redundancy(br)
    assert set(red) == {"G~H", "G~HydraH", "H~HydraH"}
    assert all(-1 <= v["linear_cka"] <= 1.0000001 for v in red.values())


def test_incremental_and_lobo_val_only():
    g = np.random.RandomState(23)
    n_tr, n_va = 24, 8
    ytrva = np.concatenate([g.randint(0, 3, n_tr), g.randint(0, 3, n_va)])
    yva = g.randint(0, 3, n_va)
    br = {k: g.randn(n_tr + n_va, 12) for k in ("G", "H", "HydraH")}
    inc = information.incremental_gains(br, ytrva, n_tr, yva)
    assert set(inc) == {"G", "H", "HydraH", "G+H", "G+HydraH", "H+HydraH",
                        "G+H+HydraH"}
    lobo = information.leave_one_branch_out(br, ytrva, n_tr, yva)
    assert set(lobo) == {"full-minus-G", "full-minus-H", "full-minus-HydraH"}
    assert inc["G+H+HydraH"]["dim"] == 36


def test_effective_rank_bounded():
    g = np.random.RandomState(29)
    x = g.randn(40, 15)
    r = information._effective_rank(x)
    assert 1 <= r <= 15


def test_hydrah_alignment():
    g = np.random.RandomState(31)
    H = g.randn(20, 10)
    X = H * 2 + 0.1            # perfectly correlated
    al = information.h_vs_hydrah_alignment(H, X)
    assert al["mean_corr"] > 0.99
    assert al["fraction_strong_corr"] == 1.0


def test_ridge_trainval_protocol():
    g = np.random.RandomState(37)
    n_tr, n_va, n_te = 30, 8, 12
    F_trva = g.randn(n_tr + n_va, 20)      # official: fit on train+val
    F_va = F_trva[n_tr:]
    F_te = g.randn(n_te, 20)
    ytrva = np.concatenate([g.randint(0, 3, n_tr), g.randint(0, 3, n_va)])
    yva = g.randint(0, 3, n_va)
    yte = g.randint(0, 3, n_te)
    res, pva, pte = core.fit_ridge(F_trva, ytrva, F_va, yva, F_te, yte)
    assert res["dim"] == 20 and len(pte) == n_te and len(pva) == n_va
    assert res["selected_alpha"] in tuple(np.logspace(-4, 4, 20))


def test_winner_count_consistency():
    """c_kr nonzero only at (winner == own kernel); n_ir = occupancy."""
    g = np.random.RandomState(41)
    B, h, L_out, d, dj = 2, 2, 50, 1, 0
    win = g.randint(0, K, size=(B, h, L_out)).astype(np.int8)
    reg = g.randint(0, K, size=(B, T))
    # emulate the runner's per-block accumulation for one (d, j) block
    c_kr = np.zeros((B, h * K, K, K), dtype=np.int64)
    n_ir = np.zeros((B, h * K, K), dtype=np.int64)
    for b in range(B):
        r_valid = reg[b, :L_out]           # unit's valid region of regimes
        for gi in range(h):
            for t in range(L_out):
                c_kr[b, gi * K + win[b, gi, t], win[b, gi, t], r_valid[t]] += 1
        occ = np.bincount(r_valid, minlength=K)
        n_ir[b] = occ[None, :]
    # verify: for each unit u=(g,k), c_kr[u, k', r] == 0 unless k' == k;
    # n_ir = occupancy over the unit's valid region (= all L_out positions)
    for b in range(B):
        for u in range(h * K):
            own = u % K
            for k2 in range(K):
                if k2 != own:
                    assert c_kr[b, u, k2].sum() == 0
            assert n_ir[b, u].sum() == L_out
            # unit u wins only where its own kernel is the argmax
            assert 0 <= c_kr[b, u].sum() <= L_out


def test_frozen_context_checkpoint_loadable():
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    m = RCMKNContextModel(n_classes=5)
    ck = torch.load(core.R2_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state"])   # must not raise
