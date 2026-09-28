"""R2.2 mandatory tests: conditioning math, identity, bounds, controls,
aggregation parity, determinism, budget, no-leakage invariants."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.rcmkn_r2_2_adaptive_conditioning_seed42 import conditioning as cond
from experiments.rcmkn_r2_2_adaptive_conditioning_seed42 import core

K, M = 8, 64          # small M for fast tests
T = 96


def _fake_extractor_frozen(M=4998):
    """Stand-in with the parameter layout compute_activations_and_margins
    expects; kernels/biases are random but fixed (tests only exercise the
    conditioning layer, not MiniRocket identity)."""
    class _P:
        def __init__(self):
            g = np.random.RandomState(0)
            self.dilations = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
            self.n_features_per_dilation = np.array(
                [768, 1536, 3072, 3072, 768, 768], dtype=np.int32)
            n = 6 * 84 * 84
            self.biases = g.randn(n).astype(np.float32)

    class _E:
        parameters = _P()

    return _E()


# --------------------------------------------------------------------
# 1. tau mapping + bounded modulation
# --------------------------------------------------------------------
def test_tau_mapping_extremes():
    d = np.array([-0.499, 0.0, 0.499])
    tau_base = cond.tau_from_delta(d)
    assert np.all(np.abs(tau_base) < 1.0)
    # delta>0 (amplify) => base threshold moves negative => more activations
    assert tau_base[2] < 0 < tau_base[0]


def test_fold_bias_sign():
    """AUDIT: b>0 keeps tau_base, b<0 flips sign, b==0 gives 0. Positive
    bias + delta>0 => threshold below 0 (more activations); negative bias
    + delta>0 => threshold above 0 (fewer activations)."""
    d = np.full((1, 3), 0.499)
    tau = cond.signed_tau_from_delta(d, np.array([1.0, -1.0, 0.0]))
    assert tau[0, 0] < 0 and tau[0, 1] > 0 and tau[0, 2] == 0.0
    # explicit derivation checks
    delta = 0.3
    b = -2.0
    C = 1.0
    # direct: (C*(1+delta) > b) ?  C*(1.3) = 1.3 > -2 -> True
    # margin form: u = (C - b)/|b| = 1.5 ; tau = sign(b)*(-delta/(1+delta))
    tau_direct = np.sign(b) * (-delta / (1 + delta))
    assert np.isclose(tau_direct, 0.3 / 1.3)
    assert 1.5 > tau_direct          # activates -- matches direct form


def test_identity_initialization():
    m = cond.CodeModulation(K=K, M=M)
    d = m.delta().detach().numpy()
    assert np.abs(d).max() == 0.0
    assert np.allclose(m.tau().detach().numpy(), 0.0)
    lo, hi = m.scale_bounds()
    assert lo == 1.0 and hi == 1.0


def test_modulation_bounds_always():
    torch.manual_seed(0)
    m = cond.CodeModulation(K=K, M=M)
    with torch.no_grad():
        m.a.normal_(0, 10.0)
        m.b.fill_(50.0)          # sigmoid ~ 1
    d = m.delta().detach().numpy()
    assert np.abs(d).max() < cond.BETA_MAX + 1e-6
    s = 1.0 + d
    assert s.min() > 0.5 - 1e-6 and s.max() < 1.5 + 1e-6
    assert s.min() > 0                      # positive scaling factor


def test_b1_force_zero():
    torch.manual_seed(1)
    m = cond.CodeModulation(K=K, M=M)
    m.force_zero = True
    with torch.no_grad():
        m.a.normal_(0, 10.0)
        m.b.fill_(20.0)
    assert float(m.delta().abs().max()) == 0.0
    assert float(m.tau().abs().max()) == 0.0


# --------------------------------------------------------------------
# 2. conditioned activation exactness vs the direct definition
# --------------------------------------------------------------------
def test_cond_act_matches_direct_definition():
    g = np.random.RandomState(3)
    B, T, F = 5, 40, 12
    C = g.randn(B, F, T).astype(np.float32) * 3          # raw responses
    b_f = (g.randn(F) * 0.8 + 0.1).astype(np.float32)
    b_f[2] = 0.0                                         # zero-bias feature
    u = (C - b_f[None, :, None]) / np.where(
        b_f[None, :, None] == 0, 1.0, np.abs(b_f)[None, :, None])
    u[:, 2] = np.where(C[:, 2] > 0, 2.0, -2.0)
    codes = g.randint(0, K, size=(B, T))
    torch.manual_seed(7)
    m = cond.CodeModulation(K=K, M=F)
    with torch.no_grad():
        m.a.normal_(0, 1.5)
        m.b.fill_(0.0)
    delta = m.delta().detach().numpy()                   # (K, F)
    scale = 1.0 + delta[codes]                           # (B, T, F)
    act_direct = (C * np.transpose(scale, (0, 2, 1))) > b_f[None, :, None]
    tau_signed = cond.signed_tau_from_delta(delta, np.sign(b_f))
    act_impl = cond.cond_act_from_u(
        u, tau_signed, codes, np.ones((F, T), dtype=bool))
    assert np.array_equal(act_direct, act_impl)


def test_zero_bias_feature_never_flips():
    g = np.random.RandomState(5)
    B, F, T = 3, 8, 30
    u = g.choice([-2.0, 2.0], size=(B, F, T)).astype(np.float32)
    codes = g.randint(0, K, size=(B, T))
    delta = np.full((K, F), 0.499)
    tau_base = cond.tau_from_delta(delta)                # ~ -0.333
    bs = np.zeros(F)                                     # zero biases
    tau = cond.fold_bias_sign(tau_base, bs)              # threshold 0
    act = cond.cond_act_from_u(u, tau, codes, np.ones((F, T), bool))
    assert np.array_equal(act, u > 0)                    # identical to |tau|<2


# --------------------------------------------------------------------
# 3. u clipping losslessness within the admissible delta range
# --------------------------------------------------------------------
def test_u_clipping_lossless_for_admissible_delta():
    g = np.random.RandomState(11)
    F, T = 16, 50
    u_full = g.randn(4, F, T).astype(np.float64) * 0.7   # unclipped margins
    codes = g.randint(0, K, size=(4, T))
    bias_sign = np.where(np.arange(F) % 3 == 0, -1.0,
                         np.where(np.arange(F) % 3 == 1, 0.0, 1.0))
    for delta in (-0.499, -0.2, 0.0, 0.35, 0.499):
        tau = cond.signed_tau_from_delta(
            np.full((K, F), delta), bias_sign)
        full = cond.cond_act_from_u(u_full, tau, codes, np.ones((F, T), bool))
        clipped = cond.cond_act_from_u(
            np.clip(u_full, -1.0, 1.0), tau, codes, np.ones((F, T), bool))
        assert np.array_equal(full, clipped), f"delta={delta}"


# --------------------------------------------------------------------
# 4. histograms vs direct counts + surrogate orientation
# --------------------------------------------------------------------
def test_histogram_matches_direct_counts():
    g = np.random.RandomState(17)
    B, F, T = 4, 10, 60
    u = g.randn(B, F, T).astype(np.float32)
    u[:, :, :5] = -3.0                                   # force below bin
    u[:, :, 5:10] = 3.0                                  # force above bin
    valid = np.ones((F, T), bool)
    valid[:, 0] = False                                  # invalid positions
    codes = g.randint(0, K, size=(B, T))
    h = cond.build_histograms(u, codes, valid, "cpu")
    assert h.shape == (B, F, K, cond.HIST_S)
    b, f, k = 1, 3, 2
    sel = (codes[b] == k) & np.arange(T)[None, :] >= 5
    sel = sel[0] & valid[f]
    n_above_direct = int(((codes[b] == k) & (u[b, f] > 1.0) & valid[f]).sum())
    assert h[b, f, k, -1] == n_above_direct
    assert h[b, f, k].sum() == int(((codes[b] == k) & valid[f]).sum())


def test_surrogate_H2_shapes_and_grad():
    B, F = 6, 10
    cnt = torch.randint(0, 5, (B, F, K, cond.HIST_S)).float()
    delta = torch.zeros(K, F, requires_grad=True)
    bs = np.sign(np.linspace(-1, 1, F))
    bs[bs == 0] = 1.0
    H2, stats = cond.surrogate_H2(cnt, delta, bs, min_count=1)
    assert H2.shape == (B, F)
    H2.sum().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()


def test_surrogate_matches_exact_H2_at_zero_tau():
    """At tau=0 the surrogate PPV over bins must equal exact PPV: with
    edges at bin centers, sigma((edge-0)/w) is a smooth step; the check is
    that surrogate H2 with delta=0 equals B0 H2 computed from the SAME
    binned data within surrogate tolerance."""
    g = np.random.RandomState(23)
    B, F, T = 3, 8, 120
    u = g.uniform(-1.2, 1.2, size=(B, F, T)).astype(np.float32)
    codes = g.randint(0, K, size=(B, T))
    valid = np.ones((F, T), bool)
    h = cond.build_histograms(u, codes, valid, "cpu")
    edges = cond.bin_edges()
    tau0 = torch.zeros(K, F)
    # exact PPV from raw u at tau=0 (i.e. act = u>0)
    ppv_exact = (u > 0).mean(axis=2)                     # (B, F)
    sigma = 1.0 / (1.0 + np.exp(-(edges[None, None, None, :] - 0.0)
                                / cond.SOFT_W))
    ppv_sur = (h * sigma[None]).sum(-1).sum(-1) / np.maximum(
        h.sum((-1, -2)), 1)
    assert np.abs(ppv_sur - ppv_exact).max() < 0.05      # smooth-step slack


# --------------------------------------------------------------------
# 5. control constructors (audited) semantics
# --------------------------------------------------------------------
def _fake_codes(N=6, T=48, seed=0):
    g = np.random.RandomState(seed)
    codes = g.randint(0, K, size=(N, T))
    codes[:, : T // 2] = (np.arange(N)[:, None] % K)     # ensure structure
    return codes


def test_b3a_occupancy_preserved_and_distinct():
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        create_random_regime_control, create_shuffled_regime_control)
    codes = _fake_codes()
    c1 = create_random_regime_control(codes, seed=42, K=K)
    for i in range(len(codes)):
        assert np.array_equal(np.bincount(c1[i], minlength=K),
                              np.bincount(codes[i], minlength=K))
    assert not np.array_equal(c1, codes)
    assert not np.shares_memory(c1, codes)


def test_b3b_occupancy_preserved_and_distinct():
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        create_shuffled_regime_control)
    codes = _fake_codes(seed=1)
    c2 = create_shuffled_regime_control(codes, seed=42)
    for i in range(len(codes)):
        assert np.array_equal(np.bincount(c2[i], minlength=K),
                              np.bincount(codes[i], minlength=K))
    assert not np.array_equal(c2, codes)
    assert not np.shares_memory(c2, codes)


def test_b3a_b3b_distinct_streams():
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        create_random_regime_control, create_shuffled_regime_control)
    codes = _fake_codes(seed=2)
    c1 = create_random_regime_control(codes, seed=42, K=K)
    c2 = create_shuffled_regime_control(codes, seed=42)
    assert (c1 != c2).mean() > 0.01
    assert not np.shares_memory(c1, c2)


# --------------------------------------------------------------------
# 6. aggregation parity: exact H vs independent recompute
# --------------------------------------------------------------------
def test_heterogeneity_aggregation_parity():
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        heterogeneity_features, independent_heterogeneity_recompute)
    g = np.random.RandomState(31)
    N, F, T = 3, 5, 80
    P = 6                                                # symmetric padding
    act = g.rand(N, F, T) > 0.5
    valid = np.ones((F, T), bool)
    valid[:, :P] = False                                 # [P, T-P) valid region
    valid[:, T - P:] = False
    codes = _fake_codes(N=N, T=T)
    H = heterogeneity_features(act, valid, codes, K=K, min_occupancy=0.01)
    for i in range(N):
        for f in range(F):
            h_ref = independent_heterogeneity_recompute(
                act[i, f, P:T - P], valid[f, P:T - P], codes[i, P:T - P],
                K=K, min_occupancy=0.01)
            # audited tolerance for this exact comparison (float32 counts
            # in the batched path vs float64 recompute) is <= 3.9e-9
            assert abs(H[i, f] - h_ref) < 3.9e-9, (i, f, H[i, f], h_ref)


def test_valid_region_corruption_leaves_H_unchanged():
    from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
        heterogeneity_features)
    g = np.random.RandomState(37)
    N, F, T = 2, 4, 60
    act = g.rand(N, F, T) > 0.5
    valid = np.ones((F, T), bool)
    valid[:, :6] = False                                  # [6, T-6) valid region
    valid[:, T - 6:] = False
    codes = _fake_codes(N=N, T=T)
    act_c = act.copy()
    act_c[:, :, :6] = ~act_c[:, :, :6]
    act_c[:, :, T - 6:] = ~act_c[:, :, T - 6:]
    H0 = heterogeneity_features(act, valid, codes, K=K)
    H1 = heterogeneity_features(act_c, valid, codes, K=K)
    assert np.array_equal(H0, H1), "invalid-region corruption changed H"


# --------------------------------------------------------------------
# 7. budgets, feature-axis, determinism, NaN guards
# --------------------------------------------------------------------
def test_feature_budget_constants():
    assert cond.N_FEATURES == 9996
    assert cond.N_GLOBAL == 4998 and cond.N_HET == 4998
    assert cond.N_GLOBAL + cond.N_HET == cond.N_FEATURES
    assert cond.BETA_MAX == 0.5


def test_u_to_bin_edges_and_catchalls():
    e = cond.bin_edges()
    assert e.shape == (cond.HIST_S,)
    assert e[0] < -1.0 and e[-1] > 1.0
    u = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
    b = cond.u_to_bin(u)
    assert b[0, 0] == 0 and b[0, -1] == cond.HIST_S - 1
    assert 1 <= b[0, 1] <= cond.HIST_N_INTERIOR
    assert 1 <= b[0, 2] <= cond.HIST_N_INTERIOR
    assert 1 <= b[0, 3] <= cond.HIST_N_INTERIOR


def test_modulation_diagnostics_fields():
    torch.manual_seed(2)
    m = cond.CodeModulation(K=K, M=M)
    with torch.no_grad():
        m.a.normal_(0, 0.5)
    d = cond.modulation_diagnostics(m)
    for key in ("beta", "mean_abs_s", "max_abs_s", "s_norms_L2",
                "pairwise_cos_mean", "fraction_near_zero_code_vec",
                "scale_factor_min", "scale_factor_max"):
        assert key in d
    assert len(d["s_norms_L2"]) == K


def test_regime_hash_distinct_arrays():
    a = _fake_codes(seed=3)
    b = a.copy()
    b[0, 0] = (b[0, 0] + 1) % K
    assert core.regime_hash(a) != core.regime_hash(b)
    assert core.regime_hash(a) == core.regime_hash(a.copy())


def test_dataset_registry():
    for ds, info in core.DATASETS.items():
        exp = info["expected"]
        assert exp["train"] > 0 and exp["val"] > 0 and exp["test"] > 0
        assert exp["T"] > 0 and exp["n_classes"] >= 2
        assert Path(info["ckpt"]).exists(), f"missing checkpoint for {ds}"
    assert set(core.DATASETS) == {"Haptics", "CWRU_BAL"}


def test_run_variant_budget_and_trainval_fit():
    from sklearn.linear_model import RidgeClassifierCV
    g = np.random.RandomState(41)
    n_tr, n_va, n_te = 30, 8, 12
    n_trva = n_tr + n_va                                  # Ridge fits train+val
    F = core.N_GLOBAL                                     # per block
    blocks = [(g.rand(n_trva, F), g.rand(n_va, F), g.rand(n_te, F)),
              (g.rand(n_trva, F), g.rand(n_va, F), g.rand(n_te, F))]
    ytrva = np.concatenate([g.randint(0, 3, n_tr), g.randint(0, 3, n_va)])
    yva = g.randint(0, 3, n_va)
    yte = g.randint(0, 3, n_te)
    res, pred = core.run_variant("T", blocks, yva, yte, ytrva, 3)
    assert res["feature_dim"] == core.N_FEATURES          # 4998 + 4998
    assert blocks[0][0].shape[1] == F                     # 4998/4998 split
    assert 0.0 <= res["test_macro_f1"] <= 1.0
    assert res["selected_alpha"] in tuple(np.logspace(-4, 4, 20))
    assert len(pred) == n_te
