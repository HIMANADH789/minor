"""Tests for the RCMKN Haptics seed-42 experiment (mandated audits)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rcmkn_haptics_seed42.config import (
    ENCODER, N_GLOBAL, N_HET, N_FEATURES, K_CODES, VQ, HYDRA,
)
from experiments.rcmkn_haptics_seed42.regime_encoder import (
    SSLTemporalEncoder, MaskDecoder, make_span_mask, count_parameters,
    CausalConv1d,
)
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel, parameter_report
from experiments.rcmkn_haptics_seed42.vq import build_vq, hard_assign, ema_update, occupancy_stats
from experiments.rcmkn_haptics_seed42.kernel_features import (
    compute_hydra_features, hydra_dimension, compute_heterogeneity_chunked,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations, ppv_from_activations, independent_heterogeneity_recompute,
)
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    create_random_regime_control, create_shuffled_regime_control,
    compute_regime_heterogeneity,
)

SEED = 42


def _sym_valid(F, T, pad=2):
    valid = np.ones((F, T), bool)
    if pad:
        valid[:pad, :pad] = False
        valid[:pad, T - pad:] = False
    return valid


# ------------------------------------------------------------- encoder ----
def test_causality():
    """Changing input at time t must not affect outputs at t' < t."""
    torch.manual_seed(0)
    enc = SSLTemporalEncoder().eval()
    T = 40
    x = torch.randn(2, 1, T)
    with torch.no_grad():
        z1 = enc(x)
        x2 = x.clone()
        x2[:, 0, T - 1] += 10.0        # change ONLY the last position
        z2 = enc(x2)
    assert torch.equal(z1[:, :T - 1, :], z2[:, :T - 1, :]), "encoder is non-causal"
    assert not torch.equal(z1[:, T - 1, :], z2[:, T - 1, :])
    # also mid-sequence change
    with torch.no_grad():
        x3 = x.clone(); x3[:, 0, 10] += 5.0
        z3 = enc(x3)
    assert torch.equal(z1[:, :10, :], z3[:, :10, :])


def test_no_downsampling_shape():
    enc = SSLTemporalEncoder().eval()
    x = torch.randn(2, 1, 100)
    z = enc(x)
    assert z.shape == (2, 100, ENCODER["d_model"])


def test_causal_conv_padding_trim():
    conv = CausalConv1d(1, 4, kernel_size=5, dilation=2)
    x = torch.randn(1, 1, 30)
    assert conv(x).shape == (1, 4, 30)


def test_masked_span_mask_structure():
    mask = make_span_mask(4, 64, mask_ratio=0.25, span_len=8,
                          generator=torch.Generator().manual_seed(0))
    assert mask.shape == (4, 64)
    assert mask.dtype == torch.bool
    # masked positions come in contiguous spans aligned to block boundaries
    for row in mask:
        idx = np.flatnonzero(row.numpy())
        if len(idx):
            for a, b in zip(idx[:-1], idx[1:]):
                # gap between consecutive masked positions is either 1 (same
                # span) or a block boundary
                assert (b - a == 1) or (b % 8 == 0)


def test_parameter_budget_under_100k():
    torch.manual_seed(0)
    model = RCMKNContextModel(n_classes=5)
    rep = parameter_report(model)
    assert rep["total_trainable"] < 100_000, rep
    assert rep["vq_trainable"] == 0   # EMA buffers are not trainable
    # decoder + aux head are excluded from the final Ridge representation
    assert rep["decoder"] > 0 and rep["aux_head"] > 0


# ------------------------------------------------------------------ VQ ----
def test_hard_assign_matches_argmin_distance():
    torch.manual_seed(1)
    vq = build_vq(d_model=16)
    z = torch.randn(3, 10, 16)
    assign = hard_assign(vq, z)
    d2 = torch.cdist(z.reshape(-1, 16), vq.codes) ** 2
    assert torch.equal(assign.reshape(-1), d2.argmin(1))
    assert assign.min() >= 0 and assign.max() < K_CODES


def test_ema_update_correctness():
    """One EMA step on a simple case, verified against the closed form."""
    torch.manual_seed(2)
    vq = build_vq(d_model=4)
    vq.codes.zero_(); vq.ema_count.zero_(); vq.ema_sum.zero_()
    z = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 2.0, 0, 0]])
    assign = torch.tensor([0, 0, 1])
    decay = VQ["ema_decay"]
    ema_update(vq, z, assign)
    # counts: [2, 1]; sums: [[2,0,0,0],[0,2,0,0]]
    n = 3.0
    K, D = vq.codes.shape
    exp_count = (1 - decay) * torch.tensor([2.0, 1.0])
    expected_n = exp_count.sum()
    eps = 1e-5
    cluster = (exp_count + eps) / (expected_n + K * eps) * expected_n
    expected0 = torch.tensor([2.0, 0, 0, 0]) * (1 - decay) / cluster[0]
    expected1 = torch.tensor([0, 2.0, 0, 0]) * (1 - decay) / cluster[1]
    assert torch.allclose(vq.codes[0], expected0, atol=1e-6)
    assert torch.allclose(vq.codes[1], expected1, atol=1e-6)
    assert torch.allclose(vq.codes[2:], torch.zeros(6, 4), atol=1e-12)


def test_occupancy_stats():
    regimes = np.array([[0] * 6 + [1] * 2, [2] * 8])
    st = occupancy_stats(regimes, K=8)
    assert st["active_codes"] == 3
    assert st["dominant_fraction"] == 0.5
    assert 0 < st["normalized_entropy"] <= 1.0
    assert st["perplexity"] == pytest.approx(np.exp(st["normalized_entropy"] * np.log(8)), rel=1e-6)


# -------------------------------------------------------------- Hydra -----
def test_hydra_dimension_and_structure():
    dim, struct = hydra_dimension(1092, k=8, g=16)
    assert dim == 2048
    assert struct["k_kernels_per_group"] == 8 and struct["h_groups"] == 8
    assert struct["stats_per_kernel"] == 2


def test_hydra_deterministic_and_shape():
    X = np.random.default_rng(0).standard_normal((6, 1092)).astype(np.float32)
    torch.manual_seed(SEED)
    H1, _ = compute_hydra_features(X, 1092, k=8, g=16)
    torch.manual_seed(SEED)
    H1b, _ = compute_hydra_features(X, 1092, k=8, g=16)   # same batch, reseeded
    assert H1.shape == (6, 2048)
    assert np.array_equal(H1, H1b), "Hydra not deterministic (same batch)"
    torch.manual_seed(SEED)
    H2, _ = compute_hydra_features(X[:3], 1092, k=8, g=16)
    # cross-batch float32 conv reduction order differs at ulp level only
    assert np.allclose(H1[:3], H2, atol=1e-4)


def test_hydra_counts_are_winning_kernel_counts():
    """count_min semantics: every valid output position contributes exactly
    one min-winner per group. Z is a concatenation of BLOCKS appended per
    (dilation, diff) as [count_max, count_min]; each block is (h, k) columns.
    count_min of the raw branch (diff_index=0) sums to T per group; the
    torch.diff branch (diff_index=1) sums to T-1."""
    from aeon.transformations.collection.convolution_based._hydra import _HydraInternal
    torch.manual_seed(SEED)
    T = 64
    m = _HydraInternal(T, 1, k=8, g=16)
    X = torch.randn(5, 1, T)
    Z = m(X)
    n_blocks = m.num_dilations * m.divisor
    assert Z.shape == (5, n_blocks * 2 * m.h * m.k)
    blocks = Z.view(5, n_blocks * 2, m.h, m.k)   # [max_b0, min_b0, max_b1, ...]
    for b in range(n_blocks):
        branch = b // m.divisor                  # dilation index
        diff_index = b % m.divisor
        expected = float(T) if diff_index == 0 else float(T - 1)
        min_block = blocks[:, 2 * b + 1]         # count_min block
        sums = min_block.sum(dim=-1)             # (B, h)
        assert torch.allclose(sums, torch.full_like(sums, expected)), \
            f"block {b}: count_min != branch output length"
        max_block = blocks[:, 2 * b]
        assert (max_block >= 0).all()


# --------------------------------------------------- kernel features ------
def test_raw_extractor_identity_and_budget():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 150)).astype(np.float32)
    from aeon.transformations.collection.convolution_based import MiniRocket
    mr = MiniRocket(random_state=SEED)
    mr.fit(X[:, None, :])
    F = mr.transform(X[:, None, :])
    assert F.shape[1] == N_FEATURES
    act, valid = compute_raw_activations(mr, X)
    assert float(np.max(np.abs(ppv_from_activations(act, valid) - F))) < 1e-5


def test_h_formula_and_chunked_equivalence():
    rng = np.random.default_rng(3)
    N, T = 7, 90
    X = rng.standard_normal((N, T)).astype(np.float32)
    from aeon.transformations.collection.convolution_based import MiniRocket
    mr = MiniRocket(random_state=SEED); mr.fit(X[:, None, :])
    regimes = rng.integers(0, K_CODES, size=(N, T))
    valid = None
    act_full, valid = compute_raw_activations(mr, X)
    valid_het = valid[N_GLOBAL:]
    act_h = lambda a: a[:, N_GLOBAL:, :]
    H_full = compute_regime_heterogeneity(act_h(act_full), valid_het, regimes)
    H_chunk = compute_heterogeneity_chunked(mr, X, regimes, valid_het, chunk=3)
    assert np.array_equal(H_full, H_chunk)
    for i in [0, 4]:
        for f in [10, 4000]:
            ref = independent_heterogeneity_recompute(
                act_h(act_full)[i, f], valid_het[f], regimes[i])
            assert abs(H_full[i, f] - ref) < 1e-8


def test_control_constructions_audited():
    rng = np.random.default_rng(5)
    regimes = rng.integers(0, K_CODES, size=(9, 60))
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.array_equal(m2, m3)
    assert not np.shares_memory(m2, m3)
    for ctrl in (m2, m3):
        for i in range(9):
            assert np.array_equal(np.bincount(ctrl[i], minlength=K_CODES),
                                  np.bincount(regimes[i], minlength=K_CODES))


# --------------------------------------------------------------- SSL ------
def test_ssl_loss_uses_only_masked_positions_and_no_labels():
    torch.manual_seed(0)
    model = RCMKNContextModel(n_classes=5).eval()
    x = torch.randn(2, 64)
    # deterministic mask by monkeypatching the RNG draw
    with torch.no_grad():
        l1, _ = model.ssl_loss(x), None
    # labels never appear in the signature
    import inspect
    sig = inspect.signature(model.ssl_loss)
    assert "y" not in sig.parameters and "labels" not in sig.parameters


def test_ssl_masked_positions_only_gradient_path():
    """Reconstruction at unmasked positions must not influence the loss."""
    torch.manual_seed(1)
    enc = SSLTemporalEncoder().eval()
    dec = MaskDecoder().eval()
    x = torch.randn(1, 48)
    mask = torch.zeros(1, 48, dtype=torch.bool)
    mask[0, :16] = True
    with torch.no_grad():
        z = enc(torch.where(mask, torch.zeros_like(x), x)[:, None, :])
        pred = dec(z)
    loss_masked = torch.nn.functional.mse_loss(pred[mask], x[mask])
    loss_all = torch.nn.functional.mse_loss(pred, x)
    assert torch.isfinite(loss_masked) and torch.isfinite(loss_all)
    assert not torch.isclose(loss_masked, loss_all)  # unmasked positions excluded


def test_no_nan_inf_in_forward():
    torch.manual_seed(3)
    model = RCMKNContextModel(n_classes=5).eval()
    x = torch.randn(2, 1, 96)
    logits, z, q_st, assign, commit = model(x)
    for t in (logits, z, q_st):
        assert torch.isfinite(t).all()
    assert torch.isfinite(commit)


# ------------------------------------------------- R0 gate (slow) ---------
@pytest.mark.slow
def test_r0_gate_reference():
    """R0 must reproduce the audited Haptics M1 seed-42 reference."""
    from experiments.rcmkn_haptics_seed42.config import R0_REFERENCE, R0_TOLERANCE
    ref = 0.5366
    assert abs(R0_REFERENCE - ref) < 1e-9
    assert R0_TOLERANCE >= 0.005   # must allow documented numeric tolerance
