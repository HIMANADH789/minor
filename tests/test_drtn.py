"""
Unit tests for DRTN (spec sec. 26, 10 tests). All must pass before the
official Haptics R0-R5 run.
"""
import copy
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.drtn.model import (  # noqa: E402
    CausalConv1d, CausalEncoder, HardVQ, SoftCodebook, TrajectoryTransformer,
    build_model, count_params, usage_stats,
)

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# 1. causal convolution: future inputs must not alter earlier outputs
# ---------------------------------------------------------------------------
def test_causal_conv_no_future_leakage():
    torch.manual_seed(0)
    conv = CausalConv1d(1, 4, kernel_size=3, dilation=4)
    x = torch.randn(1, 1, 50)
    y1 = conv(x)
    x2 = x.clone()
    x2[:, :, 30:] = 999.0                       # mutate only the future
    y2 = conv(x2)
    assert torch.allclose(y1[:, :, :30], y2[:, :, :30], atol=1e-6)
    assert not torch.allclose(y1[:, :, 30:], y2[:, :, 30:])


def test_encoder_preserves_length_and_causality():
    torch.manual_seed(0)
    enc = CausalEncoder()
    x = torch.randn(2, 1, 128)
    z = enc(x)
    assert z.shape == (2, 128, 64)              # no downsampling
    x2 = x.clone()
    x2[:, :, 64:] = 500.0
    z2 = enc(x2)
    assert torch.allclose(z[:, :64], z2[:, :64], atol=1e-5)


# ---------------------------------------------------------------------------
# 2. VQ: every timestep has exactly one selected code
# ---------------------------------------------------------------------------
def test_vq_single_assignment_per_timestep():
    torch.manual_seed(0)
    vq = HardVQ(64, n_codes=8)
    z = torch.randn(3, 20, 64)
    q_st, assign, commit = vq.quantize(z)
    assert assign.shape == (3, 20)
    assert assign.min() >= 0 and assign.max() < 8
    assert assign.dtype == torch.int64


# ---------------------------------------------------------------------------
# 3. straight-through: forward uses code embedding, gradient reaches encoder
# ---------------------------------------------------------------------------
def test_straight_through_gradients():
    torch.manual_seed(0)
    vq = HardVQ(64, n_codes=8)
    z = torch.randn(2, 10, 64, requires_grad=True)
    q_st, assign, commit = vq.quantize(z)
    assert torch.allclose(q_st, z + (vq.codes[assign.reshape(-1)]
                                     .reshape(z.shape) - z).detach())
    q_st.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    # codebook itself gets NO gradient from this path (EMA updates it)
    assert not vq.codes.requires_grad


# ---------------------------------------------------------------------------
# 4. EMA codebook updates correctly
# ---------------------------------------------------------------------------
def test_ema_update_math():
    torch.manual_seed(0)
    vq = HardVQ(4, n_codes=2, ema_decay=0.5)
    vq.ema_count.zero_()
    vq.ema_sum.zero_()
    z = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [3.0, 0, 0, 0]])
    assign = torch.tensor([0, 1, 0])
    vq.ema_step(z, assign, step=0)
    # counts: decay*0 + 0.5*[2,1] = [1, .5]
    assert torch.allclose(vq.ema_count, torch.tensor([1.0, 0.5]))
    # sums: 0.5*[[4,0,0,0],[0,1,0,0]]
    assert torch.allclose(vq.ema_sum[0], torch.tensor([2.0, 0, 0, 0]))
    # c_k = m_k / n_k = mean of assigned latents
    assert torch.allclose(vq.codes[0], torch.tensor([2.0, 0, 0, 0]))
    assert torch.allclose(vq.codes[1], torch.tensor([0, 1.0, 0, 0]))


# ---------------------------------------------------------------------------
# 5. dead-code revival actually reinitializes
# ---------------------------------------------------------------------------
def test_dead_code_revival():
    torch.manual_seed(0)
    vq = HardVQ(8, n_codes=4, dead_threshold=1e-3, revival_patience=3)
    # force code 0 to be the nearest to z=[1,0,...]; codes 1-3 get zero mass
    with torch.no_grad():
        vq.codes.zero_()
        vq.codes[0, 0] = 0.5
    z = torch.zeros(8, 8)
    z[:, 0] = 1.0
    before = vq.codes[3].clone()
    revived_all = []
    for step in range(3):
        revived_all += vq.ema_step(z, torch.zeros(8, dtype=torch.long),
                                   step=step)
    # patience 3: codes 1-3 hit the threshold on the 3rd update and are revived
    assert 3 in revived_all and vq.total_revivals == 3
    assert len(vq.revival_log) == 3
    entry = [e for e in vq.revival_log if e["code"] == 3][0]
    assert entry["source"] == "current_batch_latent"
    assert entry["prev_usage"] == 0.0
    # revived code sits ON a batch latent (code 0 direction = [1,0,...])
    assert torch.allclose(vq.codes[3], z[0])
    assert not torch.allclose(before, vq.codes[3])
    # patience counter was reset by the revival
    assert vq.dead_steps[3] == 0


# ---------------------------------------------------------------------------
# 6. diversity: entropy from AGGREGATE batch usage, not per-timestep
# ---------------------------------------------------------------------------
def test_diversity_is_population_level():
    torch.manual_seed(0)
    m = build_model("R4", n_codes=8)
    # each TIMESTEP is one-hot (instance level); batch histograms differ
    B, T, K = 2, 96, 8                            # T divisible by K -> exactly uniform
    assign_uniform = torch.arange(T)[None].repeat(B, 1) % K
    assign_degenerate = torch.zeros(B, T, dtype=torch.long)
    l_u = m.diversity_loss_from_assign(assign_uniform)
    l_d = m.diversity_loss_from_assign(assign_degenerate)
    # per-timestep entropies are identical (0) in both — only the population
    # histogram differs, proving the loss is aggregate-level
    assert l_d.item() > l_u.item() + 1.0
    # uniform histogram: H = log K, so L_div = -log K
    assert abs(l_u.item() + math.log(K)) < 1e-4


# ---------------------------------------------------------------------------
# 7. no label leakage: diversity never touches labels
# ---------------------------------------------------------------------------
def test_diversity_label_independent():
    torch.manual_seed(0)
    m = build_model("R4", n_codes=8)
    x = torch.randn(4, 1, 60)
    y_a = torch.tensor([0, 1, 2, 3])
    y_b = torch.tensor([1, 1, 1, 1])
    q_st, assign, commit = m.vq_forward(x)
    logits = m.classifier(m.pool_and_head(q_st))
    _, parts_a = m.loss_terms(logits, y_a, assign, commit)
    _, parts_b = m.loss_terms(logits, y_b, assign, commit)
    # diversity is identical under different labels
    assert parts_a["div"].item() == parts_b["div"].item()
    # and is a pure function of the assignments
    l_again = m.diversity_loss_from_assign(assign)
    assert l_again.item() == parts_a["div"].item()


# ---------------------------------------------------------------------------
# 8. trajectory transformer: causal mask blocks future attention
# ---------------------------------------------------------------------------
def test_transformer_causal():
    torch.manual_seed(0)
    tt = TrajectoryTransformer(d_model=64, n_layers=1)
    tt.eval()                                    # disable dropout
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        h1 = tt(x)
        x2 = x.clone()
        x2[:, 8:] = 50.0
        h2 = tt(x2)
    assert torch.allclose(h1[:, :8], h2[:, :8], atol=1e-5)


# ---------------------------------------------------------------------------
# 9. sequence length preserved through every full model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rung", ["R0", "R1", "R2", "R3", "R4", "R5"])
def test_all_models_end_to_end(rung):
    torch.manual_seed(42)
    m = build_model(rung, n_classes=5)
    x = torch.randn(2, 1, 1092)                  # actual Haptics length
    logits, extra = m.forward_with_assign(x)
    assert logits.shape == (2, 5)
    if rung in ("R3", "R4", "R5"):
        assert extra["assign"].shape == (2, 1092)
        commit = m.vq.quantize(m.encoder(x))[2]
        assert torch.isfinite(commit)
    # backward through the full loss path
    y = torch.tensor([0, 1])
    if hasattr(m, "loss_terms"):
        q_st, assign, commit = m.vq_forward(x)
        logits = m.classifier(m.pool_and_head(q_st))
        total, parts = m.loss_terms(logits, y, assign, commit)
    else:
        logits = m(x)
        total = F.cross_entropy(logits, y)
    total.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# 10. determinism: repeated seed-42 init reproduces identical behavior
# ---------------------------------------------------------------------------
def test_seed42_determinism():
    def run():
        torch.manual_seed(42)
        np.random.seed(42)
        m = build_model("R4", n_classes=5)
        x = torch.randn(3, 1, 200)
        with torch.no_grad():
            q_st, assign, _ = m.vq_forward(x)
            logits = m.classifier(m.pool_and_head(q_st))
        return logits.numpy().copy(), assign.numpy().copy(), \
            copy.deepcopy(m.vq.codes.numpy())

    l1, a1, c1 = run()
    l2, a2, c2 = run()
    assert np.array_equal(l1, l2)
    assert np.array_equal(a1, a2)
    assert np.array_equal(c1, c2)


# ---------------------------------------------------------------------------
# bonus: usage stats math + VQ diag helper
# ---------------------------------------------------------------------------
def test_usage_stats_math():
    q = np.array([0.5, 0.25, 0.25, 0.0])
    s = usage_stats(q)
    H = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert abs(s["entropy"] - H) < 1e-9
    assert abs(s["perplexity"] - math.exp(H)) < 1e-9
    assert s["active_codes"] == 3
    assert s["dominant_fraction"] == 0.5
    assert abs(s["normalized_entropy"] - H / math.log(4)) < 1e-9


def test_soft_codebook_shapes_and_diag():
    torch.manual_seed(0)
    cb = SoftCodebook(64, n_codes=8, tau=0.5)
    z = torch.randn(2, 30, 64)
    r, a = cb(z)
    assert r.shape == z.shape and a.shape == (2, 30, 8)
    row = a[0, 0]
    assert abs(row.sum().item() - 1.0) < 1e-5     # soft assignment row
    assert (row > 0).all()                        # continuous: all codes present
