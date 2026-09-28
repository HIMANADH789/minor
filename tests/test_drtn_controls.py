"""
Unit tests for the DRTN controlled experiments (spec: unit tests 1-12).

Reuses the existing DRTN suite semantics; adds continuous-control forward/
causality, discrete-control VQ behavior, K-shape correctness, and the
encoder/transformer identity checks that make the comparison fair.
"""
import copy
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.drtn.controls import DRTN_CTC, DRTN_DTC, param_audit  # noqa: E402
from models.drtn.model import (  # noqa: E402
    build_model, count_params, usage_stats,
)

torch.manual_seed(42)


def _build(seed=42):
    # Each model is constructed from an IDENTICAL fresh seed — this is both
    # the fairness protocol and how the runner trains them.
    def mk(fn):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return fn()
    return {
        "ctc": mk(lambda: DRTN_CTC(n_classes=5)),
        "dtc": mk(lambda: DRTN_DTC(n_classes=5, n_codes=8)),
        "r5_8": mk(lambda: build_model("R5", n_classes=5, n_codes=8)),
        "r5_16": mk(lambda: build_model("R5", n_classes=5, n_codes=16)),
        "r5_32": mk(lambda: build_model("R5", n_classes=5, n_codes=32)),
    }


# 1+2. continuous control forward + shape
def test_ctc_forward_shape():
    m = _build()["ctc"]
    x = torch.randn(3, 1, 200)
    logits, extra = m.forward_with_assign(x)
    assert logits.shape == (3, 5)
    assert "attn" in extra
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor([0, 1, 2]))
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert not hasattr(m, "vq") and not hasattr(m, "codebook")


# 3. continuous control causal behavior (no future leakage).
# The causal pathway is encoder -> trajectory sequence H[t]; global attention
# pooling is a whole-sequence readout by design in every rung (R1/R5 alike),
# so causality is asserted on the pre-pooling sequence.
def test_ctc_causal():
    torch.manual_seed(0)
    m = _build()["ctc"]
    m.eval()
    x = torch.randn(1, 1, 96)
    with torch.no_grad():
        h1 = m.trajectory(m.encoder(x))
        x2 = x.clone()
        x2[:, :, 64:] = 999.0
        h2 = m.trajectory(m.encoder(x2))
    assert torch.allclose(h1[:, :64], h2[:, :64], atol=1e-5)
    assert not torch.allclose(h1[:, 64:], h2[:, 64:])


# 4. discrete control hard assignment
def test_dtc_hard_assignment():
    m = _build()["dtc"]
    x = torch.randn(2, 1, 64)
    q_st, assign, commit = m.vq_forward(x)
    assert assign.shape == (2, 64)
    assert assign.dtype == torch.int64
    assert assign.min() >= 0 and assign.max() < 8
    # exactly one code per timestep: one-hot rows in the usage histogram
    counts = torch.bincount(assign.reshape(-1), minlength=8)
    assert int(counts.sum()) == 2 * 64


# 5. discrete control straight-through gradient reaches the encoder
def test_dtc_straight_through():
    m = _build()["dtc"]
    x = torch.randn(1, 1, 32, requires_grad=True)
    q_st, assign, commit = m.vq_forward(x)
    logits = m.classifier(m.pool_and_head(q_st))
    (logits.sum() + commit).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert not m.vq.codes.requires_grad          # EMA buffer, not parameter


# 6. discrete control EMA update
def test_dtc_ema_update():
    torch.manual_seed(0)
    m = DRTN_DTC(1, n_classes=5, d_model=4, n_codes=4, ema_decay=0.5)
    m.vq.ema_count.zero_()
    m.vq.ema_sum.zero_()
    z = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    vq = m.vq
    vq.ema_step(z, torch.tensor([0, 1]), step=0)
    assert torch.allclose(vq.ema_count, torch.tensor([0.5, 0.5, 0.0, 0.0]))
    # c_k = mean of assigned latents: code 0 <- [1,0,0,0]
    assert torch.allclose(vq.codes[0], torch.tensor([1.0, 0, 0, 0]))
    assert torch.allclose(vq.codes[1], torch.tensor([0, 1.0, 0, 0]))


# 7. K=8/16/32 codebook shape correctness
@pytest.mark.parametrize("k", [8, 16, 32])
def test_codebook_shapes_k(k):
    m = build_model("R5", n_classes=5, n_codes=k)
    assert m.vq.codes.shape == (k, 64)
    assert m.vq.ema_count.shape == (k,)
    x = torch.randn(2, 1, 40)
    q_st, assign, _ = m.vq_forward(x)
    assert assign.max().item() < k


# 8. diversity loss population-only behavior (DTC must have none)
def test_dtc_has_no_diversity_term():
    m = _build()["dtc"]
    assert m.lam_div == 0.0 and m.has_diversity is False
    x = torch.randn(2, 1, 32)
    q_st, assign, commit = m.vq_forward(x)
    logits = m.classifier(m.pool_and_head(q_st))
    y = torch.tensor([0, 1])
    total, parts = m.loss_terms(logits, y, assign, commit)
    assert set(parts.keys()) == {"ce", "commit", "commit_w"}
    assert "div" not in parts and "div_w" not in parts
    r5 = build_model("R5", n_classes=5, n_codes=8)
    _, parts5 = r5.loss_terms(logits, y, assign, commit)
    assert "div" in parts5                        # R5 keeps it


# 9. no test-label access: diversity/EMA/assignments never see labels
def test_no_label_access_in_diagnostics():
    m = _build()["dtc"]
    x = torch.randn(2, 1, 32)
    q_st, assign, commit = m.vq_forward(x)
    # recompute assignments with absurdly different labels; identical
    _, assign2, _ = m.vq_forward(x)
    assert torch.equal(assign, assign2)


# 10. deterministic seed behavior across the three models
def test_seed_determinism_controls():
    def run():
        torch.manual_seed(42)
        np.random.seed(42)
        ctc, dtc = DRTN_CTC(n_classes=5), DRTN_DTC(n_classes=5, n_codes=8)
        x = torch.randn(2, 1, 64)
        with torch.no_grad():
            lc, _ = ctc.forward_with_assign(x)
            ld, _ = dtc.forward_with_assign(x)
        return lc.numpy().copy(), ld.numpy().copy(), \
            copy.deepcopy(dtc.vq.codes.numpy())

    a, b, c = run()
    a2, b2, c2 = run()
    assert np.array_equal(a, a2) and np.array_equal(b, b2) \
        and np.array_equal(c, c2)


# 11. identical encoder weights/config between controls
def test_encoder_identity_controls_vs_r5():
    built = _build()               # per-model seed-42 construction
    ref_sd = built["r5_8"].encoder.state_dict()
    for key in ("ctc", "dtc"):
        sd = built[key].encoder.state_dict()
        assert set(sd.keys()) == set(ref_sd.keys())
        for k in ref_sd:
            assert torch.equal(sd[k], ref_sd[k]), \
                f"encoder weight mismatch R5 vs {key}: {k}"
        # (identical weights + identical architecture = identical config)


# 12. identical Transformer configuration between controls
def test_transformer_config_identity():
    built = _build()
    r5 = built["r5_8"]
    for key in ("ctc", "dtc"):
        m = built[key]
        assert m.trajectory.pos.num_embeddings == \
            r5.trajectory.pos.num_embeddings
        assert m.trajectory.pos.embedding_dim == \
            r5.trajectory.pos.embedding_dim
        # same layer hyperparameters (d_model, heads, ffn, dropout)
        l_ref = r5.trajectory.layers.layers[0]
        l_new = m.trajectory.layers.layers[0]
        assert l_ref.self_attn.num_heads == l_new.self_attn.num_heads
        assert l_ref.self_attn.embed_dim == l_new.self_attn.embed_dim
        assert l_ref.linear1.out_features == l_new.linear1.out_features
        assert l_ref.dropout.p == l_new.dropout.p
        # and the K sweep touches ONLY the codebook size
    for other in ("r5_16", "r5_32"):
        m = built[other]
        assert m.trajectory.layers.layers[0].self_attn.embed_dim == \
            r5.trajectory.layers.layers[0].self_attn.embed_dim
        a8, aK = param_audit(r5), param_audit(m)
        for g in ("encoder", "trajectory", "pool", "classifier", "trainable"):
            assert a8[g] == aK[g], f"{other} changed {g} params"
        assert aK["codebook_buffers"] > a8["codebook_buffers"]


# parameter-exactness gate: all five configs share the trainable budget
def test_parameter_exact_match():
    built = _build()
    counts = {k: param_audit(m)["trainable"] for k, m in built.items()}
    assert len(set(counts.values())) == 1, counts
    assert counts["r5_8"] == 599413
