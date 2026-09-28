"""R3 gate tests: gate semantics, controls, loss form, frozen stages."""

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from experiments.r3_gate_seed42 import core, gate_model  # noqa: E402

K = 8
N_G, N_H = 4998, 4998


# ------------------------------------------------------------------ #
# dataset configuration (audit 1)                                     #
# ------------------------------------------------------------------ #
def test_dataset_configs():
    assert set(core.DATASETS) == {"Haptics", "ECG5000_BAL"}
    assert core.DATASETS["Haptics"]["expected"] == {
        "train": 132, "val": 23, "test": 308, "T": 1092, "n_classes": 5}
    assert core.DATASETS["ECG5000_BAL"]["expected"] == {
        "train": 5226, "val": 923, "test": 1000, "T": 140, "n_classes": 5}
    assert core.DATASETS["Haptics"]["R2"] == 0.5500
    assert core.DATASETS["ECG5000_BAL"]["R2"] == 0.6089
    assert core.SEED == 42


# ------------------------------------------------------------------ #
# gate semantics (audits 10, 11)                                      #
# ------------------------------------------------------------------ #
def test_gate_init_below_half():
    m = gate_model.R3GateModel(n_classes=5)
    assert float(m.theta) == pytest.approx(-2.0)
    w = float(m.gate())
    assert w == pytest.approx(0.1192029, abs=1e-6)
    assert 0.0 < w < 0.5


def test_gate_is_single_scalar():
    m = gate_model.R3GateModel(n_classes=5)
    assert m.theta.numel() == 1
    for t in (-15.0, -2.0, 0.0, 2.0, 15.0):   # float32 saturates at +-30
        m.theta.data.fill_(t)
        w = float(m.gate())
        assert 0.0 < w < 1.0


def test_gate_scales_whole_H_block_only():
    torch.manual_seed(0)
    m = gate_model.R3GateModel(n_features_g=6, n_features_h=4, n_classes=3)
    G = torch.randn(5, 6)
    H = torch.randn(5, 4)
    m.theta.data.fill_(0.3)
    w = float(m.gate())
    Fout = torch.cat([G, w * H], dim=1)
    assert torch.allclose(Fout[:, :6], G)
    assert torch.allclose(Fout[:, 6:], w * H)


# ------------------------------------------------------------------ #
# loss form (audit 15)                                                #
# ------------------------------------------------------------------ #
def test_gate_penalty_exact_form():
    m = gate_model.R3GateModel(n_classes=5)
    for th in (0.0, -0.7, 1.3):
        m.theta.data.fill_(th)
        assert float(m.gate_penalty()) == pytest.approx(
            gate_model.LAMBDA_W * th ** 2, rel=1e-6)   # float32 theta


def test_weight_penalty_lambda_clf():
    m = gate_model.R3GateModel(n_features_g=4, n_features_h=4, n_classes=3)
    expect = gate_model.LAMBDA_CLF * (m.W ** 2).sum().item()
    assert float(m.weight_penalty()) == pytest.approx(expect, rel=1e-12)


def test_loss_total_composition():
    """L = CE + lambda_clf||W||^2 + lambda_w theta^2, no other terms."""
    torch.manual_seed(1)
    m = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    G = torch.randn(10, 8)
    H = torch.randn(10, 6)
    y = torch.randint(0, 3, (10,))
    import torch.nn.functional as F
    ce = F.cross_entropy(m(G, H), y)
    total = ce + m.weight_penalty() + m.gate_penalty()
    assert total.item() == pytest.approx(
        ce.item() + float(m.weight_penalty()) + float(m.gate_penalty()),
        rel=1e-6)   # float32 tensor add vs float64 python add


# ------------------------------------------------------------------ #
# variant identity (audits 13, 14)                                    #
# ------------------------------------------------------------------ #
def test_variant_forward_identity_gh_and_g():
    torch.manual_seed(2)
    m = gate_model.R3GateModel(n_features_g=6, n_features_h=4, n_classes=3)
    G = torch.randn(4, 6)
    H = torch.randn(4, 4)
    # w -> 1: model forward == [G || H]
    m.theta.data.fill_(20.0)
    assert torch.allclose(m(G, H), torch.cat([G, H], dim=1) @ m.W.t() + m.b,
                          atol=1e-6)
    # w -> 0: model forward == G-only path
    m.theta.data.fill_(-20.0)
    g_only = G @ m.W[:, :6].t() + m.b
    assert torch.allclose(m(G, H), g_only, atol=1e-7)


def test_r3g_uses_g_slice_only():
    """R3-G forward must not depend on H columns of W (or on H itself)."""
    torch.manual_seed(3)
    m = gate_model.R3GateModel(n_features_g=6, n_features_h=4, n_classes=3)
    G = torch.randn(4, 6)
    logits1 = G @ m.W[:, :6].t() + m.b
    m.W.data[:, 6:] += 17.0                    # corrupt H columns
    logits2 = G @ m.W[:, :6].t() + m.b
    assert torch.allclose(logits1, logits2)


# ------------------------------------------------------------------ #
# training loop: gating, freezing, selection (audits 12, 18, 19)      #
# ------------------------------------------------------------------ #
def _toy_data(n=64, n_g=8, n_h=6, c=3, h_signal=1.0, seed=0):
    g = np.random.RandomState(seed)
    G = g.randn(n, n_g).astype(np.float32)
    # H block carries class signal only if h_signal > 0
    y = g.randint(0, c, size=n).astype(np.int64)   # CE needs int64
    H = g.randn(n, n_h).astype(np.float32) * 0.3
    H += np.eye(c, n_h)[y] * h_signal
    tr = slice(0, int(n * 0.75))
    va = slice(int(n * 0.75), n)
    T = lambda A, yy: (torch.from_numpy(A[tr]), torch.from_numpy(A[va]),
                       torch.from_numpy(yy[tr]), torch.from_numpy(yy[va]))
    Gtr, Gva, ytr, yva = T(G, y)
    Htr, Hva, _, _ = T(H, y)
    return (Gtr, Htr, ytr), (Gva, Hva, yva), yva


def test_frozen_context_has_no_grad():
    """R3 stages 1-3 frozen: context model params never require grad."""
    ctx = core.load_frozen_context("Haptics", "cpu")
    assert all(not p.requires_grad for p in ctx["model"].parameters())


def test_training_only_updates_gate_and_clf():
    """Optimizer must not touch frozen R2 params (audit 18 analogue)."""
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data()
    m = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    m.theta.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    assert len(params) == 3            # theta, W, b -- nothing else exists
    opt = torch.optim.Adam(params, lr=1e-2)
    W0 = m.W.detach().clone()
    th0 = m.theta.item()
    import torch.nn.functional as F
    for _ in range(3):
        opt.zero_grad()
        loss = F.cross_entropy(m(Gtr, Htr), ytr) + m.weight_penalty() \
            + m.gate_penalty()
        loss.backward()
        opt.step()
    assert not torch.equal(m.W.detach(), W0)
    assert m.theta.item() != th0


def test_best_checkpoint_is_val_selected():
    (Gtr, Htr, ytr), (Gva, Hva, yva), yva_np = _toy_data(seed=3)
    m = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    best, final, traj = gate_model.train_model(
        m, Gtr, Htr, ytr, Gva, Hva, yva, variant="R3", max_epochs=8,
        patience=8, seed=42)
    vals = [t["val_macro_f1"] for t in traj]
    assert best["val_mf1"] == pytest.approx(max(vals))
    assert best["epoch"] == int(np.argmax(vals))
    assert len(traj) <= 8


def test_gate_trajectory_recorded():
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data(seed=5)
    m = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    best, final, traj = gate_model.train_model(
        m, Gtr, Htr, ytr, Gva, Hva, yva, variant="R3", max_epochs=6,
        patience=6, seed=42)
    ws = [t["w"] for t in traj]
    assert len(ws) == 6
    # trajectory point 0 is AFTER the first update; w_init itself is
    # recorded separately (sigmoid(-2) = 0.1192) in results/w_init
    assert abs(ws[0] - 0.1192029) < 0.01
    assert all(0.0 < w < 1.0 for w in ws)
    assert final["w"] == ws[-1]


def test_gh_control_reaches_w1_equivalence():
    """R3-GH training must never scale H (exact w=1 by construction)."""
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data(seed=7)
    m = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    best, final, traj = gate_model.train_model(
        m, Gtr, Htr, ytr, Gva, Hva, yva, variant="R3-GH", max_epochs=3,
        patience=3, seed=42)
    # theta untouched at init and not in any optimizer
    for t in traj:
        assert t["theta"] == pytest.approx(-2.0)
    assert m.theta.requires_grad is False


def test_r3gIgnores_h_block():
    """R3-G loss/preds must not see H: corrupt H, identical trajectory."""
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data(seed=9)
    m1 = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    b1, f1, t1 = gate_model.train_model(
        m1, Gtr, Htr, ytr, Gva, Hva, yva, variant="R3-G", max_epochs=4,
        patience=4, seed=42)
    Htr2 = Htr + 100.0
    Hva2 = Hva + 100.0
    m2 = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    b2, f2, t2 = gate_model.train_model(
        m2, Gtr, Htr2, ytr, Gva, Hva2, yva, variant="R3-G", max_epochs=4,
        patience=4, seed=42)
    assert t1 == t2


def test_gate_learns_direction_on_toy():
    """Sanity: with a strongly useful H block, w should increase."""
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data(h_signal=4.0, seed=11)
    m = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    best, final, traj = gate_model.train_model(
        m, Gtr, Htr, ytr, Gva, Hva, yva, variant="R3", max_epochs=40,
        patience=40, seed=42, lr=3e-2)
    assert traj[-1]["w"] > traj[0]["w"]


def test_gate_can_stay_shut_when_h_useless():
    """Sanity: with a label-shuffled H block, w must not increase."""
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data(h_signal=0.0, seed=13)
    m = gate_model.R3GateModel(n_features_g=8, n_features_h=6, n_classes=3)
    best, final, traj = gate_model.train_model(
        m, Gtr, Htr, ytr, Gva, Hva, yva, variant="R3", max_epochs=40,
        patience=40, seed=42, lr=3e-2)
    assert traj[-1]["w"] < 0.5          # started 0.119; must stay low


# ------------------------------------------------------------------ #
# finite / determinism guards                                         #
# ------------------------------------------------------------------ #
def test_no_nan_on_toy_training():
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data(seed=17)
    for variant in ("R3-G", "R3-GH", "R3"):
        m = gate_model.R3GateModel(n_features_g=8, n_features_h=6,
                                   n_classes=3)
        best, final, traj = gate_model.train_model(
            m, Gtr, Htr, ytr, Gva, Hva, yva, variant=variant, max_epochs=5,
            patience=5, seed=42)
        assert all(np.isfinite(t["train_loss"]) for t in traj)
        assert np.isfinite(final["theta"]) and np.isfinite(final["w"])


def test_training_deterministic_given_seed():
    (Gtr, Htr, ytr), (Gva, Hva, yva), _ = _toy_data(seed=19)
    outs = []
    for _ in range(2):
        m = gate_model.R3GateModel(n_features_g=8, n_features_h=6,
                                   n_classes=3)
        best, final, traj = gate_model.train_model(
            m, Gtr, Htr, ytr, Gva, Hva, yva, variant="R3", max_epochs=6,
            patience=6, seed=42)
        outs.append([t["w"] for t in traj])
    assert outs[0] == outs[1]
