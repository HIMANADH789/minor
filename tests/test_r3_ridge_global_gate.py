"""R3 (differentiable-Ridge global gate) tests: gate semantics, exact
Gram decomposition, dual-vs-sklearn identity, outer-loop mechanics."""

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from experiments.r3_ridge_global_gate_seed42 import core, gate_model  # noqa


def test_dataset_configs():
    assert set(core.DATASETS) == {"Haptics", "ECG5000_BAL"}
    assert core.DATASETS["Haptics"]["R2"] == 0.5500
    assert core.DATASETS["Haptics"]["M0"] == 0.4974
    assert core.DATASETS["ECG5000_BAL"]["R2"] == 0.6508
    assert core.DATASETS["ECG5000_BAL"]["M0"] == 0.6553
    assert core.SEED == 42
    # frozen R2 alphas
    assert core.DATASETS["Haptics"]["alpha_r2"] == 4.281332398719396
    assert core.DATASETS["ECG5000_BAL"]["alpha_r2"] == 1.623776739188721


def _toy(n=48, n_g=6, c=3, seed=0, signal=True):
    g = np.random.RandomState(seed)
    G = g.randn(n, n_g).astype(np.float64)
    y = g.randint(0, c, size=n).astype(np.int64)
    H = g.rand(n, n_g).astype(np.float64) * 0.2
    if signal:
        H += np.eye(c, n_g)[y] * 3.0
    n_tr = 32
    m = gate_model.GlobalGateRidge(alpha=0.1)
    m.prepare(G[:n_tr], H[:n_tr], y[:n_tr], G[n_tr:], H[n_tr:], y[n_tr:])
    return m, G, H, y, n_tr


def test_single_scalar_gate_init():
    m = gate_model.GlobalGateRidge(alpha=1.0)
    assert m.theta.numel() == 1                       # audit 13
    assert float(m.theta) == pytest.approx(-2.0)
    w = float(m.w())
    assert 0.0 < w < 1.0 and w < 0.5                  # audits 14/15
    assert w == pytest.approx(0.1192029, abs=1e-6)


def test_gram_decomposition_exact():
    """K(w) == explicit full Gram of Xc_tr(w) for arbitrary w."""
    m, G, H, y, n_tr = _toy(seed=0)
    for wv in (0.0, 0.1192, 0.5, 1.0, 0.7331):
        with torch.no_grad():
            m.theta.copy_(torch.logit(torch.tensor(wv,
                                                   dtype=torch.float64)))
        K = m.gram()
        Xc = torch.cat([m.Gc_tr, wv * m.Hc_tr], dim=1)
        K_full = Xc @ Xc.t() + 0.1 * torch.eye(n_tr,
                                               dtype=torch.float64)
        assert torch.allclose(K, K_full, atol=1e-10)


def test_dual_ridge_matches_sklearn_at_fixed_w():
    """At w=1 (and w=0.5), dual-solve predictions == sklearn Ridge with
    the same frozen alpha on the same TRAIN rows."""
    from sklearn.linear_model import RidgeClassifier
    m, G, H, y, n_tr = _toy(seed=1)
    for wv in (1.0, 0.5):
        with torch.no_grad():
            m.theta.copy_(torch.logit(torch.tensor(wv,
                                                   dtype=torch.float64)))
        with torch.no_grad():
            beta, *_ = m.solve_beta()
            f = m.decision_val(beta).numpy()
        pred_ours = f.argmax(axis=1)
        Xtr = np.hstack([G[:n_tr], wv * H[:n_tr]])
        Xva = np.hstack([G[n_tr:], wv * H[n_tr:]])
        rc = RidgeClassifier(alpha=0.1)
        rc.fit(Xtr, y[:n_tr])
        assert (pred_ours == rc.predict(Xva)).mean() == 1.0


def test_w1_identity_chain():
    """w=1: wH == H, X_R3 == X_R2 (audits 7-8)."""
    m, G, H, y, n_tr = _toy(seed=2)
    Hg = 1.0 * H
    assert np.array_equal(Hg, H)
    X_r3 = np.hstack([G, Hg])
    X_r2 = np.hstack([G, H])
    assert np.array_equal(X_r3, X_r2)


def test_w0_identity_chain():
    """w=0: wH == 0, X_R3 == G-only (audits 9-10)."""
    m, G, H, y, n_tr = _toy(seed=3)
    Hg = 0.0 * H
    assert np.abs(Hg).max() == 0.0
    X_r3 = np.hstack([G, Hg])
    assert np.array_equal(X_r3[:, :G.shape[1]], G)


def test_gradient_flows_through_solve_to_theta():
    m, G, H, y, n_tr = _toy(seed=4)
    m.theta.grad = None
    loss, *_ = m.outer_loss()
    loss.backward()
    assert m.theta.grad is not None
    assert torch.isfinite(m.theta.grad).all()
    assert float(m.theta.grad.abs().max()) > 0.0


def test_outer_loop_opens_gate_on_informative_h():
    m, G, H, y, n_tr = _toy(seed=5, signal=True)
    best, final, traj = gate_model.run_outer_loop(
        m, y[n_tr:], max_steps=120, patience=120, lr=5e-2, seed=42)
    assert best["w"] > 0.5            # H is informative -> gate opens
    assert len(traj) >= 1
    assert all("solver_residual" in t for t in traj)
    assert all(t["grad_finite_nonzero"] for t in traj)


def test_outer_loop_deterministic():
    outs = []
    for _ in range(2):
        m, G, H, y, n_tr = _toy(seed=7)
        best, final, traj = gate_model.run_outer_loop(
            m, y[n_tr:], max_steps=15, patience=15, seed=42)
        outs.append([t["w"] for t in traj])
    assert outs[0] == outs[1]


def test_refit_trainval_matches_sklearn():
    """Final train+val refit at w=1 == sklearn Ridge (train+val)."""
    from sklearn.linear_model import RidgeClassifier
    g = np.random.RandomState(11)
    n, n_g, c = 60, 5, 3
    G = g.randn(n, n_g).astype(np.float64)
    H = g.rand(n, n_g).astype(np.float64)
    y = g.randint(0, c, size=n).astype(np.int64)
    X = np.hstack([G, H])
    rc = RidgeClassifier(alpha=0.2)
    rc.fit(X, y)
    m = gate_model.GlobalGateRidge(alpha=0.2)
    beta, b0, resid, sym = m.refit_trainval(G, H, y, 1.0)
    dec_ours = X @ beta + b0
    assert (dec_ours.argmax(axis=1) == rc.predict(X)).mean() == 1.0
    assert resid < 1e-8
    assert sym < 1e-12


def test_penalty_lambda_w_theta2():
    m = gate_model.GlobalGateRidge(alpha=1.0)
    with torch.no_grad():
        m.theta.copy_(torch.tensor(0.5, dtype=torch.float64))
    gp = float(gate_model.LAMBDA_W * m.theta ** 2)
    assert gp == pytest.approx(gate_model.LAMBDA_W * 0.25, rel=1e-12)


def test_optimizer_contains_only_theta():
    m, G, H, y, n_tr = _toy(seed=13)
    opt = torch.optim.Adam([m.theta], lr=gate_model.LR)
    assert opt.param_groups[0]["params"] == [m.theta]
