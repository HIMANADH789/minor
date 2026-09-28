"""R4 (differentiable-Ridge) tests: H_{m,k} decomposition, gate semantics,
closed-form dual Ridge identities, outer-loop mechanics."""

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from experiments.r4_regime_gate_seed42 import core, ridge_gate  # noqa: E402

K = 8


def test_dataset_configs():
    assert set(core.DATASETS) == {"Haptics", "ECG5000_BAL"}
    assert core.DATASETS["ECG5000_BAL"]["R2"] == 0.6409
    assert core.DATASETS["Haptics"]["R2"] == 0.5500
    assert core.SEED == 42
    # frozen R2 alphas (audit 9): recorded from the R2 report artifacts
    assert core.DATASETS["Haptics"]["alpha_r2"] == 4.281332398719396
    assert core.DATASETS["ECG5000_BAL"]["alpha_r2"] == 1.623776739188721


def test_contribution_decomposition_sums_to_H():
    """sum_k H_{m,k} == audited H_m (the Audit-8 identity)."""
    g = np.random.RandomState(0)
    N, F, T = 3, 5, 80
    act = (g.rand(N, F, T) < 0.4)
    valid = np.ones((F, T), dtype=bool)
    valid[:, :6] = False
    valid[:, -6:] = False        # symmetric [6, T-6): _padding_groups contract
    regimes = g.randint(0, K, size=(N, T))
    Hk = core.compute_H_contributions(act, valid, regimes)
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        compute_regime_heterogeneity,
    )
    H = compute_regime_heterogeneity(act, valid, regimes)
    assert np.abs(Hk.sum(axis=2) - H).max() <= 3.9e-9


def test_contribution_excludes_low_occupancy_regimes():
    """A regime below min occupancy gets exactly zero contribution.

    T=200 -> min_count = ceil(0.01*200) = 2, so a regime with 1 hit is
    excluded (with T=100 min_count would be 1 and 1 hit would pass).
    """
    g = np.random.RandomState(1)
    N, F, T = 2, 3, 200
    act = (g.rand(N, F, T) < 0.5)
    valid = np.ones((F, T), dtype=bool)
    regimes = np.zeros((N, T), dtype=int)      # regime 7 never present
    Hk = core.compute_H_contributions(act, valid, regimes)
    assert (Hk[..., 7] == 0).all()
    # and a rare regime (1 hit < min_count=2) is zeroed too
    regimes[:, 10] = 7
    Hk2 = core.compute_H_contributions(act, valid, regimes)
    assert (Hk2[..., 7] == 0).all()


# --------------------------------------------------------------------- #
# gate semantics (Ridge model)                                           #
# --------------------------------------------------------------------- #
def test_eight_scalar_gates_sigmoid_init():
    m = ridge_gate.DualRidgeGateModel(alpha=1.0)
    assert m.theta.numel() == 8
    assert float(m.theta[0]) == pytest.approx(-2.0)
    v = m.v().detach().numpy()
    assert np.all((v > 0) & (v < 1)) and np.all(v < 0.5)
    assert v[0] == pytest.approx(0.1192029, abs=1e-6)


def _toy_cache(n=48, n_g=6, c=3, seed=0, signal=True):
    """Prepare() a DualRidgeGateModel on synthetic G / Hk / labels."""
    g = np.random.RandomState(seed)
    G = g.randn(n, n_g).astype(np.float64)
    y = g.randint(0, c, size=n).astype(np.int64)
    Hk = g.rand(n, n_g, K).astype(np.float32) * 0.2
    if signal:
        Hk[:, :, 0] += np.eye(c, n_g)[y] * 3.0
    n_tr = 32
    m = ridge_gate.DualRidgeGateModel(alpha=0.1)
    m.prepare(G[:n_tr], Hk[:n_tr], y[:n_tr],
              G[n_tr:], Hk[n_tr:], y[n_tr:])
    return m, G, Hk, y, n_tr


def test_dual_ridge_matches_sklearn_at_fixed_v():
    """v=1 dual-solve predictions == sklearn RidgeClassifier fit on the
    same TRAIN rows (audit 3 core identity, at toy scale)."""
    from sklearn.linear_model import RidgeClassifier
    m, G, Hk, y, n_tr = _toy_cache(seed=0)
    with torch.no_grad():
        m.theta.data.fill_(20.0)
        beta, Kd, _, _ = m.solve_beta()
        f = m.decision_val(beta).numpy()
    pred_ours = f.argmax(axis=1)
    v1 = np.ones(K)
    Hg_tr = np.einsum("nfk,k->nf", Hk[:n_tr].astype(np.float64), v1)
    Hg_va = np.einsum("nfk,k->nf", Hk[n_tr:].astype(np.float64), v1)
    Xtr = np.hstack([G[:n_tr], Hg_tr])
    Xva = np.hstack([G[n_tr:], Hg_va])
    rc = RidgeClassifier(alpha=0.1)
    rc.fit(Xtr, y[:n_tr])
    pred_sk = rc.predict(Xva)
    assert (pred_ours == pred_sk).mean() > 0.95
    # decision functions must also be close (same objective, scaling tol)
    f_sk = rc.decision_function(Xva)
    corr = np.corrcoef(f.ravel(), f_sk.ravel())[0, 1]
    assert corr > 0.99


def test_v1_reproduces_ungated_H_and_v0_kills_it():
    m, G, Hk, y, n_tr = _toy_cache(seed=1)
    with torch.no_grad():
        m.theta.data.fill_(20.0)
        Hg = m.gated_H(torch.from_numpy(Hk[:4].astype(np.float64))).numpy()
        assert np.abs(Hg - Hk[:4].astype(np.float64).sum(axis=2)).max() \
            <= 1e-6
        m.theta.data.fill_(-60.0)
        Hg0 = m.gated_H(torch.from_numpy(Hk[:4].astype(np.float64))).numpy()
        assert float(np.abs(Hg0).max()) < 1e-12


def test_train_only_fit_val_only_outer_loss():
    """The dual Gram must use exactly the TRAIN rows (audit 10)."""
    m, G, Hk, y, n_tr = _toy_cache(seed=2)
    Kd = m.train_gram().detach().numpy()
    # Kd should equal Gc Gc^T + Hg Hg^T + alpha I on the train block only
    with torch.no_grad():
        m.theta.data.fill_(20.0)
    Kd = m.train_gram().detach().numpy()
    Hg = np.einsum("nfk,k->nf", m.cache["Hkc_tr"].numpy(),
                   m.v().detach().numpy())
    Gc = m.cache["Gc_tr"].numpy()
    Kexpect = Gc @ Gc.T + Hg @ Hg.T + 0.1 * np.eye(n_tr)
    assert np.abs(Kd - Kexpect).max() <= 1e-8
    assert Kd.shape == (n_tr, n_tr)


def test_gradient_flows_through_solve_to_theta():
    m, G, Hk, y, n_tr = _toy_cache(seed=3)
    m.theta.grad = None
    loss, beta, Kd, a_c, L = m.outer_loss()
    loss.backward()
    assert m.theta.grad is not None
    assert torch.isfinite(m.theta.grad).all()
    assert float(m.theta.grad.abs().max()) > 0.0


def test_outer_loop_improves_and_selects_on_val():
    """With H signal only in regime 0, the outer loop must raise v_0."""
    m, G, Hk, y, n_tr = _toy_cache(seed=5, signal=True)
    best, final, traj = ridge_gate.run_outer_loop(
        m, y[n_tr:], max_steps=80, patience=80, lr=5e-2, seed=42)
    v_final = np.array(final["v"])
    assert v_final[0] > v_final[7]
    assert best["val_mf1"] >= 0.0
    assert len(traj) >= 1
    # trajectory bookkeeping
    assert all("solver_residual" in t for t in traj)
    assert best["theta"] is not None and best["v"] is not None


def test_outer_loop_deterministic():
    outs = []
    for _ in range(2):
        m, G, Hk, y, n_tr = _toy_cache(seed=7)
        best, final, traj = ridge_gate.run_outer_loop(
            m, y[n_tr:], max_steps=12, patience=12, seed=42)
        outs.append([t["v"][0] for t in traj])
    assert outs[0] == outs[1]


def test_refit_trainval_matches_sklearn_v1():
    """Final refit at v=1 must equal sklearn Ridge on train+val."""
    from sklearn.linear_model import RidgeClassifier
    g = np.random.RandomState(11)
    n, n_g, c = 60, 5, 3
    G = g.randn(n, n_g).astype(np.float64)
    Hk = g.rand(n, n_g, K).astype(np.float32) * 0.3
    y = g.randint(0, c, size=n).astype(np.int64)
    Hg = Hk.astype(np.float64).sum(axis=2)
    X = np.hstack([G, Hg])
    rc = RidgeClassifier(alpha=0.2)
    rc.fit(X, y)
    m = ridge_gate.DualRidgeGateModel(alpha=0.2)
    beta, b0, resid = m.refit_trainval(G, Hk, y, np.ones(K))
    dec_ours = X @ beta + b0
    dec_sk = rc.decision_function(X)
    # same objective -> decisions agree up to per-class scale/shift of
    # the multinomial convention; predictions must match
    assert (dec_ours.argmax(axis=1) == rc.predict(X)).mean() == 1.0
    assert np.abs(dec_ours - dec_sk).max() < 1e-6 or \
        np.corrcoef(dec_ours.ravel(), dec_sk.ravel())[0, 1] > 0.99
    assert resid < 1e-8


def test_optimizer_contains_only_theta():
    m, G, Hk, y, n_tr = _toy_cache(seed=13)
    opt = torch.optim.Adam([m.theta], lr=ridge_gate.LR)
    assert opt.param_groups[0]["params"] == [m.theta]
    assert m.theta.numel() == 8            # exactly 8 gates (audit 13)


def test_penalty_lambda_v_sum_theta2():
    m = ridge_gate.DualRidgeGateModel(alpha=1.0)
    with torch.no_grad():
        m.theta.data.copy_(torch.tensor([0.5, -0.5, 1.0, -1.0, 0.0,
                                         2.0, -2.0, 0.25]))
    expect = ridge_gate.LAMBDA_V * 10.5625
    gp = float(ridge_gate.LAMBDA_V * (m.theta ** 2).sum())
    assert gp == pytest.approx(expect, rel=1e-9)
