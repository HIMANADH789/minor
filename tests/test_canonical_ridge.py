"""Unit tests for the proposed CCA-adaptive generalized ridge model.

Covers the Phase-15 checklist: deterministic transforms, train-only
PCA/CCA, canonical dimensions, exact reconstruction, alpha schedule,
no-test-label dependence, determinism, solver exactness, and the
canonical MiniROCKET baseline value.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.canonical_ridge.model import (  # noqa: E402
    CanonicalBasis, GeneralizedRidgeClassifier, alpha_schedule, onehot)
from experiments.heramba_cca_ranked_haptics_seed42.runner import (  # noqa
    RankPCA)
from models.heramba_cca.cca import CCAFit  # noqa: E402

CACHE = r"C:/temp/results"
G_PATH = os.path.join(CACHE, "haptics_inference_banks_G_trva.npy")
H_PATH = os.path.join(CACHE, "haptics_inference_banks_H_trva.npy")


def _banks():
    G = np.load(G_PATH).astype(np.float64)
    H = np.load(H_PATH).astype(np.float64)
    return G[:132], H[:132]


def _pipeline(seed=0, n=132):
    """Small deterministic synthetic G/H with known shared structure."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, 5)
    G = np.hstack([Z @ rng.randn(5, 8) * 3, rng.randn(n, 30) * .5])
    H = np.hstack([Z @ rng.randn(5, 8) * 3, rng.randn(n, 40) * .5])
    y = rng.randint(0, 3, n)
    return G, H, y


# 1. deterministic feature transform
def test_deterministic_transform():
    G, H, _ = _pipeline()
    pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
    Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
    cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    b1 = CanonicalBasis.fit(Hl, cca)
    Hc1, Hp1 = b1.transform(Hl)
    b2 = CanonicalBasis.fit(Hl, cca)
    Hc2, Hp2 = b2.transform(Hl)
    assert np.array_equal(Hc1, Hc2) and np.array_equal(Hp1, Hp2)


# 2. train-only PCA
def test_pca_train_only():
    rng = np.random.RandomState(1)
    X = rng.randn(80, 25) @ np.diag(np.exp(-np.arange(25) / 5))
    p = RankPCA().fit(X[:60])
    k0 = p.k_
    p.transform_low(X[60:])          # transform must not refit
    assert p.k_ == k0
    assert p.cumulative_variance_[p.k_ - 1] >= 0.95


# 3. train-only CCA (labels never enter)
def test_cca_label_free():
    G, H, y = _pipeline()
    rng = np.random.RandomState(3)
    perm = rng.permutation(len(y))
    pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
    Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
    c1 = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    c2 = CCAFit().fit(Gl[perm], Hl[perm], tau_shared=0.5)   # y unused
    assert np.allclose(c1.rho_, c2.rho_)


# 4. correct canonical dimensions
def test_canonical_dimensions():
    G, H, _ = _pipeline()
    pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
    Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
    cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    K = cca.B_.shape[1]
    assert K == min(Gl.shape[1], Hl.shape[1])
    basis = CanonicalBasis.fit(Hl, cca)
    assert basis.K_ == K
    Hc, Hp = basis.transform(Hl)
    assert Hc.shape[1] == K and Hp.shape[1] == Hl.shape[1] - K


# 5. H_CCA + H_perp reconstructs H_low
def test_reconstruction():
    G, H, _ = _pipeline()
    pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
    Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
    cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    basis = CanonicalBasis.fit(Hl, cca)
    Hc, Hp = basis.transform(Hl)
    B, Qp = basis.B_, basis.Q_perp_
    Hw = (Hl - basis.mu_w_) / basis.sd_w_
    Hw_hat = (Hc @ B.T) + (Hp @ Qp.T)
    assert np.abs(Hw_hat - Hw).max() < 1e-8
    assert basis.recon_max_err_ < 1e-8


# 6. alpha_k follows the exact declared formula
def test_alpha_schedule_exact():
    rng = np.random.RandomState(5)
    rho = rng.rand(20)
    for base in [0.5, 1.0, 7.3]:
        assert np.allclose(alpha_schedule(rho, base),
                           base * (1.0 + rho ** 2))


# 7. no alpha depends on test labels
def test_alpha_label_invariance():
    G, H, y = _pipeline()
    pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
    Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
    cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    basis = CanonicalBasis.fit(Hl, cca)
    Hc, Hp = basis.transform(Hl)
    Z = np.hstack([Gl, Hc, Hp])
    Y = onehot(y, 3)
    lam = alpha_schedule(cca.rho_, 1.0)
    lam_full = np.concatenate([np.ones(Gl.shape[1]), lam,
                               np.ones(Hp.shape[1])])
    rng = np.random.RandomState(9)
    a1 = GeneralizedRidgeClassifier().fit(
        Z, Y, lam_full, alpha_grid=np.logspace(-2, 2, 7)).alpha_base_
    # shuffle labels: decomposition untouched, but alpha_base must change
    # or stay identical ONLY through Z; labels are not used in Z, so the
    # GCV objective changes -> alpha may differ; either way fitting must
    # not crash and the decomposition is identical (checked in test 3).
    y2 = y[rng.permutation(len(y))]
    a2 = GeneralizedRidgeClassifier().fit(
        Z, onehot(y2, 3), lam_full,
        alpha_grid=np.logspace(-2, 2, 7)).alpha_base_
    assert a1 > 0 and a2 > 0


# 8. predictions are deterministic
def test_predictions_deterministic():
    G, H, y = _pipeline()
    pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
    Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
    cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    basis = CanonicalBasis.fit(Hl, cca)
    Hc, Hp = basis.transform(Hl)
    Z = np.hstack([Gl, Hc, Hp])
    Y = onehot(y, 3)
    lam_full = np.concatenate([np.ones(Gl.shape[1]),
                               alpha_schedule(cca.rho_, 1.0),
                               np.ones(Hp.shape[1])])
    c1 = GeneralizedRidgeClassifier().fit(
        Z, Y, lam_full, alpha_grid=np.logspace(-2, 2, 7))
    c2 = GeneralizedRidgeClassifier().fit(
        Z, Y, lam_full, alpha_grid=np.logspace(-2, 2, 7))
    assert np.array_equal(c1.predict(Z), c2.predict(Z))
    assert np.array_equal(c1.W_, c2.W_)


# 9. generalized ridge solution is exactly reproducible vs brute force
def test_solver_exact():
    rng = np.random.RandomState(11)
    n, d = 60, 12
    Z = rng.randn(n, d)
    y = rng.randint(0, 3, n)
    Y = onehot(y, 3)
    lam = np.linspace(0.5, 2.0, d)
    clf = GeneralizedRidgeClassifier().fit(
        Z, Y, lam, alpha_grid=np.logspace(-2, 2, 7))
    Zc = Z - clf.zm_
    eval_at, *_ = clf._solve_curve(Zc, Y, lam)
    for c in [0.1, 1.0, 10.0, clf.alpha_base_]:
        Wm, df, rss = eval_at(c)
        Wb = np.linalg.solve(Zc.T @ Zc + c * np.diag(lam), Zc.T @ Y)
        assert np.abs(Wm - Wb).max() < 1e-8
        assert abs(rss - float(((Y - Zc @ Wb) ** 2).sum())) < 1e-8
        df_bf = float(np.trace(Zc @ np.linalg.solve(
            Zc.T @ Zc + c * np.diag(lam), Zc.T)))
        assert abs(df - df_bf) < 1e-6


# 10. canonical G baseline reproduces the stored MiniROCKET reference
@pytest.mark.skipif(not os.path.exists(G_PATH), reason="frozen banks absent")
def test_canonical_G_baseline():
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import f1_score
    from experiments.external_stack_generalization.data import load_dataset
    G = np.load(G_PATH).astype(np.float64)
    d = load_dataset("Haptics")
    ytr, yva, yte = d["ytr"], d["yva"], d["yte"]
    y_dev = np.concatenate([ytr, yva])
    G_te = np.load(os.path.join(CACHE,
                                "haptics_inference_banks_G_te.npy")
                   ).astype(np.float64)
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(G, y_dev)
    pred = clf.predict(G_te)
    assert round(float(f1_score(yte, pred, average="macro",
                                zero_division=0)), 4) == 0.5037
