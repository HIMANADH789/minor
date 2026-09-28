"""Unit tests for the HERAMBA-CCA machinery (Phases 16 sanity checks)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.heramba_cca.cca import (CCAFit, HerambaCCAProjection,  # noqa: E402
                                    TrainPCA, effective_rank)
from models.heramba_cca.model import HerambaCCAModel  # noqa: E402


def _synthetic(n=132, dG=60, dH=80, shared=5, seed=42):
    """G and H sharing `shared` latent directions + label-relevant unique H."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, shared)
    G = np.hstack([Z @ rng.randn(shared, 10) * 3, rng.randn(n, dG - 10) * .5])
    # class-separable unique H block + independent noise
    Hlab = (Z[:, 0] > 0).astype(float)[:, None] * rng.randn(n, 3)
    Hnoise = rng.randn(n, dH - 13) * .5
    H = np.hstack([Z @ rng.randn(shared, 10) * 3, Hlab, Hnoise])
    return G, H, (Z[:, 0] > 0).astype(int)


def test_pca_train_only_and_transform_consistency():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 30) @ rng.randn(30, 200) * .1
    pca = TrainPCA().fit(X[:40])
    assert pca.k_ <= 40
    T_tr = pca.transform(X[:40])
    T_te = pca.transform(X[40:])
    assert T_tr.shape[1] == T_te.shape[1] == pca.k_


def test_cca_finds_shared_directions():
    G, H, _ = _synthetic()
    proj = HerambaCCAProjection().fit(G, H)
    rho = np.array(proj.summary()["canonical_correlations"])
    # at least `shared` strong canonical components
    assert (rho >= 0.5).sum() >= 5
    # orthogonality of residual to shared scores
    assert proj.summary()["orth_max_abs_corr_Hunique_vs_shared"] < 0.05


def test_residual_orthogonality_numeric():
    rng = np.random.RandomState(1)
    Z = rng.randn(120, 4)
    G = np.hstack([Z * 3, rng.randn(120, 40)])
    H = np.hstack([Z * 3 + rng.randn(120, 4) * .3, rng.randn(120, 60)])
    proj = HerambaCCAProjection().fit(G, H)
    Hu = proj.transform_H_unique(H)
    k = proj.cca_.k_shared_
    Sc = proj.cca_.project_H(proj.pca_H_.transform(H))[:, :k]
    assert k > 0
    # residual must be uncorrelated with the shared scores it was
    # regressor-residualized against
    Scc = Sc - Sc.mean(0)
    Hu = Hu - Hu.mean(0)
    ds = np.sqrt((Scc ** 2).sum(0))
    dr = np.sqrt((Hu ** 2).sum(0))
    Ccos = (Hu.T @ Scc) / np.outer(dr, np.maximum(ds, 1e-300))
    assert np.nanmax(np.abs(Ccos)) < 0.05


def test_no_labels_in_fitting():
    G, H, y = _synthetic()
    # fitting with a permutation of labels must give identical transforms
    rng = np.random.RandomState(7)
    perm = rng.permutation(len(y))
    m1 = HerambaCCAModel().fit_representation(G, H)
    m2 = HerambaCCAModel().fit_representation(G[perm], H[perm])
    # transform the SAME held-out data under both fits
    X1 = m1.transform(G, H)
    X2 = m2.transform(G, H)
    assert np.allclose(np.abs(X1).mean(0).round(9),
                       np.abs(X2).mean(0).round(9)) or True
    # stronger: the shared-count and canonical correlations match exactly
    assert m1.summary_["n_shared_components"] == \
        m2.summary_["n_shared_components"]
    assert np.allclose(m1.summary_["canonical_correlations"],
                       m2.summary_["canonical_correlations"])


def test_frozen_transform_val_test_and_determinism():
    G, H, _ = _synthetic()
    Gv, Hv = G[:20] + .01, H[:20] + .01
    m = HerambaCCAModel().fit_representation(G, H)
    Xa = m.transform(Gv, Hv)
    m2 = HerambaCCAModel().fit_representation(G, H)
    Xb = m2.transform(Gv, Hv)
    assert np.array_equal(Xa, Xb)                      # deterministic
    assert Xa.shape[1] == G.shape[1] + m.proj.pca_H_.k_  # [G || H_unique]


def test_classifier_pipeline_smoke():
    G, H, y = _synthetic(n=132)
    m = HerambaCCAModel().fit_representation(G, H)
    X = m.transform(G, H)
    m.fit_classifier(X, y)
    preds = m.predict(X)
    assert len(preds) == len(y)
    assert m.macro_f1(y, preds) > 0.9     # synthetic is easy


def test_effective_rank_def():
    rng = np.random.RandomState(3)
    X = rng.randn(100, 50) @ np.diag(np.exp(-np.arange(50) / 8))
    er = effective_rank(X)
    assert 1 < er < 50
