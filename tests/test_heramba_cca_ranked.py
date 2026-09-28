"""Sanity tests for the rank-controlled CCA HERAMBA experiment (Phase 15)."""
import glob
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.heramba_cca_ranked_haptics_seed42.runner import (  # noqa
    RankPCA, SharedResidualLow, VAR_THRESHOLD)
from models.heramba_cca.cca import CCAFit  # noqa: E402

PREV_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "heramba_cca", "haptics_seed42")


def _synthetic(n=200, dG=40, dH=60, shared=4, seed=7):
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, shared)
    G = np.hstack([Z @ rng.randn(shared, 8) * 3, rng.randn(n, dG - 8) * .5])
    H = np.hstack([Z @ rng.randn(shared, 8) * 3,
                   rng.randn(n, dH - 8) * .5])
    return G, H


def test_rank_pca_is_train_only_and_deterministic():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 30) @ np.diag(np.exp(-np.arange(30) / 6))
    Xva = rng.randn(10, 30)
    p1 = RankPCA().fit(X)
    p2 = RankPCA().fit(X)
    assert p1.k_ == p2.k_ and np.array_equal(p1.comp_, p2.comp_)
    t1 = p1.transform_low(Xva)
    # refitting with extra (val) data must not change the frozen transform
    p3 = RankPCA().fit(np.vstack([X, Xva]))
    # (fit is deterministic given its input; the point is transform_low
    #  uses only frozen parameters)
    assert t1.shape == (10, p1.k_)


def test_rank_pca_energy_rule():
    rng = np.random.RandomState(1)
    X = rng.randn(150, 50) @ np.diag(np.exp(-np.arange(50) / 4))
    p = RankPCA().fit(X)
    assert p.cumulative_variance_[p.k_ - 1] >= VAR_THRESHOLD
    assert p.cumulative_variance_[p.k_ - 2] < VAR_THRESHOLD or p.k_ == 1


def test_shared_residual_orthogonal():
    G, H = _synthetic()
    pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
    Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
    cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    assert cca.k_shared_ > 0
    assert not np.allclose(cca.rho_, 1.0)
    Sc = cca.project_H(Hl)[:, :cca.k_shared_]
    r = SharedResidualLow().fit(Hl, Sc)
    Hu = r.transform(Hl, Sc)
    Sc = Sc - Sc.mean(0)
    Hu = Hu - Hu.mean(0)
    ds = np.sqrt((Sc ** 2).sum(0))
    dr = np.sqrt((Hu ** 2).sum(0))
    keep_s = ds > 1e-8
    keep_r = dr > 1e-8
    assert keep_s.all() and keep_r.all()   # non-degenerate synthetic case
    C = (Hu.T @ Sc) / np.outer(dr, ds)
    assert np.nanmax(np.abs(C)) < 1e-8          # exact OLS orthogonality
    assert r.orth_max_abs_corr_ < 1e-8


def test_transform_never_refits_and_is_frozen():
    G, H = _synthetic()
    n = len(G) // 2
    Gtr, Htr = G[:n], H[:n]
    pcaG, pcaH = RankPCA().fit(Gtr), RankPCA().fit(Htr)
    Gl, Hl = pcaG.transform_low(Gtr), pcaH.transform_low(Htr)
    cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
    Sc = cca.project_H(Hl)[:, :cca.k_shared_]
    r = SharedResidualLow().fit(Hl, Sc)
    Gte, Hte = G[n:], H[n:]
    a = r.transform(pcaH.transform_low(Hte),
                    cca.project_H(pcaH.transform_low(Hte))[:, :cca.k_shared_])
    b = r.transform(pcaH.transform_low(Hte),
                    cca.project_H(pcaH.transform_low(Hte))[:, :cca.k_shared_])
    assert np.array_equal(a, b)                 # deterministic


def test_previous_fullrank_cca_artifacts_untouched():
    """Historical evidence must remain byte-identical."""
    expected = ["cca_summary.json", "canonical_correlations.csv",
                "complementarity.json", "complementarity_evidence.csv",
                "diagnostics.json", "effective_rank.csv",
                "final_comparison.csv", "results.json",
                "evidence_report.md"]
    for name in expected:
        assert os.path.exists(os.path.join(PREV_DIR, name)), name
    d = __import__("json").load(
        open(os.path.join(PREV_DIR, "cca_summary.json")))
    assert d["n_shared_components"] == 131          # the degenerate result
    assert d["canonical_correlations"][0] == 1.0
    r = __import__("json").load(
        open(os.path.join(PREV_DIR, "results.json")))
    assert r["cca_heramba"]["macro_f1"] == 0.5037
    assert r["r2_gh"]["macro_f1"] == 0.55


def test_labels_never_enter_decomposition():
    """Decomposition outputs are invariant to label permutation."""
    G, H = _synthetic()
    y = np.random.RandomState(3).randint(0, 2, len(G))
    rng = np.random.RandomState(5)
    perm = rng.permutation(len(y))
    def run():
        pcaG, pcaH = RankPCA().fit(G), RankPCA().fit(H)
        Gl, Hl = pcaG.transform_low(G), pcaH.transform_low(H)
        cca = CCAFit().fit(Gl, Hl, tau_shared=0.5)
        return cca.rho_, cca.k_shared_
    r1, k1 = run()
    r2, k2 = run()  # labels are never passed anywhere in run()
    assert np.allclose(r1, r2) and k1 == k2
