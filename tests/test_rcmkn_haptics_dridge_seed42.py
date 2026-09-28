"""Tests for the Haptics differential-Ridge control (fast, synthetic)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rcmkn_haptics_dridge_seed42.config import (
    ALPHA_G_CANON, GAMMAS, K_FOLDS, M0_TEST, R2_TEST, R2_VAL, SEED, TIE_TOL,
)
from experiments.rcmkn_haptics_dridge_seed42.core import (
    dridge_decision_function, dridge_dual_direct, dridge_fit,
    dridge_predict, dridge_primal_direct, select_gamma,
)


def test_gamma_grid_and_frozen_constants():
    assert GAMMAS == [1, 2, 4, 8, 16, 32, 64]
    assert SEED == 42 and K_FOLDS == 5 and TIE_TOL == 0.001
    assert abs(R2_TEST - 0.55) < 1e-12 and M0_TEST == 0.4974
    assert R2_VAL == 0.9014
    assert abs(ALPHA_G_CANON - 4.281332398719396) < 1e-9


def _toy(n=60, pG=12, pH=9, seed=0, n_classes=3):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, pG))
    H = rng.standard_normal((n, pH)) * 0.5
    W = rng.standard_normal((pG + pH, n_classes)) * 0.7
    logits = np.hstack([G, H]) @ W
    y = np.argmax(logits + 0.3 * rng.standard_normal((n, n_classes)), 1)
    Y = -np.ones((n, n_classes))          # +/-1 coding as RidgeClassifier
    Y[np.arange(n), y] = 1.0
    return G, H, y, Y


@pytest.mark.parametrize("gamma", [1, 2, 8, 64])
def test_scaling_trick_equals_direct_block_penalty(gamma):
    G, H, y, Y = _toy()
    a = 0.7
    # (1) scaling trick: RidgeClassifier on [G || H/sqrt(gamma)], mapped back
    est, sc = dridge_fit(G, H, y, alpha_G=a, gamma=gamma)
    pred_trick = dridge_predict(est, sc, G, H)
    # (2) direct primal solve of the SAME block-penalty objective
    b = dridge_primal_direct(G, H, Y, alpha_G=a, gamma=gamma)
    Xc, Yc = None, None
    X = np.hstack([G, H])
    Xc = X - X.mean(0, keepdims=True)
    pred_primal = np.argmax(Xc @ b + Y.mean(0), axis=1)
    assert (pred_trick == pred_primal).mean() > 0.97
    # coefficient agreement in ORIGINAL coordinates (one-vs-all columns)
    W_trick = np.concatenate([est.coef_[:, :G.shape[1]],
                              est.coef_[:, G.shape[1]:] * sc], axis=1)
    scale = np.linalg.norm(b) + 1e-12
    rel = np.max(np.abs(W_trick.T - b)) / scale
    assert rel < 0.05, rel


def test_dual_primal_agreement():
    G, H, y, Y = _toy(seed=1)
    a, g = 0.7, 4.0
    b_pr = dridge_primal_direct(G, H, Y, a, g)
    b_du = dridge_dual_direct(G, H, Y, a, g)
    assert np.max(np.abs(
        b_pr / np.linalg.norm(b_pr) -
        b_du / np.linalg.norm(b_du))) < 1e-8


def test_gamma1_scaling_trick_is_ordinary_ridge():
    G, H, y, _ = _toy(seed=2)
    est, sc = dridge_fit(G, H, y, alpha_G=0.7, gamma=1)
    from sklearn.linear_model import RidgeClassifier
    ref = RidgeClassifier(alpha=0.7).fit(np.hstack([G, H]), y)
    assert np.allclose(est.coef_, ref.coef_)
    assert np.allclose(est.intercept_, ref.intercept_)
    assert sc == 1.0


def test_select_gamma_tie_prefers_smaller():
    cv = {1: (0.8010, 0.01, []), 2: (0.8015, 0.01, []),
          4: (0.7950, 0.01, [])}
    assert select_gamma(cv, TIE_TOL) == 1      # 0.8010 within 0.001 of best
    cv2 = {1: (0.7900, 0.01, []), 2: (0.8015, 0.01, [])}
    assert select_gamma(cv2, TIE_TOL) == 2     # outside tolerance


def test_more_penalty_shrinks_H_block_and_not_G():
    G, H, y, _ = _toy(seed=3)
    e1, s1 = dridge_fit(G, H, y, alpha_G=0.7, gamma=1)
    e8, s8 = dridge_fit(G, H, y, alpha_G=0.7, gamma=8)
    # original-coordinate H coefficients: beta_H = u * scaler
    h1 = e1.coef_[:, G.shape[1]:] * s1
    h8 = e8.coef_[:, G.shape[1]:] * s8
    assert np.linalg.norm(h8) < np.linalg.norm(h1)
    # G block barely moves (same alpha_G on the same columns)
    assert np.linalg.norm(e8.coef_[:, :G.shape[1]] -
                          e1.coef_[:, :G.shape[1]]) < \
        0.25 * np.linalg.norm(e1.coef_[:, :G.shape[1]])


def test_cv_and_selection_smoke():
    from experiments.rcmkn_haptics_dridge_seed42.core import (
        cv_macro_f1_for_gamma)
    G, H, y, _ = _toy(n=90, seed=4)
    res = cv_macro_f1_for_gamma(G, H, y, [1, 8], k_folds=3, alpha_G=0.7)
    assert set(res) == {1, 8}
    g = select_gamma(res, TIE_TOL)
    assert g in (1, 8)
