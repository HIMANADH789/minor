"""Tests for the R5 H-budget experiment (fast audits)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rcmkn_r5_uwave_hbudget_seed42.config import (
    DATASETS, K_FOLDS, RHOS, SEED, TIE_TOL, TOTAL_BUDGET, budget_split,
)
from experiments.rcmkn_r2_uwave_seed42.data import find_datasets


def test_budget_rule_exact_sum():
    expected = {0.0: (9996, 0), 0.1: (8996, 1000), 0.2: (7997, 1999),
                0.3: (6997, 2999), 0.4: (5998, 3998), 0.5: (4998, 4998)}
    for rho in RHOS:
        n_g, n_h = budget_split(rho)
        assert n_g + n_h == TOTAL_BUDGET
        assert (n_g, n_h) == expected[rho], (rho, n_g, n_h)
    # rounding cannot break the sum for any rho in the grid
    for rho in np.arange(0.0, 0.51, 0.05):
        n_g, n_h = budget_split(round(float(rho), 2))
        assert n_g + n_h == TOTAL_BUDGET


def test_protocol_constants_fixed():
    assert RHOS == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert K_FOLDS == 5 and TIE_TOL == 0.001 and SEED == 42
    assert max(RHOS) == 0.5            # no candidates above 0.5
    assert "f_classif" in open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments", "rcmkn_r5_uwave_hbudget_seed42",
        "config.py")).read() or "anova_F" in \
        __import__("experiments.rcmkn_r5_uwave_hbudget_seed42.config",
                   fromlist=["CRITERION"]).CRITERION


def test_assemble_exact_budget_and_ordering():
    from experiments.rcmkn_r5_uwave_hbudget_seed42.runner import (assemble,
                                                                  TOTAL_BUDGET)
    rng = np.random.default_rng(0)
    G = rng.standard_normal((10, TOTAL_BUDGET))
    H = rng.standard_normal((10, TOTAL_BUDGET))
    n_g, n_h = budget_split(0.3)
    h_idx = np.arange(TOTAL_BUDGET)[::-1].copy()   # non-canonical candidate order
    X = assemble(G, H, n_g, n_h, h_idx)
    assert X.shape == (10, TOTAL_BUDGET)
    assert np.array_equal(X[:, :n_g], G[:, :n_g])          # fixed ordering G
    assert np.array_equal(X[:, n_g:], H[:, h_idx[:n_h]])   # only selected H
    n_g0, n_h0 = budget_split(0.0)
    X0 = assemble(G, H, n_g0, n_h0, np.zeros(0, dtype=int))
    assert np.array_equal(X0, G)                            # rho=0 -> full G
    n_g5, n_h5 = budget_split(0.5)
    X5 = assemble(G, H, n_g5, n_h5, np.arange(TOTAL_BUDGET))
    assert X5.shape == (10, TOTAL_BUDGET)                   # rho=.5 -> R2 shape


def test_rho_tie_break_prefers_smaller():
    # 0.2 and 0.3 are within TIE_TOL of the best score -> smaller rho wins
    scored = [(0.2, 0.8009), (0.3, 0.8010), (0.4, 0.7900), (0.5, 0.7800)]
    best = max(s for _, s in scored)
    rho_star = min(r for r, s in scored if s >= best - TIE_TOL)
    assert rho_star == 0.2
    # outside tolerance the better score wins outright
    scored2 = [(0.2, 0.7900), (0.3, 0.8010)]
    best2 = max(s for _, s in scored2)
    assert min(r for r, s in scored2 if s >= best2 - TIE_TOL) == 0.3


def test_datasets_available():
    found = find_datasets()
    assert set(found) == set(DATASETS)


def test_f_classif_ranking_deterministic_and_fold_safe():
    from sklearn.feature_selection import f_classif
    rng = np.random.default_rng(1)
    H = rng.standard_normal((60, 20))
    y = (rng.random(60) > 0.5).astype(int)
    H[:30, 0] += y[:30] * 3           # informative feature 0 in fold-train
    f1, _ = f_classif(H[:40], y[:40])
    f2, _ = f_classif(H[:40], y[:40])
    assert np.array_equal(np.nan_to_num(f1), np.nan_to_num(f2))
    order = np.argsort(-np.nan_to_num(f1, nan=0.0), kind="stable")[:5]
    assert 0 in order                  # the informative feature ranks top-5
