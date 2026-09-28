"""Tests for the two-dataset mechanism test (fast, no heavy compute)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.dridge_mechanism_seed42.config import (
    ALPHA_G_CANON, DATASETS, GAMMAS, REFS, SEED, TIE_TOL,
)
from experiments.rcmkn_haptics_dridge_seed42.core import dridge_fit


def test_config_constants():
    assert DATASETS == ["Haptics", "UWaveGestureLibraryY"]
    assert GAMMAS == [1, 2, 4, 8, 16, 32, 64]
    assert SEED == 42 and TIE_TOL == 0.001
    assert abs(ALPHA_G_CANON - 4.281332398719396) < 1e-9


def test_frozen_refs_match_stored_artifacts():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = json.load(open(os.path.join(
        repo, "results", "rcmkn_haptics_seed42", "report.json")))
    assert h["results"]["R2"]["test_macro_f1"] == REFS["Haptics"]["R2"]
    assert h["results"]["R2"]["val_macro_f1"] == REFS["Haptics"]["R2_val"]
    u = json.load(open(os.path.join(
        "C:/temp/results/r2_uwave_seed42", "UWaveGestureLibraryY",
        "result.json")))
    assert u["results"]["R2"]["test_macro_f1"] == REFS[
        "UWaveGestureLibraryY"]["R2"]
    r5 = json.load(open(os.path.join(
        "C:/temp/results/r5_uwave_hbudget_seed42", "UWaveGestureLibraryY",
        "result.json")))
    assert r5["r5"]["test_macro_f1"] == REFS["UWaveGestureLibraryY"]["R5_test"]


def test_gamma1_identity_on_toy():
    rng = np.random.default_rng(0)
    G = rng.standard_normal((40, 12))
    H = rng.standard_normal((40, 9))
    y = (G @ rng.standard_normal(12) + H @ rng.standard_normal(9)
         > 0).astype(int)
    est, sc = dridge_fit(G, H, y, ALPHA_G_CANON, gamma=1)
    from sklearn.linear_model import RidgeClassifier
    ref = RidgeClassifier(alpha=ALPHA_G_CANON).fit(np.hstack([G, H]), y)
    assert np.allclose(est.coef_, ref.coef_) and sc == 1.0


def test_tie_rule_smaller_gamma():
    cv = {"1": 0.8010, "2": 0.8015, "4": 0.7950}
    best = max(cv.values())
    assert min(int(g) for g, v in cv.items()
               if v >= best - TIE_TOL) == 1
