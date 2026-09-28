"""Fast tests for the R5 random-H control (no datasets needed)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rcmkn_r5_uwave_random_h_control_seed42.config import (
    DATASETS, EXPECTED, OUTCOME_RULE, P_H, RANDOM_SEEDS, SEED, TOTAL_BUDGET,
    classify_outcome,
)
from experiments.rcmkn_r5_uwave_hbudget_seed42.config import budget_split


def test_frozen_expectations_match_r5_results():
    import json
    from experiments.rcmkn_r5_uwave_hbudget_seed42.config import OUT_DIR
    for ds, exp in EXPECTED.items():
        r5 = json.load(open(os.path.join(OUT_DIR, ds, "result.json")))
        assert r5["selected_rho"] == exp["rho"]
        assert r5["r5"]["test_macro_f1"] == exp["ranked_test"]
        n_g, n_h = budget_split(exp["rho"])
        assert n_g + n_h == TOTAL_BUDGET


def test_budget_counts_consistent_with_rho():
    expected_nh = {"UWaveGestureLibraryAll": 1000,
                   "UWaveGestureLibraryX": 2999,
                   "UWaveGestureLibraryY": 1000,
                   "UWaveGestureLibraryZ": 3998}
    for ds, exp in EXPECTED.items():
        _, n_h = budget_split(exp["rho"])
        assert n_h == expected_nh[ds]


def test_classify_outcome_rule():
    assert classify_outcome(0.02, 0.005) == "A"    # > +1 std
    assert classify_outcome(-0.02, 0.005) == "C"   # < -1 std
    assert classify_outcome(0.003, 0.005) == "B"   # within 1 std
    assert classify_outcome(0.005, 0.005) == "B"   # boundary -> B
    assert classify_outcome(-0.005, 0.005) == "B"


def test_random_subsets_deterministic_and_disjoint_fixed():
    # reproduce the exact draws the runner performs
    draws = []
    for rs in RANDOM_SEEDS:
        rng = np.random.default_rng(rs)
        draws.append(np.sort(rng.choice(P_H, size=1000, replace=False)))
    assert len(set(map(tuple, draws))) == len(RANDOM_SEEDS)  # all distinct
    for d in draws:
        assert len(np.unique(d)) == 1000                    # w/o replacement
        assert d.min() >= 0 and d.max() < P_H
    # expected pairwise Jaccard for random draws ~ 1/3 of union overlap
    def jac(a, b):
        return len(np.intersect1d(a, b)) / len(np.union1d(a, b))
    pair = [jac(draws[i], draws[j]) for i in range(5)
            for j in range(i + 1, 5)]
    assert 0.0 < np.mean(pair) < 0.2


def test_independent_rng_stream_not_the_training_seed():
    assert SEED == 42
    assert all(rs > 420000 for rs in RANDOM_SEEDS)
    rng_a = np.random.default_rng(420001)
    set42 = np.random.default_rng(SEED)
    assert not np.array_equal(rng_a.permutation(10), set42.permutation(10))


def test_outcome_rule_documented():
    assert set(OUTCOME_RULE) == {"A_ranking_works", "B_ranking_neutral",
                                 "C_ranking_worse"}
    assert DATASETS == ["UWaveGestureLibraryAll", "UWaveGestureLibraryX",
                        "UWaveGestureLibraryY", "UWaveGestureLibraryZ"]
