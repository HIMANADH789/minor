"""Fast tests for the kaggle-context2 R2 runner."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_context3_seed42.core import (
    DATASET_SPECS)


def test_expected_splits_match_context3_protocol():
    from experiments.rcmkn_r2_kaggle_context2_seed42.runner import EXPECTED
    for ds in ("ItalyPowerDemand", "FordA"):
        spec = DATASET_SPECS[ds]
        n_val = int(np.ceil(0.15 * spec["train"]))
        assert EXPECTED[ds]["val"] == n_val
        assert EXPECTED[ds]["train"] == spec["train"] - n_val
        assert EXPECTED[ds]["train"] + n_val == spec["train"]
        assert EXPECTED[ds]["test"] == spec["test"]


def test_m0_references_frozen():
    from experiments.rcmkn_r2_kaggle_context2_seed42.runner import M0_REF
    assert M0_REF == {"ItalyPowerDemand": 0.9650, "FordA": 0.9499}


def test_budget_constants():
    from experiments.rcmkn_r2_kaggle_context2_seed42.runner import (
        ALPHAS, M0_TOL, N_FEATURES, N_GLOBAL)
    assert len(ALPHAS) == 20 and M0_TOL == 0.0011
    assert N_FEATURES == 9996 and N_GLOBAL == 4998
