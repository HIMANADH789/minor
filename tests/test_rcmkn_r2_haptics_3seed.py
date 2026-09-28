"""Tests for the R2 Haptics 3-seed robustness experiment (fast audits)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rcmkn_r2_haptics_3seed.config import (
    DATASET, ENCODER, JOINT, M0_REF, MINIROCKET_SEED, N_FEATURES, N_GLOBAL,
    N_HET, R2_REF_SEED42, SEEDS, VQ,
)
from experiments.rcmkn_haptics_seed42.config import (
    ENCODER as REF_ENCODER, JOINT as REF_JOINT, VQ as REF_VQ,
    N_FEATURES as REF_N_FEATURES, N_GLOBAL as REF_N_GLOBAL, N_HET as REF_N_HET,
    K_CODES as REF_K,
)
from experiments.rcmkn_r2_haptics_3seed.core import (
    sha16, znorm, stack_trva, h_block_diagnostics,
)
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    compute_regime_heterogeneity,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    independent_heterogeneity_recompute,
)


def test_seeds_and_dataset_fixed():
    assert SEEDS == [42, 43, 44]
    assert DATASET == "Haptics"
    assert MINIROCKET_SEED == 42


def test_config_matches_audited_r2():
    """The 3-seed config must be the audited R2 config, byte-for-byte."""
    assert ENCODER == REF_ENCODER
    assert VQ == REF_VQ
    assert JOINT == REF_JOINT
    assert N_FEATURES == REF_N_FEATURES == 9996
    assert N_GLOBAL == REF_N_GLOBAL == 4998
    assert N_HET == REF_N_HET == 4998
    assert VQ["K"] == REF_K == 8


def test_references_unchanged():
    assert R2_REF_SEED42 == 0.5500
    assert M0_REF == 0.4974


def test_znorm_is_per_sample_and_deterministic():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 200)).astype(np.float32)
    Z = znorm(X)
    assert np.array_equal(Z, znorm(X.copy()))
    assert np.all(np.abs(Z.mean(-1)) < 1e-4)
    assert np.all(np.abs(Z.std(-1) - 1) < 1e-3)
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as tr
    assert np.array_equal(Z, tr.znorm(X))


def test_stack_trva_ordering():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    S = stack_trva(a, b)
    assert np.array_equal(S[:2], a) and np.array_equal(S[2:], b)


def test_h_formula_matches_independent_reference():
    """Valid regions must be symmetric contiguous slices [p, T-p), matching
    MiniROCKET's padding convention (see _padding_groups assertion)."""
    rng = np.random.default_rng(1)
    N, T, F = 4, 60, 7
    act = rng.random((N, F, T)) > 0.6
    valid = np.ones((F, T), bool)
    valid[:2, :3] = False                    # symmetric contiguous padding
    valid[:2, T - 3:] = False
    regimes = rng.integers(0, 8, size=(N, T))
    H = compute_regime_heterogeneity(act, valid, regimes, K=8, min_occupancy=0.01)
    assert H.shape == (N, F)
    for i in range(N):
        ref = independent_heterogeneity_recompute(act[i, 3], valid[3],
                                                  regimes[i])
        assert abs(H[i, 3] - ref) < 1e-8
    # corrupting invalid positions of the padded features must not change H
    act2 = act.copy()
    act2[:, :2, :3] = ~act2[:, :2, :3]
    act2[:, :2, T - 3:] = ~act2[:, :2, T - 3:]
    H2 = compute_regime_heterogeneity(act2, valid, regimes, K=8,
                                      min_occupancy=0.01)
    assert np.array_equal(H, H2)


def test_h_diagnostics_fields():
    H = np.array([[0.0, 0.1, 0.2], [0.0, 0.0, 0.5]])
    d = h_block_diagnostics(H)
    assert d["zero_fraction"] == pytest.approx(3 / 6)
    assert d["max"] == pytest.approx(0.5)


def test_sha16_stable_and_sensitive():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 7]])
    assert sha16(a) == sha16(a.copy())
    assert sha16(a) != sha16(b)


@pytest.mark.slow
def test_minirocket_identity_quick():
    """Canonical MiniROCKET with random_state=42 is deterministic."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    rng = np.random.default_rng(2)
    X = rng.standard_normal((8, 120)).astype(np.float32)
    m1 = MiniRocket(random_state=42).fit(X[:, None, :])
    m2 = MiniRocket(random_state=42).fit(X[:, None, :])
    assert np.array_equal(m1.transform(X[:, None, :]),
                          m2.transform(X[:, None, :]))
