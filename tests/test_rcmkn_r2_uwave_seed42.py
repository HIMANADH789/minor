"""Tests for the R2 UWave Motion generalization experiment (fast audits)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rcmkn_r2_uwave_seed42.config import (
    CANONICAL, DATASETS, M0_TOL, SEED, VAL_FRAC,
)
from experiments.rcmkn_r2_uwave_seed42.data import find_datasets, znorm
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    create_random_regime_control, create_shuffled_regime_control,
    compute_regime_heterogeneity,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    independent_heterogeneity_recompute,
)
from experiments.rcmkn_haptics_seed42.config import (
    ENCODER, JOINT, K_CODES, N_FEATURES, VQ,
)


def test_dataset_specs_canonical():
    assert set(DATASETS) == set(CANONICAL)
    assert CANONICAL["UWaveGestureLibraryAll"]["T"] == 945
    for ds in ("UWaveGestureLibraryX", "UWaveGestureLibraryY",
               "UWaveGestureLibraryZ"):
        assert CANONICAL[ds]["T"] == 315
    for ds in DATASETS:
        assert CANONICAL[ds]["train"] == 896
        assert CANONICAL[ds]["test"] == 3582
        assert CANONICAL[ds]["n_classes"] == 8


def test_local_datasets_found():
    found = find_datasets()
    assert set(found) == set(DATASETS)
    for ds, p in found.items():
        assert os.path.exists(p["train"]) and os.path.exists(p["test"])
        assert os.path.getsize(p["train"]) > 0


def test_val_fraction_matches_repo_rule():
    assert VAL_FRAC == 0.15 and SEED == 42
    assert abs(int(896 * VAL_FRAC) - 134) <= 2  # ~135 val, ~761 train


def test_znorm_per_sample():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 100)).astype(np.float32) * 5 + 3
    Z = znorm(X)
    assert np.all(np.abs(Z.mean(-1)) < 1e-4)
    assert np.all(np.abs(Z.std(-1) - 1) < 1e-3)
    assert np.array_equal(Z, znorm(X.copy()))


def test_r2_config_unchanged_from_haptics():
    assert ENCODER["channels"] == (32, 32, 64, 64)
    assert ENCODER["kernel_sizes"] == (3, 5, 7, 9)
    assert ENCODER["dilations"] == (1, 2, 4, 8)
    assert ENCODER["d_model"] == 32          # 64 -> 32 projection
    assert ENCODER["mask_ratio"] == 0.10 and ENCODER["span_len"] == 16
    assert VQ["K"] == K_CODES == 8
    assert JOINT["lambda_cls"] == 0.10
    assert N_FEATURES == 9996
    assert M0_TOL == 0.0011


def test_controls_occupancy_preserving_and_distinct():
    rng = np.random.default_rng(1)
    regimes = rng.integers(0, K_CODES, size=(6, 80))
    c1 = create_random_regime_control(regimes, seed=SEED, K=K_CODES)
    c2 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.array_equal(c1, c2)
    assert not np.shares_memory(c1, c2)
    for ctrl in (c1, c2):
        for i in range(len(regimes)):
            assert np.array_equal(
                np.bincount(ctrl[i], minlength=K_CODES),
                np.bincount(regimes[i], minlength=K_CODES))


def test_h_formula_and_valid_region_small_t():
    rng = np.random.default_rng(2)
    N, T, F = 3, 40, 6
    act = rng.random((N, F, T)) > 0.6
    valid = np.ones((F, T), bool)
    valid[:2, :3] = False
    valid[:2, T - 3:] = False
    regimes = rng.integers(0, K_CODES, size=(N, T))
    H = compute_regime_heterogeneity(act, valid, regimes, K=K_CODES,
                                     min_occupancy=0.01)
    for i in range(N):
        ref = independent_heterogeneity_recompute(act[i, 3], valid[3],
                                                  regimes[i])
        assert abs(H[i, 3] - ref) < 1e-8
    act2 = act.copy()
    act2[:, :2, :3] = ~act2[:, :2, :3]
    act2[:, :2, T - 3:] = ~act2[:, :2, T - 3:]
    assert np.array_equal(
        H, compute_regime_heterogeneity(act2, valid, regimes, K=K_CODES,
                                        min_occupancy=0.01))


def test_ssl_encoder_causal_and_undownsampled():
    from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel
    torch.manual_seed(0)
    model = RCMKNContextModel(n_classes=8).eval()
    x = torch.randn(1, 1, 90)
    with torch.no_grad():
        z = model.encoder(x)
        x2 = x.clone()
        x2[:, 0, -1] += 10.0
        z2 = model.encoder(x2)
    assert z.shape == (1, 90, 32)            # no downsampling, d=32
    assert torch.equal(z[:, :-1, :], z2[:, :-1, :])   # causal
    assert not torch.equal(z[:, -1, :], z2[:, -1, :])


@pytest.mark.slow
def test_minrocket_feature_count_uwave_t315():
    if find_datasets is None:  # pragma: no cover
        pytest.skip("datasets unavailable")
    from aeon.transformations.collection.convolution_based import MiniRocket
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, 315)).astype(np.float32)
    mr = MiniRocket(random_state=SEED).fit(X[:, None, :])
    assert mr.transform(X[:, None, :]).shape[1] == N_FEATURES
