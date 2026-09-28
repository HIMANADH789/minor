"""Tests for the R2-on-GunPoint experiment (fast audits)."""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_context3_seed42.core import (
    DATASET_SPECS, load_kaggle_ucr, znorm,
)
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
from experiments.rcmkn_haptics_seed42.model import RCMKNContextModel

SEED = 42
EXPECTED_SPLIT = {"train": 42, "val": 8, "test": 150, "T": 150,
                  "n_classes": 2}
M0_REF = 0.9933


def test_context3_gunpoint_reference_is_recorded():
    """The gate must point at the published context3 canonical value."""
    assert M0_REF == 0.9933
    spec = DATASET_SPECS["GunPoint"]
    assert spec["train"] == 50 and spec["test"] == 150 and spec["T"] == 150


def test_loader_split_and_labels():
    if not os.path.isdir(os.path.join(ROOT, "data", "kaggle", "GunPoint")):
        pytest.skip("Kaggle GunPoint data not installed")
    d = load_kaggle_ucr("GunPoint", seed=SEED)
    assert {"train": len(d["Xtr"]), "val": len(d["Xva"]),
            "test": len(d["Xte"]), "T": d["L"],
            "n_classes": d["n_classes"]} == EXPECTED_SPLIT
    assert sorted(set(d["ytr"].tolist()) | set(d["yte"].tolist())) == [0, 1]
    assert d["val_source"].startswith("stratified_15pct_of_train_seed42")


def test_loader_split_is_deterministic():
    if not os.path.isdir(os.path.join(ROOT, "data", "kaggle", "GunPoint")):
        pytest.skip("Kaggle GunPoint data not installed")
    d1 = load_kaggle_ucr("GunPoint", seed=SEED)
    d2 = load_kaggle_ucr("GunPoint", seed=SEED)
    assert np.array_equal(d1["Xtr"], d2["Xtr"])
    assert np.array_equal(d1["ytr"], d2["ytr"])


def test_znorm_matches_project_convention():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 60)).astype(np.float32)
    Z = znorm(X)
    assert np.all(np.abs(Z.mean(-1)) < 1e-4)
    assert np.all(np.abs(Z.std(-1) - 1) < 1e-3)


def test_controls_occupancy_preserving_and_distinct():
    rng = np.random.default_rng(3)
    regimes = rng.integers(0, K_CODES, size=(7, 50))
    c1 = create_random_regime_control(regimes, seed=SEED, K=K_CODES)
    c2 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.array_equal(c1, c2)
    assert not np.shares_memory(c1, c2)
    for ctrl in (c1, c2):
        for i in range(len(regimes)):
            assert np.array_equal(
                np.bincount(ctrl[i], minlength=K_CODES),
                np.bincount(regimes[i], minlength=K_CODES))


def test_h_formula_and_valid_region():
    rng = np.random.default_rng(5)
    N, T, F = 3, 50, 6
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


def test_context_model_config_unchanged():
    """The Haptics R2 config is used verbatim (no GunPoint tuning)."""
    assert ENCODER["channels"] == (32, 32, 64, 64)
    assert ENCODER["mask_ratio"] == 0.10 and ENCODER["span_len"] == 16
    assert VQ["K"] == K_CODES == 8
    assert JOINT["lambda_cls"] == 0.10
    assert N_FEATURES == 9996


def test_context_model_forward_shapes_gunpoint_t():
    torch.manual_seed(0)
    model = RCMKNContextModel(n_classes=2).eval()
    x = torch.randn(2, 1, 150)
    with torch.no_grad():
        z = model.encoder(x)
    assert z.shape == (2, 150, 32)


@pytest.mark.slow
def test_minirocket_feature_count_gunpoint_t():
    if not os.path.isdir(os.path.join(ROOT, "data", "kaggle", "GunPoint")):
        pytest.skip("Kaggle GunPoint data not installed")
    from aeon.transformations.collection.convolution_based import MiniRocket
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 150)).astype(np.float32)
    mr = MiniRocket(random_state=SEED).fit(X[:, None, :])
    assert mr.transform(X[:, None, :]).shape[1] == N_FEATURES
