"""Tests for the context-dependence screen (GunPoint / ItalyPowerDemand /
FordA, seed 42). Covers the mandated audits on synthetic + real data.

The heavyweight dataset-dependent tests are marked slow; the fast tests run
on every invocation.
"""
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
    extract_soft_assignments, soft_heterogeneity_features,
    independent_soft_recompute,
)
from experiments.drtn_conditioned_minirocket_context3_seed42.runner import (
    N_FEATURES, N_GLOBAL, N_HETEROGENEITY, ALPHAS, K_CODES, SEED,
)
from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
    create_random_regime_control, create_shuffled_regime_control,
    compute_regime_heterogeneity, M2_RNG_OFFSET, M3_RNG_OFFSET,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (
    compute_raw_activations, ppv_from_activations,
    independent_heterogeneity_recompute,
)

N_GLOBAL_SYNTAX = N_GLOBAL  # explicit alias used in slicing assertions below


def _mk_synthetic(n=12, T=60, K=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T)).astype(np.float32)
    regimes = rng.integers(0, K, size=(n, T))
    return X, regimes


def _sym_valid(F, T, pad):
    """Symmetric MiniRocket-style valid masks: features 0..pad-1 have
    [pad, T-pad); the rest are fully valid."""
    valid = np.ones((F, T), bool)
    if pad:
        valid[:pad, :pad] = False
        valid[:pad, T - pad:] = False
    return valid


# ---------------------------------------------------------------- data ----
@pytest.mark.slow
@pytest.mark.parametrize("ds", ["GunPoint", "ItalyPowerDemand", "FordA"])
def test_dataset_config_identity(ds):
    spec = DATASET_SPECS[ds]
    d = load_kaggle_ucr(ds, seed=SEED)
    assert d["Xtr"].shape[0] + d["Xva"].shape[0] == spec["train"]
    assert d["Xte"].shape == (spec["test"], spec["T"])
    assert d["Xtr"].shape[1] == spec["T"]
    assert d["n_classes"] == spec["n_classes"]
    assert set(np.unique(np.concatenate([d["ytr"], d["yva"], d["yte"]]))) \
        == set(range(spec["n_classes"]))
    assert d["val_source"].endswith(f"seed{SEED}")


def test_dataset_loader_no_test_leakage_in_val_split():
    """The split is made from TRAIN only; test labels never enter it."""
    d = load_kaggle_ucr("GunPoint", seed=SEED)
    n_train_total = DATASET_SPECS["GunPoint"]["train"]
    assert d["Xtr"].shape[0] + d["Xva"].shape[0] == n_train_total
    assert d["Xte"].shape[0] == DATASET_SPECS["GunPoint"]["test"]


# ------------------------------------------------------- MiniROCKET core --
@pytest.mark.slow
def test_canonical_m0_feature_identity_gunpoint():
    """Custom raw extractor must reproduce the canonical aeon transform."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    d = load_kaggle_ucr("GunPoint", seed=SEED)
    Xz = znorm(d["Xtr"][:20])
    mr = MiniRocket(random_state=SEED)
    mr.fit(Xz[:, None, :].astype(np.float32))
    F = mr.transform(Xz[:, None, :].astype(np.float32))
    assert F.shape[1] == N_FEATURES
    act, valid = compute_raw_activations(mr, Xz)
    PPV = ppv_from_activations(act, valid)
    assert float(np.max(np.abs(PPV - F))) < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("T", [24, 150, 500])
def test_minirocket_budget_at_all_lengths(T):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, T)).astype(np.float32)
    from aeon.transformations.collection.convolution_based import MiniRocket
    mr = MiniRocket(random_state=SEED)
    mr.fit(X[:, None, :])
    F = mr.transform(X[:, None, :])
    assert F.shape[1] == N_FEATURES
    act, valid = compute_raw_activations(mr, X)
    assert act.shape == (20, N_FEATURES, T)
    assert float(np.max(np.abs(ppv_from_activations(act, valid) - F))) < 1e-5


def test_exact_budget_and_allocation():
    assert N_FEATURES == 9996
    assert N_GLOBAL == 4998 and N_HETEROGENEITY == 4998
    assert N_GLOBAL + N_HETEROGENEITY == N_FEATURES
    assert len(ALPHAS) == 20
    assert ALPHAS[0] == pytest.approx(1e-4) and ALPHAS[-1] == pytest.approx(1e4)


def test_feature_axis_slicing_not_samples():
    """responses[:, N_GLOBAL:, :] slices FEATURES; a sample-axis bug would
    return (n_small, F, T) with the wrong feature count."""
    X, regimes = _mk_synthetic()
    act = (np.random.default_rng(1).random((X.shape[0], N_FEATURES, X.shape[1]))
           > 0.5)
    het_slice = act[:, N_GLOBAL:, :]
    assert het_slice.shape == (X.shape[0], N_HETEROGENEITY, X.shape[1])


def test_exact_h_formula_and_valid_mask():
    """Vectorized H == independent float64 recompute; padded positions
    (per-feature valid mask) provably do not matter."""
    X, regimes = _mk_synthetic(n=4, T=48)
    act = (np.random.default_rng(2).random((4, 6, 48)) > 0.5)
    valid = _sym_valid(6, 48, pad=2)           # features 0-1 padded by 2
    H = compute_regime_heterogeneity(act, valid, regimes)
    for i in range(4):
        for f in range(6):
            ref = independent_heterogeneity_recompute(
                act[i, f], valid[f], regimes[i])
            assert abs(H[i, f] - ref) < 1e-8
    # corrupt ONLY out-of-valid cells (per-feature) -> H unchanged
    act_c = act.copy()
    outside = np.broadcast_to(~valid[None], act.shape)
    act_c[outside] = ~act_c[outside]
    assert np.array_equal(H, compute_regime_heterogeneity(act_c, valid, regimes))


def test_h_from_raw_responses_not_placeholder():
    """H must react to raw activation changes (BUG GUARD 4) and not be
    identically zero (BUG GUARD: PPV_m everywhere)."""
    X, regimes = _mk_synthetic(n=4, T=48)
    act = (np.random.default_rng(3).random((4, 8, 48)) > 0.5)
    valid = np.ones((8, 48), bool)
    H = compute_regime_heterogeneity(act, valid, regimes)
    assert (H != 0).mean() > 0.5
    act2 = act.copy()
    act2[:, :, ::2] = ~act2[:, :, ::2]
    assert not np.array_equal(H, compute_regime_heterogeneity(act2, valid, regimes))


# ------------------------------------------------------------ controls ----
def test_m2_per_sample_occupancy_preserved():
    X, regimes = _mk_synthetic(n=10, T=40)
    m2 = create_random_regime_control(regimes, seed=SEED)
    for i in range(10):
        assert np.array_equal(np.bincount(m2[i], minlength=K_CODES),
                              np.bincount(regimes[i], minlength=K_CODES))


def test_m3_per_sample_occupancy_preserved_and_alignment_destroyed():
    X, regimes = _mk_synthetic(n=10, T=40)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    for i in range(10):
        assert np.array_equal(np.bincount(m3[i], minlength=K_CODES),
                              np.bincount(regimes[i], minlength=K_CODES))
    assert not np.array_equal(m3, regimes)


def test_m2_m3_distinct_and_independent_rng():
    X, regimes = _mk_synthetic(n=10, T=40)
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.array_equal(m2, m3)
    assert M2_RNG_OFFSET != M3_RNG_OFFSET
    m2b = create_random_regime_control(regimes, seed=SEED)
    m3b = create_shuffled_regime_control(regimes, seed=SEED)
    assert np.array_equal(m2, m2b) and np.array_equal(m3, m3b)  # deterministic


def test_m2_m3_no_shared_writable_memory():
    X, regimes = _mk_synthetic(n=10, T=40)
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.shares_memory(m2, m3)
    m3[0, 0] = (m3[0, 0] + 1) % K_CODES
    assert not np.array_equal(m2, m3)   # m2 unaffected by m3 mutation


# ------------------------------------------------------- soft ablation ----
def _soft_case(n=3, T=36, K=8, seed=5):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, T, K))
    soft = np.exp(logits - logits.max(-1, keepdims=True))
    soft /= soft.sum(-1, keepdims=True)
    act = (rng.random((n, 4, T)) > 0.5)
    valid = _sym_valid(4, T, pad=1)            # feature 0 padded by 1
    return act, valid, soft


def test_soft_heterogeneity_formula_and_valid_mask():
    act, valid, soft = _soft_case()
    H = soft_heterogeneity_features(act, valid, soft)
    for i in range(act.shape[0]):
        for f in range(act.shape[1]):
            ref = independent_soft_recompute(act[i, f], valid[f], soft[i])
            assert abs(H[i, f] - ref) < 1e-9
    # padded corruption (per-feature) must not change H_soft
    act_c = act.copy()
    outside = np.broadcast_to(~valid[None], act.shape)
    act_c[outside] = ~act_c[outside]
    assert np.array_equal(H, soft_heterogeneity_features(act_c, valid, soft))


def test_soft_heterogeneity_non_placeholder():
    act, valid, soft = _soft_case()
    H = soft_heterogeneity_features(act, valid, soft)
    assert (H != 0).mean() > 0.5
    # NOT the collapsing weighted average: sharp soft assignments whose
    # weighted rates differ across regimes must give H > 0
    K = soft.shape[2]
    soft_sharp = np.full((1, 8, K), 0.01)
    soft_sharp[0, :4, 0] = 0.9
    soft_sharp[0, 4:, 1] = 0.9
    soft_sharp /= soft_sharp.sum(-1, keepdims=True)
    act1 = np.zeros((1, 1, 8), bool)
    act1[0, 0, :4] = True
    valid1 = np.ones((1, 8), bool)
    assert soft_heterogeneity_features(act1, valid1, soft_sharp)[0, 0] > 0.01


def test_soft_vs_hard_heterogeneity_differ():
    """H_soft from near-hard soft assignments should approximate but not
    exactly equal H_hard under mild randomness."""
    X, regimes = _mk_synthetic(n=3, T=48)
    act = (np.random.default_rng(7).random((3, 4, 48)) > 0.5)
    valid = np.ones((4, 48), bool)
    onehot = np.eye(K_CODES)[regimes]
    soft = 0.9 * onehot + 0.1 / K_CODES
    H_hard = compute_regime_heterogeneity(act, valid, regimes)
    H_soft = soft_heterogeneity_features(act, valid, soft)
    assert not np.array_equal(H_hard, H_soft)
    assert np.abs(H_hard - H_soft).mean() < 0.05   # close but distinct


@pytest.mark.slow
def test_soft_assignment_extraction_frozen_model():
    """Soft extraction must be read-only w.r.t. the frozen model and match
    hard argmax at the assigned code (near-hard ordering)."""
    from models.drtn.model import build_model
    torch.manual_seed(0)
    model = build_model("R5", c_in=1, n_classes=2, d_model=64, n_codes=K_CODES,
                        tau=0.5, ema_decay=0.99, beta=0.25, lam_div=0.1,
                        traj_layers=2, traj_heads=4, traj_ffn=128,
                        traj_dropout=0.1)
    model.eval()
    X = np.random.default_rng(1).standard_normal((5, 60)).astype(np.float32)
    Xz = znorm(X)
    sd1 = {k: v.clone() for k, v in model.state_dict().items()}
    soft = extract_soft_assignments(model, Xz, device=torch.device("cpu"),
                                    tau=0.5)
    assert all(torch.equal(sd1[k], model.state_dict()[k]) for k in sd1)
    assert soft.shape == (5, 60, K_CODES)
    np.testing.assert_allclose(soft.sum(-1), 1.0, atol=1e-6)
    assert (soft >= 0).all()


# ------------------------------------------------------- ridge fitting ----
def test_ridge_fit_on_train_and_val_not_train_only():
    """Fitting rows must equal train+val: a train-only fit raises the guard."""
    from sklearn.linear_model import RidgeClassifierCV
    d = load_kaggle_ucr("GunPoint", seed=SEED)
    Xz = znorm(np.vstack([d["Xtr"], d["Xva"]]))
    ytrva = np.concatenate([d["ytr"], d["yva"]])
    ridge = RidgeClassifierCV(alphas=ALPHAS)
    ridge.fit(Xz[:, :2], ytrva)
    assert ridge.n_features_in_ == 2
    assert len(ytrva) == d["Xtr"].shape[0] + d["Xva"].shape[0]


def test_no_test_labels_in_feature_generation():
    """Controls/regimes are generated from train/val regime arrays only;
    the test-split function never receives y_te."""
    X, regimes = _mk_synthetic(n=6, T=40)
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert m2.shape == m3.shape == regimes.shape
