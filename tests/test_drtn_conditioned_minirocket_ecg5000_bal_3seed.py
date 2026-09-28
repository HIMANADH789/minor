"""
Stage B tests: ECG5000_BAL 3-seed DRTN-conditioned MiniROCKET confirmation.

Covers the bug-guard checklist on the shared audited implementation plus the
ECG5000_BAL dataset resolution:
    canonical budget / allocation
    M2 vs M3: distinct arrays, distinct features, occupancy preserved
    no shared writable memory, determinism
    heterogeneity from raw responses; valid-region convention
    train+val fitting; no test-label dependency
    dataset config resolution (T, n_classes, npz path)
    canonical M0 seed-42 smoke reproduction
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (  # noqa: E402
    create_random_regime_control,
    create_shuffled_regime_control,
    M2_RNG_OFFSET,
    M3_RNG_OFFSET,
)
from experiments.drtn_conditioned_minirocket_ecg5000_bal_3seed.runner import (  # noqa: E402
    load_data, NPZ_PATH, N_CLASSES, CANONICAL_M0_REF, M0_TOLERANCE,
    ALPHAS, N_FEATURES, N_GLOBAL,
)

K = 8
SEED = 42


def _regimes(n=10, T=300, seed=5):
    rng = np.random.RandomState(seed)
    probs = np.array([0.05, 0.0, 0.08, 0.4, 0.02, 0.01, 0.04, 0.4])
    return rng.choice(K, size=(n, T), p=probs / probs.sum()).astype(np.int64)


# ---------------- dataset config ----------------
def test_dataset_config_resolution():
    assert os.path.exists(NPZ_PATH), NPZ_PATH
    d = load_data()
    assert d["n_classes"] == N_CLASSES == 5
    assert d["Xtr"].shape[1] == d["Xva"].shape[1] == d["Xte"].shape[1]
    assert d["L"] == d["Xtr"].shape[1]
    assert set(np.unique(d["ytr"])) <= set(range(5))
    assert not np.isnan(d["Xtr"].astype(np.float64)).any()


def test_split_sizes_and_alignment():
    d = load_data()
    assert len(d["Xtr"]) == len(d["ytr"])
    assert len(d["Xva"]) == len(d["yva"])
    assert len(d["Xte"]) == len(d["yte"])
    assert d["val_source"].endswith("seed42")


# ---------------- classifier protocol ----------------
def test_alpha_grid_canonical():
    assert np.allclose(ALPHAS, np.logspace(-4, 4, 20))


def test_canonical_reference_recorded():
    assert abs(CANONICAL_M0_REF - 0.6553) < 1e-9
    assert M0_TOLERANCE == pytest.approx(0.0011)


# ---------------- feature budget ----------------
def test_feature_budget():
    assert N_FEATURES == 9996 and N_GLOBAL == 4998
    A = np.zeros((4, N_GLOBAL))
    for H in [np.zeros((4, 4998)) for _ in range(3)]:
        assert np.hstack([A, H]).shape[1] == 9996


# ---------------- audited controls (shared implementation) ----------------
def test_m2_m3_distinct_and_occupancy_preserving():
    regimes = _regimes()
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.array_equal(m2, m3)
    assert not np.shares_memory(m2, m3)
    for ctrl in (m2, m3):
        for i in range(len(regimes)):
            assert np.array_equal(np.bincount(ctrl[i], minlength=K),
                                  np.bincount(regimes[i], minlength=K))


def test_independent_rng_offsets():
    assert M2_RNG_OFFSET == 900001 and M3_RNG_OFFSET == 900002
    r2 = np.random.RandomState(SEED + M2_RNG_OFFSET)
    r3 = np.random.RandomState(SEED + M3_RNG_OFFSET)
    assert not np.array_equal(r2.permutation(64), r3.permutation(64))


def test_controls_deterministic():
    regimes = _regimes()
    for fn in (create_random_regime_control, create_shuffled_regime_control):
        assert np.array_equal(fn(regimes, seed=SEED), fn(regimes, seed=SEED))


# ---------------- heterogeneity / valid-region ----------------
def test_heterogeneity_valid_region_only():
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import \
        compute_regime_heterogeneity
    T, pad, n_f = 200, 2, 3
    act = np.zeros((1, n_f, T), dtype=bool)
    valid = np.zeros((n_f, T), dtype=bool)
    valid[:, pad:T - pad] = True
    act[0] = np.random.RandomState(0).rand(n_f, T) > 0.5
    act[0][:, :pad] = False
    act[0][:, T - pad:] = False
    r1 = np.zeros((1, T), dtype=np.int64)
    r2 = r1.copy()
    r1[0, :2] = 3; r1[0, 198:] = 4      # padded-region labels
    r2[0, :2] = 0; r2[0, 198:] = 0
    h1 = compute_regime_heterogeneity(act, valid, r1)
    h2 = compute_regime_heterogeneity(act, valid, r2)
    assert np.array_equal(h1, h2)       # padded labels change nothing


def test_m1_aligned_beats_control_scale():
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import \
        compute_regime_heterogeneity
    T = 800
    act = np.zeros((1, 1, T), dtype=bool)
    act[0, 0, :T // 2] = True
    valid = np.ones((1, T), dtype=bool)
    regimes = np.zeros((1, T), dtype=np.int64)
    regimes[0, :T // 2] = 1
    regimes[0, T // 2:] = 2
    h1 = compute_regime_heterogeneity(act, valid, regimes)
    h3 = compute_regime_heterogeneity(
        act, valid, create_shuffled_regime_control(regimes, SEED))
    assert h1[0, 0] > 5 * h3[0, 0]


# ---------------- structural guards ----------------
def test_runner_variant_local_features():
    """BUG GUARD 2: prediction must use the same variant's test features."""
    p = os.path.join(ROOT, "experiments",
                     "drtn_conditioned_minirocket_ecg5000_bal_3seed", "runner.py")
    code = open(p).read()
    # the classifier loop uses explicitly local names and index alignment
    assert 'ridge.predict(F_te_v)' in code
    assert 'ridge.fit(F_tr, ytrva)' in code


def test_no_test_label_dependency():
    import re
    p = os.path.join(ROOT, "experiments",
                     "drtn_conditioned_minirocket_ecg5000_bal_3seed", "runner.py")
    code = open(p).read()
    for fn in ["create_random_regime_control", "create_shuffled_regime_control"]:
        m = re.search(rf"def {fn}\(.*?(?=\ndef |\Z)",
                      open(os.path.join(ROOT, "experiments",
                                        "drtn_conditioned_minirocket_haptics_3seed",
                                        "runner.py")).read(), re.S)
        body = m.group(0)
        assert "yte" not in body and "y_test" not in body


# ---------------- canonical M0 smoke (slow, opt-in) ----------------
@pytest.mark.slow
def test_canonical_m0_seed42_smoke_reproduction():
    """M0 at seed 42 must reproduce the canonical 0.6553 benchmark value.

    The canonical benchmark (benchmark_baselines.py) z-normalizes each split
    BEFORE MiniRocket; the experiment must use the identical convention.
    """
    from sklearn.linear_model import RidgeClassifierCV
    from experiments.drtn_conditioned_minirocket_ecg5000_bal_3seed.runner import (
        set_seed, macro_f1, znorm)
    from aeon.transformations.collection.convolution_based import MiniRocket

    d = load_data()
    set_seed(42)
    ext = MiniRocket(random_state=42, n_jobs=-1)
    ext.fit(znorm(d["Xtr"])[:, None, :].astype(np.float32))
    Ftr = ext.transform(znorm(d["Xtr"])[:, None, :].astype(np.float32))
    Fva = ext.transform(znorm(d["Xva"])[:, None, :].astype(np.float32))
    Fte = ext.transform(znorm(d["Xte"])[:, None, :].astype(np.float32))
    ytrva = np.concatenate([d["ytr"], d["yva"]])
    ridge = RidgeClassifierCV(alphas=ALPHAS).fit(
        np.vstack([Ftr, Fva]), ytrva)
    mf1 = macro_f1(d["yte"], ridge.predict(Fte))
    assert abs(mf1 - CANONICAL_M0_REF) < M0_TOLERANCE, mf1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
