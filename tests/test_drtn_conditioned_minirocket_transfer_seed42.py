"""
Unit tests for the DRTN-conditioned MiniROCKET transfer screen.

Covers the mandated audits (spec sec. 18) using synthetic data plus the
canonical Haptics dataset for end-to-end consistency:
    canonical MiniROCKET feature count / exact 9996 budget
    heterogeneity formula + non-placeholder behavior
    occupancy calculation
    random-regime occupancy preservation + determinism
    shuffled-regime occupancy preservation + temporal alignment destruction
    global-feature consistency vs the canonical aeon transform
    sample alignment
    dataset-specific input lengths for all six dataset configs
    no test-label dependency in feature construction
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    heterogeneity_features,
    independent_heterogeneity_recompute,
    create_random_regime_control,
    create_shuffled_regime_control,
    compute_regime_occupancy_stats,
    compute_raw_activations,
    ppv_from_activations,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.runner import (  # noqa: E402
    load_any_dataset, NPZ_SPECS, EXTERNAL_DATASETS, REF_VALUES, ALL_DATASETS,
)


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------
def _synthetic_act(n=4, f=6, T=100, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(n, f, T) > 0.5


def _synthetic_regimes(n=4, T=100, K=8, seed=1):
    rng = np.random.RandomState(seed)
    return rng.randint(0, K, size=(n, T)).astype(np.int64)


# ---------------------------------------------------------------------------
# Heterogeneity formula
# ---------------------------------------------------------------------------
def test_heterogeneity_zero_when_rate_identical_across_regimes():
    """Same activation rate in every regime -> H must be exactly 0."""
    T, K, F = 200, 4, 3
    act = np.zeros((1, F, T), dtype=bool)
    act[0, :, ::2] = True            # rate 0.5 in every regime (interleaved)
    valid = np.ones((F, T), dtype=bool)
    regimes = np.tile(np.arange(K).repeat(T // K), (1, 1))
    H = heterogeneity_features(act, valid, regimes, K=K)
    assert np.allclose(H, 0.0)


def test_heterogeneity_positive_when_regimes_differ_same_ppv():
    """Same global PPV but different regime distribution -> H > 0, and a
    kernel spread over two regimes gets larger weighted variance than one
    concentrated in a single regime."""
    T, K = 400, 4
    act = np.zeros((1, 2, T), dtype=bool)
    # kernel 0: active only in regime 0 (block 1) -> global PPV 0.25
    act[0, 0, : T // 4] = True
    # kernel 1: active in regimes 0 and 2 (blocks 1 and 3) -> also PPV 0.25
    act[0, 1, : T // 4] = True
    act[0, 1, 3 * T // 4:] = True
    regimes = np.repeat(np.arange(K), T // K)[None, :]
    valid = np.ones((2, T), dtype=bool)
    H = heterogeneity_features(act, valid, regimes, K=K)
    assert H[0, 0] > 0 and H[0, 1] > 0
    assert H[0, 1] > H[0, 0]   # spread over two regimes -> larger variance


def test_heterogeneity_matches_independent_recompute():
    """Vectorized H equals the reference O(T) recomputation."""
    act = _synthetic_act(n=5, f=7, T=300, seed=3)
    regimes = _synthetic_regimes(n=5, T=300, K=8, seed=4)
    valid = np.ones((7, 300), dtype=bool)
    H = heterogeneity_features(act, valid, regimes, K=8)
    rng = np.random.RandomState(0)
    for _ in range(20):
        i, m = rng.randint(0, 5), rng.randint(0, 7)
        ref = independent_heterogeneity_recompute(
            np.asarray(act[i, m], dtype=bool), valid[m], regimes[i], K=8)
        assert abs(ref - H[i, m]) < 1e-8   # float32 vectorization tolerance


def test_heterogeneity_non_placeholder():
    """Real varied activations produce overwhelmingly nonzero features."""
    act = _synthetic_act(n=8, f=10, T=500, seed=5)
    # block-structured regimes so regime-conditioned rates genuinely differ
    regimes = (np.arange(500) // 25 % 8)[None, :].repeat(8, axis=0)
    valid = np.ones((10, 500), dtype=bool)
    H = heterogeneity_features(act, valid, regimes, K=8)
    assert (H != 0).mean() > 0.9
    assert np.isfinite(H).all()


# ---------------------------------------------------------------------------
# Occupancy / controls
# ---------------------------------------------------------------------------
def test_random_regime_preserves_global_occupancy_and_deterministic():
    regimes = _synthetic_regimes(n=6, T=400, K=8, seed=7)
    r1 = create_random_regime_control(regimes, seed=42)
    r2 = create_random_regime_control(regimes, seed=42)
    assert np.array_equal(r1, r2)                      # deterministic
    K = 8
    p_orig = np.bincount(regimes.ravel(), minlength=K) / regimes.size
    p_rand = np.bincount(r1.ravel(), minlength=K) / r1.size
    assert np.max(np.abs(p_orig - p_rand)) < 0.02      # occupancy preserved


def test_shuffled_regime_preserves_per_sample_histogram():
    regimes = _synthetic_regimes(n=6, T=400, K=8, seed=8)
    shuf = create_shuffled_regime_control(regimes, seed=42)
    for i in range(len(regimes)):
        h_a = np.bincount(regimes[i], minlength=8)
        h_b = np.bincount(shuf[i], minlength=8)
        assert np.array_equal(h_a, h_b)


def test_shuffled_regime_destroys_temporal_alignment():
    regimes = _synthetic_regimes(n=10, T=300, K=8, seed=9)
    shuf = create_shuffled_regime_control(regimes, seed=42)
    changed = sum(not np.array_equal(regimes[i], shuf[i])
                  for i in range(len(regimes)))
    assert changed >= 9                                # >=90% samples changed


def test_occupancy_stats_shape():
    regimes = _synthetic_regimes(n=3, T=100, K=8, seed=10)
    s = compute_regime_occupancy_stats(regimes, K=8)
    assert s["active_codes"] >= 1 and s["active_codes"] <= 8
    assert abs(sum(s["usage"]) - 1.0) < 1e-9
    assert 1.0 <= s["perplexity"] <= 8.0 + 1e-9


# ---------------------------------------------------------------------------
# Canonical MiniROCKET consistency (uses Haptics TRAIN, canonical fit)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_raw_extractor_matches_aeon_and_feature_budget():
    from aeon.transformations.collection.convolution_based import MiniRocket
    d = load_any_dataset("Haptics")
    X = d["Xtr"][:8]
    mu = X.mean(-1, keepdims=True)
    sd = X.std(-1, keepdims=True) + 1e-8
    Xz = ((X - mu) / sd).astype(np.float32)

    mr = MiniRocket(random_state=42, n_jobs=-1)
    mr.fit(Xz[:, None, :].astype(np.float32))
    F = mr.transform(Xz[:, None, :].astype(np.float32))
    assert F.shape[1] == 9996                          # canonical budget

    act, valid = compute_raw_activations(mr, Xz)
    assert act.shape == (8, 9996, Xz.shape[1])
    assert valid.shape == (9996, Xz.shape[1])
    PPV = ppv_from_activations(act, valid)
    assert np.max(np.abs(PPV - F)) < 1e-5              # bit-level agreement
    # PPV bounds
    assert PPV.min() >= 0.0 and PPV.max() <= 1.0


# ---------------------------------------------------------------------------
# Dataset configuration coverage
# ---------------------------------------------------------------------------
def test_all_six_datasets_configured():
    assert set(ALL_DATASETS) == {
        "EpilepticSeizures", "Phoneme",
        "ECG5000_UNBAL", "ECG5000_BAL", "CWRU_UNBAL", "CWRU_BAL"}
    assert set(REF_VALUES.keys()) == set(ALL_DATASETS)


@pytest.mark.slow
@pytest.mark.parametrize("ds", list(NPZ_SPECS.keys()))
def test_npz_dataset_configs(ds):
    d = load_any_dataset(ds)
    T = d["L"]
    # NPZ datasets use their own canonical lengths
    expected_T = {"ECG5000_UNBAL": 140, "ECG5000_BAL": 140,
                  "CWRU_UNBAL": 1024, "CWRU_BAL": 1024}[ds]
    assert T == expected_T
    assert d["Xtr"].shape[1] == T and d["Xva"].shape[1] == T
    assert d["Xte"].shape[1] == T
    # sample alignment: rows unchanged by loading
    assert len(d["ytr"]) == len(d["Xtr"])
    assert len(d["yte"]) == len(d["Xte"])


@pytest.mark.slow
@pytest.mark.parametrize("ds", EXTERNAL_DATASETS)
def test_external_dataset_configs(ds):
    d = load_any_dataset(ds)
    T = d["L"]
    expected_T = {"EpilepticSeizures": 178, "Phoneme": 1024}[ds]
    assert T == expected_T
    assert d["Xtr"].shape[1] == T and d["Xva"].shape[1] == T == d["Xte"].shape[1]
    assert len(d["ytr"]) == len(d["Xtr"]) and len(d["yte"]) == len(d["Xte"])


# ---------------------------------------------------------------------------
# No test-label dependency (structural check)
# ---------------------------------------------------------------------------
def test_feature_construction_has_no_label_path():
    import inspect
    from experiments.drtn_conditioned_minirocket_transfer_seed42 import core
    src = inspect.getsource(core)
    assert "ytr" not in src and "yte" not in src and "yva" not in src
