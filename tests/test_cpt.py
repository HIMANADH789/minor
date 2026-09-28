"""Unit tests for TURS-CPT (src/features/cpt.py)."""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.features.cpt import (
    estimate_period,
    estimate_period_autocorrelation,
    estimate_period_fft,
    _cpt_from_taus,
    cpt_transform,
    rank_kernels_by_discriminative_power,
    select_top_k_transformers,
    period_summary,
)
from src.features.ktm import ktm_transform, _activation_stats
from aeon.transformations.collection.convolution_based import MiniRocket


# ---------------------------------------------------------------------------
# 1. Period estimation tests
# ---------------------------------------------------------------------------

def test_acf_detects_periodic_signal():
    """Autocorrelation should detect the period of a clean sine wave."""
    T = 200
    P_true = 50
    t = np.arange(T, dtype=np.float64)
    x = np.sin(2 * np.pi * t / P_true).astype(np.float32)
    p, method, q = estimate_period_autocorrelation(x, max_lag=T // 2)
    assert method == 'ACF'
    assert abs(p - P_true) <= 2  # allow ±2 samples


def test_fft_fallback_detects_periodic():
    """FFT fallback should also detect sine period."""
    T = 200
    P_true = 40
    t = np.arange(T, dtype=np.float64)
    x = np.sin(2 * np.pi * t / P_true).astype(np.float32)
    p, method, q = estimate_period_fft(x, min_period=5, max_period=T // 2)
    assert method == 'FFT'
    assert abs(p - P_true) <= 2


def test_combined_estimator_sine():
    """Combined estimator on sine wave."""
    T = 200
    P_true = 60
    t = np.arange(T, dtype=np.float64)
    x = np.sin(2 * np.pi * t / P_true).astype(np.float32)
    p, method, q = estimate_period(x)
    assert abs(p - P_true) <= 5  # generous tolerance


def test_estimator_never_returns_none():
    """Period estimator must always return a valid period."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        x = rng.normal(size=100).astype(np.float32)
        p, method, q = estimate_period(x)
        assert p is not None
        assert p >= 2
        assert method in ('ACF', 'FFT', 'FALLBACK')


# ---------------------------------------------------------------------------
# 2. Phase computation tests (numba _cpt_from_taus)
# ---------------------------------------------------------------------------

def test_cpt_zero_activations():
    taus = np.array([], dtype=np.int64)
    conc, phase = _cpt_from_taus(taus, 100, 50.0)
    assert conc == 0.0
    assert phase == 0.0


def test_cpt_single_activation():
    taus = np.array([25], dtype=np.int64)
    P = 100.0
    conc, phase = _cpt_from_taus(taus, 100, P)
    assert conc == 1.0  # single point has concentration 1
    assert abs(phase - 0.25) < 1e-10  # 25 % 100 / 100 = 0.25


def test_cpt_all_same_phase():
    """All activations at same phase -> concentration = 1."""
    P = 50.0
    taus = np.array([10, 60, 110, 160], dtype=np.int64)  # all at phase 0.2
    conc, phase = _cpt_from_taus(taus, 200, P)
    assert abs(conc - 1.0) < 1e-10
    assert abs(phase - 0.2) < 1e-10


def test_cpt_uniform_spread():
    """Activations uniformly spread -> concentration ~ 0."""
    P = 10.0
    taus = np.arange(0, 100, dtype=np.int64)  # every position
    conc, phase = _cpt_from_taus(taus, 100, P)
    assert conc < 0.05  # nearly zero concentration


def test_cpt_two_opposite_phases():
    """Two activations at opposite phases -> concentration = 0."""
    P = 100.0
    taus = np.array([0, 50], dtype=np.int64)  # phase 0 and 0.5
    conc, phase = _cpt_from_taus(taus, 100, P)
    assert conc < 0.01  # nearly zero


def test_cpt_phase_range():
    """Phase must be in [0, 1)."""
    rng = np.random.default_rng(42)
    P = 37.0
    for _ in range(20):
        k = rng.integers(1, 50)
        taus = rng.integers(0, 200, size=k).astype(np.int64)
        conc, phase = _cpt_from_taus(taus, 200, P)
        assert 0 <= conc <= 1
        assert 0 <= phase < 1


# ---------------------------------------------------------------------------
# 3. CPT transform integration test
# ---------------------------------------------------------------------------

def _make_znorm(n=6, T=140, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T)).astype(np.float32)
    return ((X - X.mean(1, keepdims=True))
            / (X.std(1, keepdims=True) + 1e-8)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted_cpt():
    X = _make_znorm(n=6, T=140, seed=0)
    mr = MiniRocket(random_state=42)
    mr.fit_transform(X[:, None, :])
    # Estimate periods
    periods = np.array([estimate_period(X[i])[0] for i in range(len(X))])
    conc, phase, counts = cpt_transform(X, mr, periods, return_counts=True)
    return X, mr, periods, conc, phase, counts


def test_cpt_no_nan_inf(fitted_cpt):
    _, _, _, conc, phase, _ = fitted_cpt
    assert np.isfinite(conc).all()
    assert np.isfinite(phase).all()


def test_cpt_concentration_range(fitted_cpt):
    _, _, _, conc, _, _ = fitted_cpt
    assert conc.min() >= 0
    assert conc.max() <= 1.0 + 1e-6


def test_cpt_phase_range(fitted_cpt):
    _, _, _, _, phase, _ = fitted_cpt
    assert phase.min() >= -1e-6
    assert phase.max() < 1.0 + 1e-6


def test_cpt_shape(fitted_cpt):
    X, mr, _, conc, phase, counts = fitted_cpt
    F = conc.shape[1]
    assert conc.shape == (X.shape[0], F)
    assert phase.shape == (X.shape[0], F)
    assert counts.shape == (X.shape[0], F)


def test_cpt_zero_count_implies_zero_conc(fitted_cpt):
    _, _, _, conc, _, counts = fitted_cpt
    zero = counts == 0
    if zero.any():
        assert (conc[zero] == 0).all()


def test_cpt_single_count_implies_conc_one(fitted_cpt):
    _, _, _, conc, _, counts = fitted_cpt
    one = counts == 1
    if one.any():
        assert np.allclose(conc[one], 1.0, atol=1e-6)


def test_activation_consistency_with_ktm(fitted_cpt):
    """CPT activation counts must match KTM counts exactly."""
    X, mr, periods, conc, phase, ct_cpt = fitted_cpt
    ppv, mu, W, ct_ktm = ktm_transform(X, mr, return_counts=True)
    assert np.array_equal(ct_cpt, ct_ktm), \
        f"count mismatch: {(ct_cpt != ct_ktm).sum()}"


# ---------------------------------------------------------------------------
# 4. Top-K selection test
# ---------------------------------------------------------------------------

def test_top_k_selection_uses_training_data():
    rng = np.random.default_rng(42)
    X = _make_znorm(n=20, T=140, seed=0)
    y = rng.integers(0, 5, size=20)
    mr = MiniRocket(random_state=42)
    Z = mr.fit_transform(X[:, None, :])

    ranks, scores = rank_kernels_by_discriminative_power(Z, y)
    mask, feat_mask = select_top_k_transformers(mr, ranks, K=10)

    assert mask.sum() == 10  # exactly K kernels selected
    assert feat_mask.sum() > 0  # features corresponding to those kernels


def test_top_k_deterministic():
    """Same data -> same selection."""
    rng = np.random.default_rng(42)
    X = _make_znorm(n=20, T=140, seed=0)
    y = rng.integers(0, 5, size=20)
    mr = MiniRocket(random_state=42)
    Z = mr.fit_transform(X[:, None, :])

    ranks, scores = rank_kernels_by_discriminative_power(Z, y)
    m1, f1 = select_top_k_transformers(mr, ranks, K=10)
    m2, f2 = select_top_k_transformers(mr, ranks, K=10)
    assert np.array_equal(m1, m2)
    assert np.array_equal(f1, f2)


def test_period_summary():
    periods = np.array([50.0, 52.0, 48.0])
    methods = np.array(['ACF', 'ACF', 'FFT'])
    s = period_summary(periods, methods)
    assert abs(s['mean'] - 50.0) < 1e-10
    assert s['acf_frac'] == pytest.approx(2 / 3)
    assert s['fft_frac'] == pytest.approx(1 / 3)
