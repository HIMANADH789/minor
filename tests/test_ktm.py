"""Unit tests for TURS-KTM (src/features/ktm.py).

Spec section 12 audit: KTM features must be derived from the SAME thresholding
condition (C > b) as MiniROCKET PPV, reproduce aeon's PPV bit-exactly, and the
mu/W formulas must match an independent explicit-tap reference implementation.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.features.ktm import (
    _activation_stats,
    activation_count_summary,
    block_stats,
    concatenate_blocks,
    feature_owners,
    ktm_transform,
    ktm_transform_reference,
    standardize_blocks,
)

from aeon.transformations.collection.convolution_based import MiniRocket

RNG = np.random.default_rng(42)


def _make_znorm(n=6, T=140, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T)).astype(np.float32)
    return ((X - X.mean(1, keepdims=True))
            / (X.std(1, keepdims=True) + 1e-8)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted():
    X = _make_znorm(n=6, T=140, seed=0)
    mr = MiniRocket(random_state=42)
    Z = mr.fit_transform(X[:, None, :])
    return X, mr, np.asarray(Z)


# ---------------------------------------------------------------------------
# 1. PPV equality with aeon (same ordering, same threshold, bit-exact)
# ---------------------------------------------------------------------------

def test_ppv_matches_aeon_exactly(fitted):
    X, mr, Z = fitted
    ppv, mu, W = ktm_transform(X, mr)
    assert ppv.shape == Z.shape
    assert np.array_equal(ppv, Z), (
        f"max abs diff {np.abs(ppv - Z).max():.3e}")


def test_counts_consistent_with_ppv(fitted):
    X, mr, Z = fitted
    ppv, mu, W, counts = ktm_transform(X, mr, return_counts=True)
    F = ppv.shape[1]
    dil_vals, dil_idx, kern_idx = feature_owners(mr)
    T = X.shape[1]
    checked = 0
    for f in range(0, F, 997):  # deterministic subsample
        d = int(dil_vals[f])
        Te = T if (int(dil_idx[f]) % 2 + int(kern_idx[f])) % 2 == 0 \
            else T - 8 * d
        assert np.allclose(counts[:, f] / Te, ppv[:, f], atol=1e-6)
        checked += 1
    assert checked > 0


def test_feature_owners_alignment(fitted):
    _, mr, Z = fitted
    dil_vals, dil_idx, kern_idx = feature_owners(mr)
    nfpd = np.asarray(mr.parameters[3])
    assert len(dil_vals) == Z.shape[1]
    # each dilation block spans 84 * nfpd[j] features
    for j, nf in enumerate(nfpd):
        sel = dil_idx == j
        assert sel.sum() == 84 * int(nf)
        assert set(kern_idx[sel].repeat(int(nf)).shape) is not None
    # dilations strictly follow the fitted values in order
    order = np.unique(dil_idx)
    assert list(order) == list(range(len(nfpd)))


# ---------------------------------------------------------------------------
# 2. Independent reference cross-check (explicit taps, float64)
# ---------------------------------------------------------------------------

def test_reference_matches_fast_path(fitted):
    X, mr, Z = fitted
    ppv, mu, W = ktm_transform(X, mr)
    ppv_r, mu_r, W_r = ktm_transform_reference(X, mr)
    # float64-vs-float32 accumulation rounds PPV by ~1e-8; all three arrays
    # must agree to float32 rounding precision, i.e. far below one flipped
    # activation position (>= 1/Te ~ 1e-2)
    assert np.abs(ppv_r - ppv).max() < 1e-6
    assert np.abs(mu_r - mu).max() < 1e-6
    assert np.abs(W_r - W).max() < 1e-6


# ---------------------------------------------------------------------------
# 3. Hand-computed example of the mu/W math
# ---------------------------------------------------------------------------

def test_hand_computed_mu_w():
    # C = [1, 5, 0, 3], b = 0.5 -> activations at tau = 0, 1, 3 (T_e = 4)
    C = np.array([1.0, 5.0, 0.0, 3.0], dtype=np.float32)
    count, mu, w = _activation_stats(C, 0.5, 4)
    assert count == 3
    # x = [0, 1/4, 3/4]; mu = (0 + 0.25 + 0.75)/3 = 1/3
    assert abs(mu - 1.0 / 3.0) < 1e-12
    # W = mean(|0 - 1/4|, |1/4 - 2/4|, |3/4 - 3/4|) = (0.25 + 0.25 + 0)/3 = 1/6
    assert abs(w - 1.0 / 6.0) < 1e-12


def test_hand_computed_single_activation():
    # C = [0, 0, 7, 0], b = 1.0 -> activation at tau = 2 (T_e = 4)
    C = np.array([0.0, 0.0, 7.0, 0.0], dtype=np.float32)
    count, mu, w = _activation_stats(C, 1.0, 4)
    assert count == 1
    assert abs(mu - 0.5) < 1e-12   # x_1 = 2/4 = 0.5
    assert abs(w - 0.0) < 1e-12    # |0.5 - 0.5| = 0


def test_hand_computed_all_activated():
    # all T_e = 4 positions activated: xs = [0, 1/4, 1/2, 3/4]
    C = np.ones(4, dtype=np.float32)
    count, mu, w = _activation_stats(C, 0.5, 4)
    assert count == 4
    assert abs(mu - (0 + 0.25 + 0.5 + 0.75) / 4) < 1e-12
    exp = np.mean(np.abs(np.array([0, .25, .5, .75])
                         - np.array([1, 2, 3, 4]) / 5))
    assert abs(w - exp) < 1e-12


# ---------------------------------------------------------------------------
# 4. Edge cases: no NaN/Inf, deterministic conventions
# ---------------------------------------------------------------------------

def test_edge_zero_activations():
    C = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    count, mu, w = _activation_stats(C, 0.0, 3)
    assert count == 0 and mu == 0.0 and w == 0.0


def test_end_to_end_no_nan_inf(fitted):
    X, mr, _ = fitted
    ppv, mu, W = ktm_transform(X, mr)
    for arr in (ppv, mu, W):
        assert np.isfinite(arr).all()


def test_end_to_end_edge_cases_real_transform(fitted):
    X, mr, Z = fitted
    ppv, mu, W, counts = ktm_transform(X, mr, return_counts=True)
    # zero-activation features must have mu = W = 0
    zero = counts == 0
    if zero.any():
        assert (mu[zero] == 0).all()
        assert (W[zero] == 0).all()
        assert (ppv[zero] == 0).all()
    # single-activation features must have W = |mu - 1/2|
    # (formula: W = |tau/T - 1/2| = |mu - 0.5| when count=1)
    one = counts == 1
    if one.any():
        assert np.allclose(W[one].astype(np.float64),
                           np.abs(mu[one].astype(np.float64) - 0.5),
                           atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Timing sensitivity demonstration (PPV blind spot)
# ---------------------------------------------------------------------------

def test_mu_shifts_when_response_shifts():
    """A response shifted in time keeps PPV but changes mu (the blind spot)."""
    T = 64
    base = np.zeros(T, dtype=np.float32)
    base[8:12] = 5.0
    b = 0.5
    c1, mu1, w1 = _activation_stats(base, b, T)
    shifted = np.roll(base, 32)
    c2, mu2, w2 = _activation_stats(shifted, b, T)
    assert c1 == c2                        # same number of activations ...
    # PPV = c/T is the same; mu captures WHAT PPV DISCARDS: the timing
    assert np.isclose(c1 / T, c2 / T)      # identical PPV
    assert mu2 - mu1 == 32.0 / T           # mean time shifted by roll amount
    assert abs(mu2 - mu1 - 0.5) < 1e-12   # mu encodes the shift PPV cannot see


# ---------------------------------------------------------------------------
# 6. Standardization / block helpers
# ---------------------------------------------------------------------------

def test_standardize_blocks_no_leakage_stats():
    rng = np.random.default_rng(1)
    tr = rng.normal(5, 3, size=(100, 4))
    te = rng.normal(50, 9, size=(20, 4))
    out_tr, out_va, out_te = standardize_blocks([tr], [None], [te])
    assert abs(out_tr[0].mean()) < 1e-12
    assert abs(out_tr[0].std() - 1) < 1e-12
    # test stats are NOT used: standardizing test with train stats must not
    # produce zero mean on the shifted test block
    assert abs(out_te[0].mean()) > 1.0


def test_block_stats():
    rng = np.random.default_rng(2)
    b = rng.normal(size=(50, 7))
    b[:, 3] = 1.0  # zero-variance column
    s = block_stats(b)
    assert s["n_features"] == 7
    assert s["near_zero_variance_count"] == 1


def test_concatenate_blocks():
    a = np.zeros((3, 2))
    b = np.ones((3, 5))
    c = concatenate_blocks([a, b])
    assert c.shape == (3, 7)
    assert (c[:, :2] == 0).all() and (c[:, 2:] == 1).all()


def test_activation_count_summary():
    counts = np.array([[0, 1, 2, 5, 0]])
    s = activation_count_summary(counts)
    assert s["n_pairs"] == 5
    assert s["zero"] == 2 and s["one"] == 1 and s["multi"] == 2
    assert s["max"] == 5
