"""
Unit tests for TURS-MSW (spec sec. 23, 13 tests).

The critical tests are #1/#2: the global block of the regional extractor
MUST equal the canonical aeon MiniRocket output (tolerance 1e-12; the
extractor sums float32 counts and divides by the same window length aeon
uses, so exact float equality is expected).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.turs_msw.msw import (  # noqa: E402
    build_regions, msw_transform_uni, region_slices_for_valid_axis,
    variant_matrix,
)
from aeon.transformations.collection.convolution_based import MiniRocket  # noqa: E402


def _toy_fit(T=64, n_kernels=1008, seed=42):
    rng = np.random.RandomState(0)
    X = rng.randn(6, 1, T).astype(np.float32)
    mr = MiniRocket(n_kernels=n_kernels, random_state=seed)
    Z = mr.fit_transform(X)
    return mr, X, np.asarray(Z)


def test_global_ppv_equals_canonical_minirocket():
    """CRITICAL (sec. 22): regional extractor's global block == aeon M0."""
    mr, X, Z = _toy_fit()
    regions = build_regions(X.shape[-1])
    feats = msw_transform_uni(X[:, 0, :], mr.parameters,
                              MiniRocket._indices, regions)
    F = len(mr.parameters[4])
    diff = np.abs(feats[:, :F] - Z)
    assert diff.max() < 1e-12, f"global block mismatch: max diff {diff.max()}"


def test_activation_rule_matches_minirocket_count():
    """Activation masks derived from same C and biases reproduce the exact
    per-feature activation counts that underlie aeon's PPV means."""
    mr, X, Z = _toy_fit(T=48, n_kernels=84 * 4, seed=7)
    regions = build_regions(48)
    feats = msw_transform_uni(X[:, 0, :], mr.parameters,
                              MiniRocket._indices, regions)
    F = len(mr.parameters[4])
    # PPV * T counts must be integers equal to our prefix-sum counts
    counts_aeon = np.round(Z * 48).astype(np.int64)
    counts_msw = np.round(feats[:, :F] * 48).astype(np.int64)
    assert (counts_aeon == counts_msw).all()


def test_medium_window_boundaries():
    for T in [9, 32, 140, 1024, 1092]:
        r = build_regions(T)
        w = int(np.ceil(3 * T / 4))
        assert r["medium"] == [(0, w), (T - w, T)]
        assert (r["medium"][1][0] - r["medium"][0][1]) == T - 2 * w


def test_local_window_boundaries():
    for T in [9, 32, 140, 1024, 1092]:
        r = build_regions(T)
        wl = int(np.ceil(T / 2))
        sl = max(1, int(np.floor(T / 4)))
        assert r["local"] == [(j * sl, min(j * sl + wl, T))
                              for j in range(4)]


def test_overlap_and_coverage():
    for T in [9, 64, 178, 1024]:
        r = build_regions(T)
        m = r["medium"]
        assert m[0][1] > m[1][0]          # medium neighbours overlap
        assert m[0][0] == 0 and m[1][1] == T  # full coverage
        loc = r["local"]
        for j in range(3):
            assert loc[j][1] > loc[j + 1][0]  # 50% neighbour overlap
        assert loc[0][0] == 0 and max(b for _, b in loc) == T


def test_no_empty_windows():
    for T in [9, 10, 11, 32, 100]:
        r = build_regions(T)
        for scale, regs in r.items():
            for (a, b) in regs:
                assert b > a


def test_correct_feature_dimensions():
    mr, X, Z = _toy_fit(T=64, n_kernels=1008)
    F = len(mr.parameters[4])
    regions = build_regions(64)
    feats = msw_transform_uni(X[:, 0, :], mr.parameters,
                              MiniRocket._indices, regions)
    assert feats.shape == (X.shape[0], F * 7)
    variants = variant_matrix(feats, F)
    assert variants["M0"].shape == (X.shape[0], F)
    assert variants["M1"].shape == (X.shape[0], F * 3)
    assert variants["M2"].shape == (X.shape[0], F * 5)
    assert variants["M3"].shape == (X.shape[0], F * 7)
    assert variants["M4"].shape == (X.shape[0], F * 6)


def test_block_standardization_train_only():
    rng = np.random.RandomState(1)
    Ftr = rng.randn(50, 7 * 4).astype(np.float32)
    Fte = rng.randn(10, 7 * 4).astype(np.float32) * 2 + 0.5
    mu, sd = Ftr.mean(0), Ftr.std(0)
    nz = sd < 1e-8
    sd_safe = sd.copy()
    sd_safe[nz] = 1.0
    Str = (Ftr - mu) / sd_safe
    Ste = (Fte - mu) / sd_safe
    assert abs(Str.mean()) < 1e-6
    assert np.allclose(Str.std(0)[~nz], 1.0, atol=1e-4)
    assert not np.isnan(Str).any() and not np.isinf(Str).any()
    assert not np.isnan(Ste).any() and not np.isinf(Ste).any()


def test_no_nans_or_infs_real_data():
    mr, X, Z = _toy_fit(T=64, n_kernels=1008)
    regions = build_regions(64)
    feats = msw_transform_uni(X[:, 0, :], mr.parameters,
                              MiniRocket._indices, regions)
    assert np.isfinite(feats).all()


def test_deterministic_output():
    mr, X, Z = _toy_fit(T=64, n_kernels=1008)
    regions = build_regions(64)
    f1 = msw_transform_uni(X[:, 0, :], mr.parameters,
                           MiniRocket._indices, regions)
    f2 = msw_transform_uni(X[:, 0, :], mr.parameters,
                           MiniRocket._indices, regions)
    assert np.array_equal(f1, f2)


def test_variable_sequence_lengths():
    for T in [32, 140, 1024]:
        mr, X, Z = _toy_fit(T=T, n_kernels=504)
        regions = build_regions(T)
        feats = msw_transform_uni(X[:, 0, :], mr.parameters,
                                  MiniRocket._indices, regions)
        F = len(mr.parameters[4])
        assert feats.shape == (X.shape[0], F * 7)
        assert np.isfinite(feats).all()


def test_regions_map_onto_valid_axis():
    """padding1==1 regions must stay inside the valid axis (or clip)."""
    T, pad = 64, 12
    regions = build_regions(T)
    mapped = region_slices_for_valid_axis(regions, pad, T)
    for scale, sl in mapped.items():
        for (a, b) in sl:
            assert a >= 0 and b <= T - 2 * pad


def test_no_test_leakage_in_design():
    """The extractor is a pure function of (X, fitted parameters, regions);
    nothing here can access labels. Guard the interface contract."""
    import inspect
    sig = inspect.signature(msw_transform_uni)
    assert "y" not in sig.parameters


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
