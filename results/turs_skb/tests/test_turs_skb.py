"""TURS-SKB: pre-registered unit tests (spec sections 36 and 46 STEP 4).

All tests MUST pass before the four-dataset benchmark runs.

Covers (spec section 36):
  1.  Deterministic kernel generation
  2.  Exact kernel normalization
  3.  Correct convolution response
  4.  Correct dilation
  5.  PPV correctness
  6.  Derivative kernel correctness
  7.  Wavelet formula correctness
  8.  Gabor formula correctness
  9.  Energy operator correctness
  10. Feature dimension correctness
  11. Zero/edge-case handling
  12. Block standardization
  13. No NaNs/infs
  14. No test leakage (quantile biases depend only on the rows passed to
      fit; transform uses only fit-time state)
  15. Identical MiniROCKET base between all variants (deterministic
      transformer equality under the fixed seed)
  16. Fixed-seed reproducibility (golden quantiles + full pipeline)
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.features.turs_skb.kernels import (
    ALPHA_DO_G_GRID,
    BANK_BUILDERS,
    MAX_SUPPORT,
    difference_of_gaussians_kernel,
    derivative_kernel,
    energy_kernel,
    gabor_kernel,
    gaussian_kernel,
    gauss_deriv_kernel,
    haar_kernel,
    mexican_hat_kernel,
    morlet_kernel,
    tent_window,
    wavelet_kernel,
)
from src.features.turs_skb.features import (
    BlockStandardizer,
    SKBExtractor,
    assemble_blocks,
    energy_kernel_response,
    golden_quantiles,
    linear_cka,
    linear_kernel_response,
    mean_abs_cross_correlation,
)

N_BIASES = 9


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _all_kernels():
    for fam, builder in BANK_BUILDERS.items():
        for kern, meta in builder():
            yield fam, kern, meta


@pytest.fixture(scope="module")
def rng_data():
    return np.random.default_rng(42).normal(size=(40, 200)).astype(np.float32)


# ----------------------------------------------------------------------------
# 1. Deterministic kernel generation
# ----------------------------------------------------------------------------
def test_01_deterministic_kernel_generation():
    """Building a bank twice yields bit-identical kernel vectors."""
    for name, builder in BANK_BUILDERS.items():
        b1 = builder()
        b2 = builder()
        assert len(b1) == len(b2) > 0, name
        for (k1, m1), (k2, m2) in zip(b1, b2):
            assert np.array_equal(k1, k2), name
            assert m1 == m2, name


def test_01b_no_randomness_in_module():
    """The kernels module must not consume global RNG state."""
    from src.features.turs_skb import kernels as K
    np.random.seed(1234)
    a = K.build_gabor_bank()
    np.random.seed(999)
    b = K.build_gabor_bank()
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# 2. Exact kernel normalization
# ----------------------------------------------------------------------------
def test_02_exact_kernel_normalization():
    """Every linear kernel is unit-L2; energy windows are nonneg sum-1."""
    for fam, kern, meta in _all_kernels():
        if meta["kind"] == "energy":
            assert kern.min() >= 0.0, fam
            assert abs(kern.sum() - 1.0) < 1e-12, fam
        else:
            assert abs(np.sqrt((kern ** 2).sum()) - 1.0) < 1e-12, fam
    for s in (1.0, 2.0, 8.0, 16.0):
        for sk in (-1.0, 0.0, 1.0):
            k = gaussian_kernel(s, sk)
            assert abs(np.sqrt((k ** 2).sum()) - 1.0) < 1e-12
    for al in ALPHA_DO_G_GRID:
        k = difference_of_gaussians_kernel(4.0, al)
        assert abs(np.sqrt((k ** 2).sum()) - 1.0) < 1e-12


def test_02b_support_cap():
    for fam, kern, meta in _all_kernels():
        if meta["kind"] == "energy":
            assert meta["w_long"] <= 128, meta
        else:
            assert kern.size <= MAX_SUPPORT, (fam, meta)


# ----------------------------------------------------------------------------
# 3. Correct convolution response
# ----------------------------------------------------------------------------
def test_03_convolution_response_matches_direct():
    """Response equals the direct dilated formula r(t)=sum_j k_j x_{t+jd}."""
    X = np.random.default_rng(7).normal(size=(3, 128)).astype(np.float32)
    k = gaussian_kernel(2.0)  # support 17
    for d in (1, 2, 4):  # d=8 would span 129 > 128 -> correctly None
        r = linear_kernel_response(X, k, d)
        L = k.size
        n_pos = 128 - (L - 1) * d
        assert r.shape == (3, n_pos)
        for i in range(3):
            for t in range(n_pos):
                direct = sum(k[j] * X[i, t + j * d] for j in range(L))
                assert abs(r[i, t] - direct) < 1e-4


def test_03b_kernel_too_long_returns_none():
    X = np.random.default_rng(7).normal(size=(2, 16)).astype(np.float32)
    k = gaussian_kernel(16.0)  # support 128 > 16
    assert linear_kernel_response(X, k, 1) is None


# ----------------------------------------------------------------------------
# 4. Correct dilation
# ----------------------------------------------------------------------------
def test_04_dilation_equals_strided_short_response():
    """Dilated response sampled at stride d == short-kernel response on the
    stride-d subsampled signal: r_full[ds] = sum_j k_j x[ds + jd] and
    r_sub[s]  = sum_j k_j xs[s + j] = sum_j k_j x[d(s + j)] are the SAME sum.
    """
    x = np.random.default_rng(0).normal(size=(1, 128)).astype(np.float32)
    for order in (1, 2):
        k_base = derivative_kernel(order)
        for d in (1, 2, 4, 8):
            r_full = linear_kernel_response(x, k_base, d)[0]
            r_sub = linear_kernel_response(x[:, ::d], k_base, 1)[0]
            assert np.allclose(r_full[::d], r_sub, atol=1e-4), (order, d)


def test_04b_effective_dilation_clip():
    from src.features.turs_skb.features import _max_dilation

    def ref(L, T):
        d = 1
        while d * 2 <= T and (L - 1) * (d * 2) + 1 <= T:
            d *= 2
        return d

    for L in (2, 3, 5, 9, 33, 129):
        for T in (8, 16, 140, 200, 1024):
            assert _max_dilation(L, T) == ref(L, T), (L, T)
    # spot values
    assert _max_dilation(129, 1024) == 4   # 128*4+1 = 513 <= 1024
    assert _max_dilation(129, 512) == 2    # 128*4+1 = 513 > 512
    assert _max_dilation(3, 1024) == 256   # 2*256+1 = 513 <= 1024


def test_04c_bank_dilations_capped_to_fit():
    """Fitted d_eff must satisfy (L-1)*d_eff + 1 <= T for every kernel."""
    X = np.random.default_rng(1).normal(size=(5, 140)).astype(np.float32)
    ex = SKBExtractor(families=["wavelet", "derivative"]).fit(X)
    T = 140
    for fam in ex.families:
        for kern, meta, d_eff in ex.kernels_[fam]:
            assert (kern.size - 1) * d_eff + 1 <= T, (fam, meta, d_eff)


# ----------------------------------------------------------------------------
# 5. PPV correctness
# ----------------------------------------------------------------------------
def test_05_ppv_correctness():
    """PPV equals the fraction of positions strictly above each bias."""
    X = np.random.default_rng(3).normal(size=(4, 100)).astype(np.float32)
    ex = SKBExtractor(families=["wavelet"]).fit(X)
    kern, meta, d_eff = ex.kernels_["wavelet"][0]
    biases = ex.biases_["wavelet"][0]
    resp = linear_kernel_response(X, kern, d_eff)
    F = ex.transform(X)["wavelet"]
    ppv0 = (resp[:, :, None] > biases[None, None, :]).mean(axis=1)
    assert np.allclose(F[:, :N_BIASES], ppv0, atol=1e-12)
    # biases are raw quantiles (unsorted golden sequence): the max-bias
    # level (0.944 quantile) must give PPV <= the min-bias level (0.090)
    q = golden_quantiles(N_BIASES)
    i_max, i_min = int(np.argmax(q)), int(np.argmin(q))
    assert (F[:, i_max] <= F[:, i_min] + 1e-9).all()


def test_05b_ppv_range_and_finiteness(rng_data):
    ex = SKBExtractor().fit(rng_data)
    F = ex.transform(rng_data)
    for fam, f in F.items():
        assert np.isfinite(f).all(), fam
        assert f.min() >= 0.0 and f.max() <= 1.0, fam


# ----------------------------------------------------------------------------
# 6. Derivative kernel correctness
# ----------------------------------------------------------------------------
def test_06_derivative_kernels():
    k1 = derivative_kernel(1)
    k2 = derivative_kernel(2)
    nz1 = k1[np.abs(k1) > 1e-12]
    assert np.allclose(nz1, np.array([-1.0, 1.0]) / np.sqrt(2.0))
    nz2 = k2[np.abs(k2) > 1e-12]
    assert np.allclose(np.sort(nz2),
                       np.sort(np.array([1.0, -2.0, 1.0])) / np.sqrt(6.0))
    # action on a ramp: first derivative ~ constant, second ~ 0
    x = np.linspace(-5, 5, 64).astype(np.float32)[None, :]
    r1 = linear_kernel_response(x, k1, 1)
    r2 = linear_kernel_response(x, k2, 1)
    assert np.ptp(r1) < 1e-5          # constant slope
    assert np.abs(r2).max() < 1e-5    # zero curvature
    # unknown order rejected
    with pytest.raises(ValueError):
        derivative_kernel(3)


def test_06b_gaussian_derivative_formula():
    """First Gaussian derivative is antisymmetric; second is symmetric with
    zero total mass (Laplacian-of-Gaussian)."""
    k = gauss_deriv_kernel(4.0, 1)
    assert abs(k.sum()) < 1e-9                      # antisymmetric
    assert k[np.argmax(k)] > 0 and k[np.argmin(k)] < 0
    assert np.allclose(k, k[::-1] * -1, atol=1e-12)
    k2 = gauss_deriv_kernel(4.0, 2)
    # truncated discrete LoG has small residual mass (support cut at 4 sigma)
    assert abs(k2.sum()) < 0.01
    assert np.allclose(k2, k2[::-1], atol=1e-12)    # symmetric
    with pytest.raises(ValueError):
        gauss_deriv_kernel(4.0, 3)


# ----------------------------------------------------------------------------
# 7. Wavelet formula correctness
# ----------------------------------------------------------------------------
def test_07_wavelet_formulas():
    # Haar (discrete, dilated 2-tap): taps -1/sqrt2 at 0, +1/sqrt2 at a,
    # zero padding after
    h = haar_kernel(1)
    assert h.shape == (3,)
    assert h[0] < 0 and h[1] > 0 and h[2] == 0.0
    assert abs(h[0] + h[1]) < 1e-12   # antisymmetric tap pair
    # Mexican hat: peak at center, negative tails, ~zero integral
    m = mexican_hat_kernel(2)
    c = m.size // 2
    assert m[c] == m.max()
    assert m[0] < 0 and m[-1] < 0
    assert abs(m.sum()) < 0.05
    # Morlet: oscillatory, sign changes
    mo = morlet_kernel(1)
    assert mo.max() > 0 and mo.min() < 0
    assert np.sign(mo).std() > 0
    # unknown type rejected
    with pytest.raises(ValueError):
        wavelet_kernel("coif5", 2)
    # scale a=2 has wider support than a=1 (dilation of the mother wavelet)
    assert mexican_hat_kernel(1).size < mexican_hat_kernel(2).size


# ----------------------------------------------------------------------------
# 8. Gabor formula correctness
# ----------------------------------------------------------------------------
def test_08_gabor_formula():
    freq, sigma, phase = 0.1, 4.0, 0.0
    k = gabor_kernel(freq, sigma, phase)
    u = np.arange(k.size) - (k.size - 1) / 2.0
    ref = np.exp(-(u ** 2) / (2.0 * sigma ** 2)) * np.cos(2 * np.pi * freq * u)
    ref = ref / np.sqrt((ref ** 2).sum())
    assert np.allclose(k, ref, atol=1e-10)
    # quadrature phase shifts the oscillation
    kq = gabor_kernel(freq, sigma, np.pi / 2)
    refq = np.exp(-(u ** 2) / (2.0 * sigma ** 2)) * \
        np.cos(2 * np.pi * freq * u + np.pi / 2)
    refq = refq / np.sqrt((refq ** 2).sum())
    assert np.allclose(kq, refq, atol=1e-10)
    # envelope: kernel decays to near zero at the borders
    assert abs(k[0]) < 0.05 * np.abs(k).max()


# ----------------------------------------------------------------------------
# 9. Energy operator correctness
# ----------------------------------------------------------------------------
def test_09_energy_operator():
    X = np.arange(1, 9, dtype=np.float32)[None, :]
    q_s, w, w_l, lam = energy_kernel(2, 0.5)
    q_l = tent_window(w_l)
    out = energy_kernel_response(X, q_s, q_l, 0.5)
    n_pos = X.shape[1] - w_l + 1
    e_s = np.array([float((X[0, t:t + w] ** 2 * q_s).sum())
                    for t in range(n_pos)])
    e_l = np.array([float((X[0, t:t + w_l] ** 2 * q_l).sum())
                    for t in range(n_pos)])
    assert np.allclose(out[0], e_s - 0.5 * e_l, atol=1e-5)
    # q windows: nonnegative, sum-1, symmetric tent profile
    assert q_s.min() >= 0 and abs(q_s.sum() - 1) < 1e-12
    assert q_l.min() >= 0 and abs(q_l.sum() - 1) < 1e-12
    assert np.isclose(q_s[0], q_s[-1]) and q_s.max() >= q_s[0] - 1e-12
    # lambda=0 -> pure short-window energy
    out0 = energy_kernel_response(X, q_s, q_l, 0.0)
    assert np.allclose(out0[0], e_s, atol=1e-5)
    # long window must be 4x short (deterministic relation)
    assert w_l == 4 * w


# ----------------------------------------------------------------------------
# 10. Feature dimension correctness
# ----------------------------------------------------------------------------
def test_10_feature_dimensions(rng_data):
    """Bank sizes are exact and T-independent; fitted dimensions equal
    (surviving unique kernels) x 9 biases, with duplicates collapsed."""
    bank_sizes = {f: len(b()) for f, b in BANK_BUILDERS.items()}
    assert bank_sizes == {"morphological": 140, "derivative": 32,
                          "wavelet": 192, "gabor": 192, "energy": 12}
    ex = SKBExtractor().fit(rng_data)
    dims = ex.family_dims()
    for fam in ex.families:
        n_k = len(ex.kernels_[fam])
        assert 0 < n_k <= bank_sizes[fam], fam
        assert dims[fam] == n_k * N_BIASES, fam
        assert ex.duplicates_[fam] == bank_sizes[fam] - n_k, fam
        # no duplicated identities survive (energy identity includes lambda,
        # since identical q_short with different lambda gives different B(t))
        if fam == "energy":
            keys = [(b"E", m["w_short"], m["lambda"])
                    for _, m, _ in ex.kernels_[fam]]
        else:
            keys = [(kern.tobytes(), d) for kern, _, d in ex.kernels_[fam]]
        assert len(keys) == len(set(keys)), fam
    # energy family is never collapsed (each (w, lambda) is distinct)
    assert len(ex.kernels_["energy"]) == 12 and ex.duplicates_["energy"] == 0
    F = ex.transform(rng_data)
    for fam, f in F.items():
        assert f.shape == (rng_data.shape[0], dims[fam]), fam
    meta = ex.metadata()
    assert len(meta) == sum(len(ex.kernels_[f]) for f in ex.families)


# ----------------------------------------------------------------------------
# 11. Zero / edge-case handling
# ----------------------------------------------------------------------------
def test_11_zero_and_edge_cases():
    # all-zero signal: no NaN/inf, valid PPVs
    Xz = np.zeros((3, 60), dtype=np.float32)
    ex = SKBExtractor().fit(Xz)
    F = ex.transform(Xz)
    for f in F.values():
        assert np.isfinite(f).all()
        assert f.max() <= 1.0
    # constant signal: no NaN/inf
    Xc = np.full((3, 60), 3.0, dtype=np.float32)
    ex2 = SKBExtractor().fit(Xc)
    F2 = ex2.transform(Xc)
    for f in F2.values():
        assert np.isfinite(f).all()
    # very short signal: kernels that cannot fit are skipped, no crash
    Xs = np.random.default_rng(5).normal(size=(3, 8)).astype(np.float32)
    ex3 = SKBExtractor().fit(Xs)
    assert sum(ex3.family_dims().values()) > 0
    F3 = ex3.transform(Xs)
    assert all(np.isfinite(f).all() for f in F3.values())
    # energy family may be empty at tiny T (w_long=128 > 8)
    assert ex3.family_dims()["energy"] == 0


# ----------------------------------------------------------------------------
# 12. Block standardization
# ----------------------------------------------------------------------------
def test_12_block_standardization(rng_data):
    A = rng_data[:, :10]
    B = rng_data[:, 10:22] * 50 + 3
    X = np.concatenate([A, B], axis=1)
    sc = BlockStandardizer([10, 12]).fit(X)
    Z = sc.transform(X)
    assert abs(Z[:, :10].mean()) < 1e-6   # float32 accumulation residual
    assert abs(Z[:, :10].std() - 1) < 1e-6
    assert abs(Z[:, 10:].mean()) < 1e-6
    assert abs(Z[:, 10:].std() - 1) < 1e-6
    # blocks are standardized independently (no cross-block scale leakage)
    X2 = np.concatenate([A, B * 1000], axis=1)
    Z2 = BlockStandardizer([10, 12]).fit(X2).transform(X2)
    assert np.allclose(Z2[:, :10], Z[:, :10], atol=1e-9)
    # empty blocks (0-dim families) pass through
    sc3 = BlockStandardizer([10, 0, 12]).fit(X)
    Z3 = sc3.transform(X)
    assert Z3.shape == X.shape


# ----------------------------------------------------------------------------
# 13. No NaNs / infs (full assembled pipeline)
# ----------------------------------------------------------------------------
def test_13_no_nan_inf_assembled(rng_data):
    ex = SKBExtractor().fit(rng_data)
    blocks = list(ex.transform(rng_data).values())
    X, dims = assemble_blocks(blocks)
    Z = BlockStandardizer(dims).fit(X).transform(X)
    assert np.isfinite(Z).all()
    assert not np.isnan(Z).any() and not np.isinf(Z).any()
    # near-zero-variance columns do not blow up
    assert np.abs(Z).max() < 1e6


# ----------------------------------------------------------------------------
# 14. No test leakage
# ----------------------------------------------------------------------------
def test_14_no_test_leakage_in_fit():
    """Biases come only from the rows given to fit: a transformer fitted on
    train rows assigns different quantile thresholds than one fitted on
    other rows; features of a row never depend on that row's position."""
    rng = np.random.default_rng(11)
    A = rng.normal(size=(50, 100)).astype(np.float32)
    B = rng.normal(size=(10, 100)).astype(np.float32) + 5.0
    exA = SKBExtractor(families=["morphological"]).fit(A)
    exB = SKBExtractor(families=["morphological"]).fit(B)
    bA = np.concatenate(exA.biases_["morphological"])
    bB = np.concatenate(exB.biases_["morphological"])
    assert not np.allclose(bA, bB)      # thresholds track the fit rows only
    # row-order invariance of transform: permuting transform rows permutes
    # features identically (no position-dependent adaptation)
    FA = exA.transform(A)["morphological"]
    perm = np.random.default_rng(0).permutation(len(A))
    assert np.allclose(exA.transform(A[perm])["morphological"],
                       FA[perm], atol=1e-12)


def test_14b_transform_uses_only_fit_state(rng_data):
    """transform() must not recompute quantiles from transform-time data:
    perturbing fit-time biases changes the output."""
    ex = SKBExtractor(families=["gabor"]).fit(rng_data)
    F_a = ex.transform(rng_data)["gabor"]
    ex.biases_["gabor"] = [b + 1e-3 for b in ex.biases_["gabor"]]
    F_b = ex.transform(rng_data)["gabor"]
    assert not np.allclose(F_a, F_b)


# ----------------------------------------------------------------------------
# 15. Identical MiniROCKET base between variants
# ----------------------------------------------------------------------------
def test_15_identical_minirocket_base():
    """The canonical MiniROCKET transformer is deterministic given the seed:
    two fits on identical data give bit-identical features, so every SKB
    variant that reuses the same fitted transformer shares the same base."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    X = np.random.default_rng(9).normal(size=(12, 1, 100)).astype(np.float32)
    m1 = MiniRocket(n_kernels=84, random_state=42)
    m2 = MiniRocket(n_kernels=84, random_state=42)
    m1.fit(X)
    m2.fit(X)
    Z1 = m1.transform(X)
    Z2 = m2.transform(X)
    assert Z1.shape == Z2.shape
    assert np.array_equal(Z1, Z2)


# ----------------------------------------------------------------------------
# 16. Fixed-seed reproducibility
# ----------------------------------------------------------------------------
def test_16_fixed_seed_reproducibility(rng_data):
    ex1 = SKBExtractor().fit(rng_data)
    F1 = ex1.transform(rng_data)
    ex2 = SKBExtractor().fit(rng_data)
    F2 = ex2.transform(rng_data)
    for fam in F1:
        assert np.array_equal(F1[fam], F2[fam]), fam
    # golden quantiles are the exact aeon sequence
    from aeon.transformations.collection.convolution_based._minirocket import \
        _quantiles
    assert np.array_equal(golden_quantiles(N_BIASES), _quantiles(N_BIASES))


# ----------------------------------------------------------------------------
# Extra: representation utilities
# ----------------------------------------------------------------------------
def test_cka_properties():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(60, 30))
    assert abs(linear_cka(A, A) - 1.0) < 1e-9
    B = rng.normal(size=(60, 30))
    v = linear_cka(A, B)
    assert 0.0 <= v <= 1.0
    C = B + 0.01 * rng.normal(size=(60, 30))
    assert linear_cka(A, C) < 0.5


def test_mean_abs_cross_correlation():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(200, 20))
    B = A + 0.1 * rng.normal(size=(200, 20))
    v_same = mean_abs_cross_correlation(A, B, max_feat=20)
    C = rng.normal(size=(200, 20))
    v_diff = mean_abs_cross_correlation(A, C, max_feat=20)
    assert v_same > v_diff
    assert 0.0 <= v_diff <= 1.0
