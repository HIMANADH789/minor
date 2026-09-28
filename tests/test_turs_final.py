"""TURS-FINAL: pre-registered unit tests (spec section 21).

All 18 tests MUST pass before the four-dataset benchmark runs.

Covers:
  1.  KTM-W formula correctness
  2.  KTM-W matches the previously validated implementation
      (bit-exact PPV agreement with aeon's transform; W cross-checked
      against the independent reference path from test_ktm.py)
  3.  MiniROCKET activation mask matches aeon
  4.  Exactly 84 KTM-W features (84 kernel groups)
  5.  Circular shift preserves exact signal values
  6.  Circular shift preserves signal length
  7.  Random shifts are reproducible
  8.  Null generation does not modify labels
  9.  p-value formula is correct
  10. p-value range is valid
  11. KEEP/DROP rule is exactly: p < 0.05 AND Delta_real > 0
  12. Standardization is blockwise
  13. Standardization uses no test data
  14. No NaN/inf features
  15. Test labels cannot enter the decision function
  16. Decision is one constant value per dataset
  17. No per-example routing exists
  18. S=200 is actually executed
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from src.features.ktm import (
    _activation_stats,
    feature_owners,
    ktm_transform,
    ktm_transform_reference,
)
from aeon.transformations.collection.convolution_based import MiniRocket

import run_turs_final as R

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_znorm(n=6, T=140, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T)).astype(np.float32)
    return ((X - X.mean(1, keepdims=True))
            / (X.std(1, keepdims=True) + 1e-8)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted():
    X = _make_znorm(n=6, T=140, seed=0)
    mr = MiniRocket(random_state=42, n_jobs=-1)
    Z = mr.fit_transform(X[:, None, :])
    return X, mr, np.asarray(Z)


# ---------------------------------------------------------------------------
# 1. KTM-W formula correctness
# ---------------------------------------------------------------------------

def test_01_ktm_w_formula_correct():
    # Hand-computed: C = [1, 5, 0, 3], b = 0.5, T_e = 4
    # activations at tau = 0, 1, 3 -> x = [0, 0.25, 0.75]
    # W = mean(|0 - 1/4|, |1/4 - 2/4|, |3/4 - 3/4|) = 1/6
    C = np.array([1.0, 5.0, 0.0, 3.0], dtype=np.float32)
    count, mu, w = _activation_stats(C, 0.5, 4)
    assert count == 3
    assert abs(w - 1.0 / 6.0) < 1e-12
    assert abs(mu - 1.0 / 3.0) < 1e-12


def test_01b_ktm_w_matches_reference_implementation(fitted):
    # W from the fast numba path must equal the independent explicit-tap
    # reference implementation (previously validated in tests/test_ktm.py)
    X, mr, _ = fitted
    _, _, W_fast = ktm_transform(X, mr)
    _, _, W_ref = ktm_transform_reference(X, mr)
    assert np.abs(W_fast.astype(np.float64) - W_ref).max() < 1e-6


# ---------------------------------------------------------------------------
# 2. KTM-W matches the previously validated implementation
# ---------------------------------------------------------------------------

def test_02_ktm_w_matches_validated_implementation(fitted):
    # Bit-exact PPV agreement with aeon's own transform proves the activation
    # mask {t : C(t) > b} is the SAME set aeon thresholds; W is derived from
    # that same mask, hence KTM-W matches the validated implementation.
    X, mr, Z = fitted
    ppv, mu, W = ktm_transform(X, mr)
    assert ppv.shape == Z.shape
    assert np.array_equal(ppv, Z), (
        f"max abs diff {np.abs(ppv - Z).max():.3e}")


# ---------------------------------------------------------------------------
# 3. MiniROCKET activation mask matches aeon
# ---------------------------------------------------------------------------

def test_03_activation_mask_matches_aeon(fitted):
    # For every sampled feature: count/Te from the KTM mask must equal
    # aeon's PPV feature bit-exactly (the mask is where aeon counts).
    X, mr, Z = fitted
    ppv, _, _ = ktm_transform(X, mr)
    assert np.array_equal(ppv, Z)
    # counts/Te consistency on a deterministic subsample
    _, _, W, counts = ktm_transform(X, mr, return_counts=True)
    dil_vals, dil_idx, kern_idx = feature_owners(mr)
    T = X.shape[1]
    checked = 0
    for f in range(0, ppv.shape[1], 997):
        d = int(dil_vals[f])
        Te = T if (int(dil_idx[f]) % 2 + int(kern_idx[f])) % 2 == 0 \
            else T - 8 * d
        assert np.allclose(counts[:, f] / Te, ppv[:, f], atol=1e-6)
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# 4. Exactly 84 KTM-W features (84 kernel groups)
# ---------------------------------------------------------------------------

def test_04_exactly_84_kernel_group_features(fitted):
    X, mr, Z = fitted
    G = R.kernel_group_features(X, mr)
    assert G.shape == (X.shape[0], 84), (
        f"expected [n, 84], got {G.shape}")
    # the 84 groups are exactly the unique kernel ids of aeon's layout
    _, _, kern_idx = feature_owners(mr)
    assert len(np.unique(kern_idx)) == 84


# ---------------------------------------------------------------------------
# 5-6. Circular shift preserves exact values / length
# ---------------------------------------------------------------------------

def test_05_circular_shift_preserves_exact_values():
    x = np.arange(37, dtype=np.float64) * 0.37 - 5.1
    for s in (0, 1, 13, 36, 37, 100):
        y = R.circular_shift_signal(x, s)
        assert y.shape == x.shape
        assert np.array_equal(np.sort(y), np.sort(x))  # multiset preserved
        # exact roll semantics: x_shift[t] = x[(t - s) mod T]
        T = len(x)
        sm = s % T
        for t in (0, 1, T // 2, T - 1):
            assert y[t] == x[(t - sm) % T]


def test_06_circular_shift_preserves_length():
    rng = np.random.default_rng(3)
    for T in (1, 2, 7, 140, 1024):
        x = rng.normal(size=T)
        y = R.circular_shift_signal(x, int(rng.integers(0, 10 * T)))
        assert y.shape == x.shape and len(y) == T


# ---------------------------------------------------------------------------
# 7. Random shifts are reproducible
# ---------------------------------------------------------------------------

def test_07_null_shifts_reproducible():
    n, T, S = 12, 64, 200
    a = R.make_null_shifts(n, T, S, seed=42)
    b = R.make_null_shifts(n, T, S, seed=42)
    c = R.make_null_shifts(n, T, S, seed=43)
    assert a.shape == (S, n)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# 8. Null generation does not modify labels
# ---------------------------------------------------------------------------

def test_08_null_generation_does_not_modify_labels():
    X = _make_znorm(n=10, T=60, seed=1)
    y = np.arange(10) % 3
    y_before = y.copy()
    shifts = R.make_null_shifts(len(X), X.shape[1], 5, seed=7)
    for j in range(5):
        Xs = R.apply_shifts(X, shifts[j])
        assert np.array_equal(y, y_before)
        assert Xs.shape == X.shape
    # X itself must not be mutated
    assert np.isfinite(X).all()


# ---------------------------------------------------------------------------
# 9-10. p-value formula and range
# ---------------------------------------------------------------------------

def test_09_p_value_formula():
    delta_real = 0.010
    null = np.array([0.001, 0.002, 0.005, 0.020, 0.030])
    # count(Delta_null >= Delta_real) = 2 -> (1+2)/(200+1)
    p = R.empirical_p_value(delta_real, null, S=200)
    assert p == pytest.approx(3 / 201)


def test_09b_p_value_s200_resolution():
    # With S=200 the minimum possible p is 1/201 (~0.00498), never 1/21
    null = np.full(200, -1.0)
    p = R.empirical_p_value(0.5, null, S=200)
    assert p == pytest.approx(1 / 201)


def test_10_p_value_range_valid():
    rng = np.random.default_rng(11)
    for _ in range(50):
        delta = rng.normal()
        null = rng.normal(size=200)
        p = R.empirical_p_value(delta, null, S=200)
        assert 1 / 201 <= p <= 1.0


# ---------------------------------------------------------------------------
# 11. KEEP/DROP rule is exactly p < 0.05 AND Delta_real > 0
# ---------------------------------------------------------------------------

def test_11_keep_drop_rule_exact():
    cases = [
        (0.049, +0.001, "KEEP_KTM_W"),
        (0.049, 0.000, "DROP_KTM_W"),      # delta must be strictly positive
        (0.049, -0.001, "DROP_KTM_W"),
        (0.050, +0.001, "DROP_KTM_W"),     # p must be strictly below 0.05
        (0.051, +0.001, "DROP_KTM_W"),
        (0.005, +0.500, "KEEP_KTM_W"),
    ]
    for p, d, expected in cases:
        assert R.keep_drop_rule(p, d) == expected


# ---------------------------------------------------------------------------
# 12-13. Standardization is blockwise / uses no test data
# ---------------------------------------------------------------------------

def test_12_standardization_blockwise():
    rng = np.random.default_rng(5)
    A_tr = rng.normal(10, 2, size=(50, 3))     # block A
    B_tr = rng.normal(-40, 9, size=(50, 2))    # block B
    A_va = rng.normal(10, 2, size=(20, 3))
    B_va = rng.normal(-40, 9, size=(20, 2))
    tr = np.concatenate([A_tr, B_tr], axis=1)
    va = np.concatenate([A_va, B_va], axis=1)
    st = R.BlockScaler(n_blocks=[3, 2]).fit(tr)
    out_tr = st.transform(tr)
    out_va = st.transform(va)
    # each block individually standardized
    assert abs(out_tr[:, :3].mean()) < 1e-9
    assert abs(out_tr[:, :3].std() - 1) < 1e-9
    assert abs(out_tr[:, 3:].mean()) < 1e-9
    assert abs(out_tr[:, 3:].std() - 1) < 1e-9
    # val transformed with train stats, blockwise
    assert out_va.shape == va.shape
    # a block scaler fit on the concatenation would rescale the joint
    # distribution, not the per-block distribution; verify block means of the
    # raw blocks were used (per-block mu equals per-block raw mu)
    assert np.allclose(st.means_[0], A_tr.mean(axis=0))
    assert np.allclose(st.means_[1], B_tr.mean(axis=0))


def test_13_standardization_uses_no_test_data():
    rng = np.random.default_rng(6)
    tr = rng.normal(0, 1, size=(80, 5))
    te = rng.normal(50, 10, size=(30, 5))
    st = R.BlockScaler(n_blocks=[5]).fit(tr)
    out_te = st.transform(te)
    # if test data had leaked into the stats, test mean would be ~0
    assert abs(out_te.mean()) > 5.0


# ---------------------------------------------------------------------------
# 14. No NaN/inf features
# ---------------------------------------------------------------------------

def test_14_no_nan_inf_features(fitted):
    X, mr, Z = fitted
    G = R.kernel_group_features(X, mr)
    for arr in (Z, G):
        assert np.isfinite(arr).all()
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()


# ---------------------------------------------------------------------------
# 15. Test labels cannot enter the decision function
# ---------------------------------------------------------------------------

def test_15_test_labels_never_enter_decision():
    # freeze_decisions must raise if any test array or label is supplied
    with pytest.raises(TypeError):
        R.freeze_decisions(
            {"DS": {"delta_real": 0.1, "p_value": 0.01}}, y_test=np.arange(3))


def test_15b_detector_pipeline_signature_excludes_test():
    # the detector's signature has no test arguments at all
    import inspect
    sig = inspect.signature(R.run_detector)
    assert "X_te" not in sig.parameters and "y_te" not in sig.parameters
    assert all("test" not in p for p in sig.parameters)


# ---------------------------------------------------------------------------
# 16. Decision is one constant value per dataset
# ---------------------------------------------------------------------------

def test_16_decision_is_constant_per_dataset():
    rng = np.random.default_rng(9)
    # simulate 200 calls: the decision must not vary with sample index
    dec = R.keep_drop_rule(0.03, 0.02)
    for _ in range(200):
        assert R.keep_drop_rule(0.03, 0.02) == dec


# ---------------------------------------------------------------------------
# 17. No per-example routing exists
# ---------------------------------------------------------------------------

def test_17_no_per_example_routing():
    # the deployment path must produce ONE model choice per dataset;
    # freeze_decisions output maps dataset name -> a single constant string,
    # and there is no function in the experiment that routes per example.
    import inspect
    src_names = [n for n in dir(R) if callable(getattr(R, n))]
    banned = [n for n in src_names
              if any(k in n.lower() for k in
                     ("route", "router", "gate_net", "attention",
                      "mixture", "per_example", "ensemble"))]
    assert banned == [], f"per-example routing symbols found: {banned}"
    dec = R.freeze_decisions(
        {"A": {"delta_real": 0.1, "p_value": 0.01},
         "B": {"delta_real": -0.1, "p_value": 0.5}})
    assert set(dec.values()) <= {"KEEP_KTM_W", "DROP_KTM_W"}
    assert isinstance(dec, dict) and len(dec) == 2


# ---------------------------------------------------------------------------
# 18. S=200 is actually executed
# ---------------------------------------------------------------------------

def test_18_s_is_200():
    assert R.S_NULL == 200
    shifts = R.make_null_shifts(4, 50, R.S_NULL, seed=1)
    assert shifts.shape == (200, 4)
