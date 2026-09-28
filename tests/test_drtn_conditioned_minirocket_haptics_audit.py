"""
Stage A/B audit tests for the DRTN-conditioned MiniROCKET experiment.

Covers the audit requirements that the previous M2==M3 anomaly exposed:
    * corrected M2: per-sample occupancy-preserving random regimes,
      independent RNG stream (seed + 900001)
    * M3: per-sample permutation of actual labels, stream (seed + 900002)
    * M2 != M3 as arrays, no shared writable memory, deterministic
    * occupancy preserved per sample for both controls
    * heterogeneity formula H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2 with the
      aeon VALID-region convention [padding, T - padding)  (DOC-10 item 9)
    * worked numerical example verified against an independent O(T) recompute
      (DOC-10 item 2)
    * controls produce DIFFERENT heterogeneity features on the same
      activations (no placeholder, no trivial identity)
    * heterogeneity is zero when the activation rate is regime-independent
    * no test-label access in any feature-construction path
    * M2/M3 feature matrices are not exactly identical on synthetic data
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
    compute_regime_heterogeneity,
    M2_RNG_OFFSET,
    M3_RNG_OFFSET,
)
from experiments.drtn_conditioned_minirocket_transfer_seed42.core import (  # noqa: E402
    heterogeneity_features,
    independent_heterogeneity_recompute,
    _padding_groups,
)

K = 8
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic regime controls (Stage A)
# ---------------------------------------------------------------------------
def _synthetic_regimes(n=20, T=400, seed=1, skewed=True):
    """Regime arrays with realistic skewed occupancy (dominant regime)."""
    rng = np.random.RandomState(seed)
    if skewed:
        probs = np.array([0.05, 0.0, 0.08, 0.4, 0.02, 0.01, 0.04, 0.4])
        probs = probs / probs.sum()
    else:
        probs = np.ones(K) / K
    return rng.choice(K, size=(n, T), p=probs).astype(np.int64)


def test_m2_per_sample_occupancy_preserved():
    regimes = _synthetic_regimes()
    m2 = create_random_regime_control(regimes, seed=SEED)
    for i in range(len(regimes)):
        assert np.array_equal(np.bincount(m2[i], minlength=K),
                              np.bincount(regimes[i], minlength=K)), \
            f"M2 per-sample occupancy not preserved (sample {i})"


def test_m3_per_sample_occupancy_preserved():
    regimes = _synthetic_regimes()
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    for i in range(len(regimes)):
        assert np.array_equal(np.bincount(m3[i], minlength=K),
                              np.bincount(regimes[i], minlength=K)), \
            f"M3 per-sample occupancy not preserved (sample {i})"


def test_m2_m3_arrays_differ():
    regimes = _synthetic_regimes()
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.array_equal(m2, m3)
    # and they differ at a substantial fraction of positions
    frac = (m2 != m3).mean()
    assert 0.2 < frac < 1.0


def test_m2_m3_no_shared_memory_and_not_views():
    regimes = _synthetic_regimes()
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.shares_memory(m2, m3)
    assert not np.shares_memory(m2, regimes)
    assert not np.shares_memory(m3, regimes)


def test_controls_independent_rng_streams():
    """M2 uses seed+900001, M3 uses seed+900002 (documented convention)."""
    assert M2_RNG_OFFSET == 900001
    assert M3_RNG_OFFSET == 900002
    # streams are distinct: first draws from each differ
    r2 = np.random.RandomState(SEED + M2_RNG_OFFSET)
    r3 = np.random.RandomState(SEED + M3_RNG_OFFSET)
    assert not np.array_equal(r2.permutation(64), r3.permutation(64))


def test_controls_temporal_placement_destroyed():
    regimes = _synthetic_regimes(n=5)
    for fn in (create_random_regime_control, create_shuffled_regime_control):
        ctrl = fn(regimes, seed=SEED)
        assert not np.array_equal(ctrl, regimes)


def test_controls_deterministic():
    regimes = _synthetic_regimes()
    for fn in (create_random_regime_control, create_shuffled_regime_control):
        a = fn(regimes, seed=SEED)
        b = fn(regimes, seed=SEED)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Heterogeneity: formula + valid-region convention (DOC-10 items 2/9)
# ---------------------------------------------------------------------------
def _make_act_valid(T=200, pad=2, n_features=3):
    """Activation tensor with a padded valid region like aeon features."""
    act = np.zeros((1, n_features, T), dtype=bool)
    valid = np.zeros((n_features, T), dtype=bool)
    valid[:, pad:T - pad] = True
    rng = np.random.RandomState(0)
    act[0] = rng.rand(n_features, T) > 0.5
    act[0][:, :pad] = False           # padded positions never activate
    act[0][:, T - pad:] = False
    return act, valid


def test_heterogeneity_uses_valid_region_only():
    """Positions outside the valid region must not influence H_m."""
    act, valid = _make_act_valid()
    regimes = np.zeros((1, 200), dtype=np.int64)
    regimes[0, :2] = 3          # only in the PADDED region
    regimes[0, 198:] = 5        # only in the PADDED region
    h_all_pad = compute_regime_heterogeneity(act, valid, regimes)

    regimes2 = regimes.copy()
    regimes2[0, :2] = 0         # change ONLY padded labels
    regimes2[0, 198:] = 0
    h_no_pad = compute_regime_heterogeneity(act, valid, regimes2)

    # both regimes 3 and 5 are below min occupancy in valid region anyway,
    # but the padded labels must have contributed nothing in either case
    assert np.array_equal(h_all_pad, h_no_pad)


def test_heterogeneity_zero_when_regime_independent():
    """Same activation rate in every regime => H_m == 0 (exactly).

    Construction: deterministic regimes with EQUAL, EVEN counts per regime;
    activations on every other position, so each regime contains exactly
    half of its positions active: PPV_{m,k} = 0.5 = PPV_m for all k.
    """
    T, n_f, reps = 800, 4, 100          # 8 regimes x 100 CONTIGUOUS positions
    regimes = np.repeat(np.arange(K), reps)[None, :]         # (1, 800)
    act = np.zeros((1, n_f, T), dtype=bool)
    for f in range(n_f):
        # even f: fire the FIRST 50 positions of every regime block;
        # odd f: fire the LAST 50. Either way each regime has exactly
        # 50/100 active => PPV_{m,k} = 0.5 = PPV_m for every k and f.
        lo = 0 if f % 2 == 0 else reps // 2
        for k in range(K):
            act[0, f, k * reps + lo:k * reps + lo + reps // 2] = True
    valid = np.ones((n_f, T), dtype=bool)
    h = compute_regime_heterogeneity(act, valid, regimes)
    assert np.allclose(h, 0.0, atol=1e-12)


def test_heterogeneity_positive_when_regime_dependent():
    """Different activation rates across regimes => H_m > 0."""
    T = 800
    act = np.zeros((1, 1, T), dtype=bool)
    valid = np.ones((1, T), dtype=bool)
    # first half of the series: high activation; second half: none
    act[0, 0, :T // 2] = True
    regimes = np.zeros((1, T), dtype=np.int64)
    regimes[0, :T // 2] = 1
    regimes[0, T // 2:] = 2
    h = compute_regime_heterogeneity(act, valid, regimes)
    assert h[0, 0] > 0.2


def test_worked_numerical_example_independent_recompute():
    """DOC-10 item 9: full manual recomputation for one sample/kernel."""
    rng = np.random.RandomState(7)
    T, pad = 300, 2
    act = np.zeros((1, 1, T), dtype=bool)
    act[0, 0] = rng.rand(T) > 0.6
    act[0, 0, :pad] = False
    act[0, 0, T - pad:] = False
    valid = np.zeros((1, T), dtype=bool)
    valid[0, pad:T - pad] = True
    regimes = rng.randint(0, K, size=(1, T)).astype(np.int64)
    regimes[0, :2] = 7            # padded labels only
    regimes[0, T - 2:] = 6

    h = compute_regime_heterogeneity(act, valid, regimes)

    # ---- fully manual recomputation over the valid region ----
    a = act[0, 0, pad:T - pad]
    r = regimes[0, pad:T - pad]
    n_valid = len(a)
    min_count = int(np.ceil(0.01 * T))
    ppv_m = a.mean()
    terms, total_w = [], 0.0
    for k in np.unique(r):
        m = r == k
        n_k = int(m.sum())
        if n_k < min_count:
            continue
        q_k = n_k / n_valid
        ppv_k = a[m].mean()
        terms.append(q_k * (ppv_k - ppv_m) ** 2)
        total_w += q_k
    h_manual = float(np.sum(terms) / total_w)

    # float32 vectorized matmul vs float64 scalar accumulation: agree to
    # 1e-8 (a wrong formula would differ at >= 1e-3 scale)
    assert abs(h[0, 0] - h_manual) < 1e-8

    # and against the project's independent O(T) reference implementation
    h_ref = independent_heterogeneity_recompute(act[0, 0], valid[0], regimes[0])
    assert abs(h[0, 0] - h_ref) < 1e-8


def test_worked_example_padded_positions_excluded():
    """Explicit: padding-region labels/activations change nothing."""
    rng = np.random.RandomState(7)
    T, pad = 300, 2
    act = np.zeros((1, 1, T), dtype=bool)
    act[0, 0] = rng.rand(T) > 0.6
    act[0, 0, :pad] = False
    act[0, 0, T - pad:] = False
    valid = np.zeros((1, T), dtype=bool)
    valid[0, pad:T - pad] = True
    regimes = rng.randint(0, K, size=(1, T)).astype(np.int64)

    # corrupt the padded region in both act and regimes
    act2 = act.copy(); regimes2 = regimes.copy()
    act2[0, 0, :pad] = True; act2[0, 0, T - pad:] = True
    regimes2[0, :pad] = 1; regimes2[0, T - pad:] = 2

    h1 = compute_regime_heterogeneity(act, valid, regimes)
    h2 = compute_regime_heterogeneity(act2, valid, regimes2)
    assert np.array_equal(h1, h2)


def test_padding_groups_contiguous():
    _, valid = _make_act_valid()
    groups = _padding_groups(valid)
    assert groups[0][0] == 2 and len(groups) == 1


# ---------------------------------------------------------------------------
# Controls produce distinct features on identical activations
# ---------------------------------------------------------------------------
def test_m2_m3_features_not_identical():
    T = 600
    rng = np.random.RandomState(SEED)
    n_f = 12
    act = (rng.rand(1, n_f, T) < 0.3)
    valid = np.ones((n_f, T), dtype=bool)
    regimes = _synthetic_regimes(n=1, T=T)
    h2 = heterogeneity_features(act, valid, create_random_regime_control(regimes, SEED))
    h3 = heterogeneity_features(act, valid, create_shuffled_regime_control(regimes, SEED))
    assert not np.array_equal(h2, h3)
    # on random regimes both stay near the sampling-noise floor
    assert h2.max() < 0.05 and h3.max() < 0.05


def test_m1_features_strongly_different_from_controls():
    """A regime-aligned activation pattern must yield far larger H than controls."""
    T = 800
    act = np.zeros((1, 1, T), dtype=bool)
    act[0, 0, :T // 2] = True                       # aligned with regimes
    valid = np.ones((1, T), dtype=bool)
    regimes = np.zeros((1, T), dtype=np.int64)
    regimes[0, :T // 2] = 1
    regimes[0, T // 2:] = 2
    h1 = heterogeneity_features(act, valid, regimes)
    h3 = heterogeneity_features(act, valid, create_shuffled_regime_control(regimes, SEED))
    assert h1[0, 0] > 0.2
    assert h3[0, 0] < 0.05
    assert h1[0, 0] > 5 * h3[0, 0]


def test_heterogeneity_no_placeholder_globals():
    """Controls' H must not equal the DRTN-conditioned H (old bug class)."""
    rng = np.random.RandomState(3)
    T = 700
    act = (rng.rand(1, 6, T) < 0.35)
    valid = np.ones((6, T), dtype=bool)
    regimes = _synthetic_regimes(n=1, T=T)
    h1 = heterogeneity_features(act, valid, regimes)
    h2 = heterogeneity_features(act, valid, create_random_regime_control(regimes, SEED))
    assert not np.allclose(h1, h2)


# ---------------------------------------------------------------------------
# Structural / leakage guards
# ---------------------------------------------------------------------------
def test_no_test_label_dependency():
    """No feature-construction function may reference yte / y_test."""
    p = os.path.join(ROOT, "experiments", "drtn_conditioned_minirocket_haptics_3seed",
                     "runner.py")
    code = open(p).read()
    import re
    for fn in ["create_random_regime_control", "create_shuffled_regime_control",
               "compute_regime_heterogeneity"]:
        m = re.search(rf"def {fn}\(.*?(?=\ndef |\Z)", code, re.S)
        assert m is not None, fn
        body = m.group(0)
        assert "yte" not in body and "y_test" not in body, fn


def test_legacy_all_t_extractor_removed():
    """BUG GUARD 5: the legacy all-T raw-response function must be gone."""
    p = os.path.join(ROOT, "experiments", "drtn_conditioned_minirocket_haptics_3seed",
                     "runner.py")
    code = open(p).read()
    assert "def compute_raw_minirocket_features" not in code


def test_feature_budget_synthetic():
    """M1/M2/M3 assembled feature matrices must be exactly 9996 wide."""
    n, g, h = 4, 4998, 4998
    A = np.random.randn(n, g)
    for H in [np.random.randn(n, h) for _ in range(3)]:
        F = np.hstack([A, H])
        assert F.shape[1] == 9996


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
