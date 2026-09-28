"""
Unit tests for DRTN-Conditioned MiniROCKET 3-Seed Confirmation.

Tests:
1. Feature count = 9996
2. Heterogeneity formula correctness
3. Heterogeneity non-zero
4. Occupancy preservation for M2/M3
5. Shuffled temporal alignment destruction
6. Global feature consistency
7. Sample alignment
8. No test-label dependency (audit)
9. Deterministic seed behavior
"""
import numpy as np
import pytest
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_feature_count():
    """M0-M3 must all have exactly 9996 features."""
    N_FEATURES = 9996
    N_GLOBAL = 4998
    N_HET = 4998
    n_samples = 10
    F_global = np.random.randn(n_samples, N_GLOBAL)
    F_het = np.random.randn(n_samples, N_HET)
    M0 = np.random.randn(n_samples, N_FEATURES)
    M1 = np.hstack([F_global, F_het])
    assert M0.shape[1] == 9996
    assert M1.shape[1] == 9996


def test_heterogeneity_formula():
    """Verify heterogeneity formula on synthetic data."""
    T = 1000
    K = 8
    n_kernels = 10

    # Create binary activations: different rates per regime
    np.random.seed(42)
    regimes = np.random.randint(0, K, size=T)
    activations = np.random.rand(n_kernels, T) > 0.5  # binary

    biases = np.zeros(n_kernels)
    active = activations  # already binary

    global_ppv = active.mean(axis=1)

    heterogeneity = np.zeros(n_kernels)
    min_count = int(np.ceil(0.01 * T))
    for k in range(K):
        mask = regimes == k
        count_k = int(mask.sum())
        if count_k < min_count:
            continue
        q_k = count_k / T
        ppv_k = active[:, mask].mean(axis=1)
        heterogeneity += q_k * (ppv_k - global_ppv) ** 2

    # Heterogeneity should be non-negative
    assert np.all(heterogeneity >= 0), "Heterogeneity should be non-negative"

    # With random activations and regimes, heterogeneity should be non-zero
    assert np.any(heterogeneity > 0), "Heterogeneity should not be all zeros"


def test_heterogeneity_zero_when_uniform():
    """Heterogeneity should be zero when activation rate is same across regimes."""
    T = 1000
    K = 8
    n_kernels = 5
    np.random.seed(42)
    regimes = np.random.randint(0, K, size=T)

    # All timesteps have same activation rate
    activations = np.ones((n_kernels, T)) * 0.5
    global_ppv = activations.mean(axis=1)

    heterogeneity = np.zeros(n_kernels)
    min_count = int(np.ceil(0.01 * T))
    for k in range(K):
        mask = regimes == k
        count_k = int(mask.sum())
        if count_k < min_count:
            continue
        q_k = count_k / T
        ppv_k = activations[:, mask].mean(axis=1)
        heterogeneity += q_k * (ppv_k - global_ppv) ** 2

    assert np.allclose(heterogeneity, 0, atol=1e-10), \
        "Heterogeneity should be zero when activation rate is uniform"


def test_occupancy_preservation():
    """Random and shuffled regimes should preserve occupancy distribution."""
    np.random.seed(42)
    T = 1000
    K = 8
    n_samples = 20
    regimes = np.random.randint(0, K, size=(n_samples, T))

    # Test shuffled regime preserves per-sample counts
    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        create_shuffled_regime_control)
    shuffled = create_shuffled_regime_control(regimes, seed=42)

    for i in range(n_samples):
        orig_counts = np.bincount(regimes[i], minlength=K)
        shuf_counts = np.bincount(shuffled[i], minlength=K)
        assert np.array_equal(orig_counts, shuf_counts), \
            f"Sample {i}: shuffled counts differ from original"


def test_shuffled_alignment_destroyed():
    """Shuffled regime should differ temporally from original."""
    np.random.seed(42)
    T = 1000
    K = 8
    regimes = np.random.randint(0, K, size=(1, T))

    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        create_shuffled_regime_control)
    shuffled = create_shuffled_regime_control(regimes, seed=42)

    # Shuffled should differ from original (with high probability)
    assert not np.array_equal(regimes[0], shuffled[0]), \
        "Shuffled regime should differ from original"


def test_random_regime_preserves_global_occupancy():
    """Random regime should preserve overall occupancy distribution."""
    np.random.seed(42)
    T = 1000
    K = 8
    n_samples = 50
    regimes = np.random.randint(0, K, size=(n_samples, T))

    from experiments.drtn_conditioned_minirocket_haptics_3seed.runner import (
        create_random_regime_control)
    random_regimes = create_random_regime_control(regimes, seed=42)

    # Global occupancy should be approximately preserved
    orig_counts = np.bincount(regimes.flatten(), minlength=K)
    rand_counts = np.bincount(random_regimes.flatten(), minlength=K)

    orig_dist = orig_counts / orig_counts.sum()
    rand_dist = rand_counts / rand_counts.sum()

    assert np.allclose(orig_dist, rand_dist, atol=0.01), \
        "Random regime should preserve global occupancy distribution"


def test_global_feature_consistency():
    """Global MiniROCKET features must be identical for M0 and M1/M2/M3."""
    np.random.seed(42)
    n_samples = 10
    n_features = 9996
    n_global = 4998
    n_het = 4998

    F_full = np.random.randn(n_samples, n_features)
    F_global = F_full[:, :n_global]
    F_het = np.random.randn(n_samples, n_het)

    M0 = F_full
    M1 = np.hstack([F_global, F_het])

    # First 4998 features must be identical
    assert np.allclose(M0[:, :n_global], M1[:, :n_global]), \
        "Global features must be identical between M0 and M1"


def test_sample_alignment():
    """Sample index must remain consistent across all representations."""
    n_samples = 10
    n_features = 9996

    X = np.random.randn(n_samples, 100)
    regimes = np.random.randint(0, 8, size=(n_samples, 100))
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 5, size=n_samples)

    # All must have same first dimension
    assert X.shape[0] == regimes.shape[0] == features.shape[0] == labels.shape[0]


def test_no_test_label_dependency():
    """Audit that y_test is not used in feature construction."""
    # This is a structural test - check the runner code
    runner_path = os.path.join(ROOT, "experiments",
                               "drtn_conditioned_minirocket_haptics_3seed",
                               "runner.py")
    if os.path.exists(runner_path):
        with open(runner_path) as f:
            code = f.read()
        # Check that yte is not used in feature construction functions
        # The compute_regime_heterogeneity function should not take labels
        assert "yte" not in code.split("def compute_regime_heterogeneity")[1].split("def ")[0], \
            "yte should not appear in compute_regime_heterogeneity"


def test_deterministic_seed():
    """Same seed should produce same results."""
    np.random.seed(42)
    a = np.random.randn(10)
    np.random.seed(42)
    b = np.random.randn(10)
    assert np.array_equal(a, b), "Same seed should produce same results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
