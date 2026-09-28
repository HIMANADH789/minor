"""
Tests for the corrected negative-dataset re-test (ECG5000_UNBAL, CWRU_UNBAL).

Covers the mandatory audits (1-8) on the shared audited implementation plus
the exact dataset configurations and canonical M0 smoke reproduction.
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
from experiments.drtn_conditioned_minirocket_corrected_negative_retest.runner import (  # noqa: E402
    CANONICAL_M0_REF, OLD_PREAUDIT, N_FEATURES, N_GLOBAL, ALPHAS, M0_TOLERANCE,
)

K = 8
SEED = 42


def _regimes(n=10, T=300, seed=5):
    rng = np.random.RandomState(seed)
    probs = np.array([0.05, 0.0, 0.08, 0.4, 0.02, 0.01, 0.04, 0.4])
    return rng.choice(K, size=(n, T), p=probs / probs.sum()).astype(np.int64)


# ---------------- dataset configs ----------------
def test_exact_dataset_configs():
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as tr
    assert set(tr.NPZ_SPECS) >= {"ECG5000_UNBAL", "CWRU_UNBAL"}
    assert tr.NPZ_SPECS["ECG5000_UNBAL"] == ("data/ecg5000_resplit.npz", 5)
    assert tr.NPZ_SPECS["CWRU_UNBAL"] == ("data/cwru_unbalanced.npz", 4)
    assert tr.NPZ_REFS["ECG5000_UNBAL"] == pytest.approx(0.5938, abs=1e-6)
    assert tr.NPZ_REFS["CWRU_UNBAL"] == pytest.approx(0.9917, abs=1e-6)


@pytest.mark.parametrize("ds", ["ECG5000_UNBAL", "CWRU_UNBAL"])
def test_dataset_resolution(ds):
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as tr
    d = tr.load_any_dataset(ds)
    assert d["n_classes"] == tr.NPZ_SPECS[ds][1]
    assert d["Xtr"].shape[1] == d["Xva"].shape[1] == d["Xte"].shape[1] == d["L"]
    assert len(d["Xtr"]) == len(d["ytr"]) and len(d["Xte"]) == len(d["yte"])
    assert not np.isnan(d["Xtr"].astype(np.float64)).any()


def test_canonical_references_recorded():
    assert CANONICAL_M0_REF == {"ECG5000_UNBAL": 0.5938, "CWRU_UNBAL": 0.9917}
    assert OLD_PREAUDIT["ECG5000_UNBAL"] == {
        "M0": 0.5938, "M1": 0.5859, "M2": 0.5716, "M3": 0.5565}
    assert OLD_PREAUDIT["CWRU_UNBAL"] == {
        "M0": 0.9917, "M1": 0.9792, "M2": 0.9625, "M3": 0.9583}


# ---------------- budget / allocation ----------------
def test_budget_and_alpha_grid():
    assert N_FEATURES == 9996 and N_GLOBAL == 4998
    assert np.allclose(ALPHAS, np.logspace(-4, 4, 20))
    assert M0_TOLERANCE == pytest.approx(0.0011)


# ---------------- audited controls ----------------
def test_m2_m3_distinct_occupancy_no_aliasing():
    regimes = _regimes()
    m2 = create_random_regime_control(regimes, seed=SEED)
    m3 = create_shuffled_regime_control(regimes, seed=SEED)
    assert not np.array_equal(m2, m3)
    assert not np.shares_memory(m2, m3)
    assert not np.shares_memory(m2, regimes)
    for ctrl in (m2, m3):
        for i in range(len(regimes)):
            assert np.array_equal(np.bincount(ctrl[i], minlength=K),
                                  np.bincount(regimes[i], minlength=K))


def test_independent_rng_streams():
    assert M2_RNG_OFFSET == 900001 and M3_RNG_OFFSET == 900002
    r2 = np.random.RandomState(SEED + M2_RNG_OFFSET)
    r3 = np.random.RandomState(SEED + M3_RNG_OFFSET)
    assert not np.array_equal(r2.permutation(64), r3.permutation(64))


def test_controls_deterministic():
    regimes = _regimes()
    for fn in (create_random_regime_control, create_shuffled_regime_control):
        assert np.array_equal(fn(regimes, seed=SEED), fn(regimes, seed=SEED))


# ---------------- H formula / valid region / raw path ----------------
def test_exact_formula_worked_example():
    rng = np.random.RandomState(7)
    T, pad = 300, 2
    act = np.zeros((1, 1, T), dtype=bool)
    act[0, 0] = rng.rand(T) > 0.6
    act[0, 0, :pad] = False
    act[0, 0, T - pad:] = False
    valid = np.zeros((1, T), dtype=bool)
    valid[0, pad:T - pad] = True
    regimes = rng.randint(0, K, size=(1, T)).astype(np.int64)
    h = compute_regime_heterogeneity(act, valid, regimes)

    a = act[0, 0, pad:T - pad]; r = regimes[0, pad:T - pad]
    n_valid = len(a); min_count = int(np.ceil(0.01 * T))
    ppv_m = a.mean(); terms, tw = [], 0.0
    for k in np.unique(r):
        m = r == k; n_k = int(m.sum())
        if n_k < min_count:
            continue
        terms.append((n_k / n_valid) * (a[m].mean() - ppv_m) ** 2)
        tw += n_k / n_valid
    assert abs(h[0, 0] - float(np.sum(terms) / tw)) < 1e-8


def test_valid_region_only():
    T, pad, n_f = 200, 2, 3
    act = np.zeros((1, n_f, T), dtype=bool)
    valid = np.zeros((n_f, T), dtype=bool)
    valid[:, pad:T - pad] = True
    act[0] = np.random.RandomState(0).rand(n_f, T) > 0.5
    act[0][:, :pad] = False
    act[0][:, T - pad:] = False
    r1 = np.zeros((1, T), dtype=np.int64); r2 = r1.copy()
    r1[0, :2] = 3; r1[0, 198:] = 4
    r2[0, :2] = 0; r2[0, 198:] = 0
    assert np.array_equal(compute_regime_heterogeneity(act, valid, r1),
                          compute_regime_heterogeneity(act, valid, r2))


def test_aligned_vs_control_scale():
    T = 800
    act = np.zeros((1, 1, T), dtype=bool)
    act[0, 0, :T // 2] = True
    valid = np.ones((1, T), dtype=bool)
    regimes = np.zeros((1, T), dtype=np.int64)
    regimes[0, :T // 2] = 1; regimes[0, T // 2:] = 2
    h1 = compute_regime_heterogeneity(act, valid, regimes)
    h3 = compute_regime_heterogeneity(
        act, valid, create_shuffled_regime_control(regimes, SEED))
    assert h1[0, 0] > 5 * h3[0, 0]


# ---------------- structural guards ----------------
def test_trainval_fit_and_variant_local_features():
    p = os.path.join(ROOT, "experiments",
                     "drtn_conditioned_minirocket_corrected_negative_retest",
                     "runner.py")
    code = open(p).read()
    assert "ridge.fit(F_tr, ytrva)" in code
    assert "ridge.predict(F_te_v)" in code


def test_no_test_label_dependency():
    import re
    p = os.path.join(ROOT, "experiments",
                     "drtn_conditioned_minirocket_haptics_3seed", "runner.py")
    code = open(p).read()
    for fn in ["create_random_regime_control", "create_shuffled_regime_control",
               "compute_regime_heterogeneity"]:
        m = re.search(rf"def {fn}\(.*?(?=\ndef |\Z)", code, re.S)
        body = m.group(0)
        assert "yte" not in body and "y_test" not in body


def test_znorm_before_minirocket_in_runner():
    """Canonical convention: znorm applied BEFORE MiniRocket (audited fix)."""
    p = os.path.join(ROOT, "experiments",
                     "drtn_conditioned_minirocket_corrected_negative_retest",
                     "runner.py")
    code = open(p).read()
    assert "extractor.fit(Xtr_z" in code
    assert "compute_raw_activations(extractor, Xtr_z)" in code


# ---------------- canonical M0 smoke (slow, opt-in) ----------------
@pytest.mark.slow
@pytest.mark.parametrize("ds,ref", [("ECG5000_UNBAL", 0.5938),
                                    ("CWRU_UNBAL", 0.9917)])
def test_canonical_m0_seed42_smoke(ds, ref):
    from sklearn.linear_model import RidgeClassifierCV
    import experiments.drtn_conditioned_minirocket_transfer_seed42.runner as tr
    from aeon.transformations.collection.convolution_based import MiniRocket

    d = tr.load_any_dataset(ds)
    tr.set_seed(42)
    ext = MiniRocket(random_state=42, n_jobs=-1)
    Xtr_z, Xva_z, Xte_z = tr.znorm(d["Xtr"]), tr.znorm(d["Xva"]), tr.znorm(d["Xte"])
    ext.fit(Xtr_z[:, None, :].astype(np.float32))
    Ftr = ext.transform(Xtr_z[:, None, :].astype(np.float32))
    Fva = ext.transform(Xva_z[:, None, :].astype(np.float32))
    Fte = ext.transform(Xte_z[:, None, :].astype(np.float32))
    ytrva = np.concatenate([d["ytr"], d["yva"]])
    ridge = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)).fit(
        np.vstack([Ftr, Fva]), ytrva)
    mf1 = float(f1 := __import__("sklearn.metrics", fromlist=["f1_score"])
                .f1_score(d["yte"], ridge.predict(Fte), average="macro",
                          zero_division=0))
    assert abs(mf1 - ref) < M0_TOLERANCE, (ds, mf1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
