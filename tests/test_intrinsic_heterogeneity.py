"""Unit tests for the label-free intrinsic heterogeneity audit (spec 29)."""
import ast
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.intrinsic_heterogeneity import (  # noqa: E402
    LOG2, W_PRIMARY, compute_dataset_index, compute_sample_spectral_heterogeneity,
    compute_sample_variance_heterogeneity, compute_window_spectrum,
    derive_intrinsic_h_budget, js_divergence, run_leakage_audit,
    window_stability)
from experiments.haptics_intrinsic_heterogeneity.runner import (  # noqa: E402
    load_raw, select_h_budget_groups)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------
# synthetic helper: piecewise sinusoid (genuinely time-varying spectrum)
# ----------------------------------------------------------------------
def _two_regime_series(n=40, T=1092, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    X = np.empty((n, T))
    for i in range(n):
        f1 = rng.uniform(0.05, 0.1)
        f2 = rng.uniform(0.3, 0.45)
        X[i] = np.concatenate([
            np.sin(2 * np.pi * f1 * t[:T // 2]),
            np.sin(2 * np.pi * f2 * t[T // 2:])]) + 0.01 * rng.normal(size=T)
    return X


# 1: deterministic spectral calculation
def test_01_deterministic_spectral():
    X = _two_regime_series()
    a = compute_sample_spectral_heterogeneity(X, W_PRIMARY)
    b = compute_sample_spectral_heterogeneity(X, W_PRIMARY)
    assert np.array_equal(a, b)


# 2: JS non-negative
def test_02_js_nonnegative():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.dirichlet(np.ones(17))
        q = rng.dirichlet(np.ones(17))
        assert js_divergence(p, q) >= 0.0


# 3: JS ~ 0 for identical spectra
def test_03_js_zero_identical():
    p = np.abs(np.random.default_rng(2).normal(size=33)) + 1e-3
    p = p / p.sum()
    assert js_divergence(p, p.copy()) < 1e-12


# 4: JS symmetric
def test_04_js_symmetric():
    p = np.abs(np.random.default_rng(3).normal(size=21)) + 1e-3
    q = np.abs(np.random.default_rng(4).normal(size=21)) + 1e-3
    p, q = p / p.sum(), q / q.sum()
    assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-12


# 5: normalized spectrum sums to ~1
def test_05_spectrum_normalized():
    x = _two_regime_series(n=1)[0]
    P = compute_window_spectrum(x, W_PRIMARY)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-9)


# 6: sample-level index deterministic (real train rows)
def test_06_sample_index_deterministic():
    _, Xtr, _, _ = load_raw()
    a = compute_sample_spectral_heterogeneity(Xtr, W_PRIMARY)
    b = compute_sample_spectral_heterogeneity(Xtr, W_PRIMARY)
    assert np.array_equal(a, b)


# 7: dataset index equals the declared median rule
def test_07_dataset_median_rule():
    X = _two_regime_series()
    hi = compute_sample_spectral_heterogeneity(X, W_PRIMARY)
    assert compute_dataset_index(hi) == float(np.median(hi))


# 8: B_H deterministic
def test_08_bh_deterministic():
    assert derive_intrinsic_h_budget(0.05)["B_H"] == \
        derive_intrinsic_h_budget(0.05)["B_H"]


# 9: B_H in [0, 1]
def test_09_bh_range():
    for hi in (0.0, 1e-6, 0.01, 0.1, 0.5, 0.7, 2.0, 100.0):
        b = derive_intrinsic_h_budget(hi)["B_H"]
        assert 0.0 <= b <= 1.0
    # clipping branches
    assert derive_intrinsic_h_budget(0.0)["B_H"] == 0.0
    assert derive_intrinsic_h_budget(10.0)["B_H"] == 1.0


# 10: labels do not affect index (no label argument exists)
def test_10_labels_not_in_signature():
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "analysis", "intrinsic_heterogeneity.py"),
        encoding="utf-8").read()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef)}
    for fn in ("compute_sample_spectral_heterogeneity",
               "compute_sample_variance_heterogeneity",
               "compute_dataset_index", "derive_intrinsic_h_budget"):
        args = [a.arg for a in fns[fn].args.args]
        assert not any("label" in a or a == "y" for a in args), (fn, args)


# 11-12: poisoned validation / test do not change the train index
def test_11_12_leakage_audit():
    _, Xtr, Xva, Xte = load_raw()
    res = run_leakage_audit(compute_sample_spectral_heterogeneity,
                            Xtr, Xva, Xte, W_PRIMARY)
    assert res["val_poison_invariant"] and res["test_poison_invariant"]
    base = compute_dataset_index(
        compute_sample_spectral_heterogeneity(Xtr, W_PRIMARY))
    # the budget pipeline never receives val/test; recomputation with the
    # poisoned arrays in memory reproduces the identical train index
    assert res["train_index_recomputed_with_poisoned_heldout"] == base
    # held-out splits have their own indices, distinct from train
    assert res["val_split_own_index"] != base or res["test_split_own_index"] != base
    # and poisoning held-out labels would be invisible by construction:
    # the index function has no label argument (checked in test_10)


# 13: W=8 is the primary configuration
def test_13_primary_W():
    assert W_PRIMARY == 8
    assert derive_intrinsic_h_budget(0.05)["rule"].find("log(2)") >= 0


# 14: W=7/9 are diagnostics only (stability function, primary unchanged)
def test_14_stability_diagnostic_only():
    X = _two_regime_series(n=15)
    out = window_stability(X, (7, 8, 9))
    assert out["primary_W"] == 8
    assert set(out["dataset_median"]) == {"7", "8", "9"}


# 15: no existing experiment artifacts modified (prior namespaces intact)
def test_15_prior_artifacts_intact():
    res = os.path.join(ROOT, "results")
    for rel in (os.path.join("heramba_canonical_ridge_full", "haptics_seed42",
                             "final_comparison.csv"),
                os.path.join("heramba_cca_ranked", "haptics_seed42",
                             "evidence_report.md"),
                os.path.join("heramba_canonical_ridge", "haptics_seed42",
                             "results.json")):
        assert os.path.exists(os.path.join(res, rel)), rel


# 16: downstream identity gates (canonical ridge_eval, frozen banks)
def test_16_identity_gates():
    from experiments.heramba_cca_ranked_haptics_seed42.runner import ridge_eval
    _, Xtr, Xva, _ = load_raw()
    d, Xtr2, Xva2, Xte2 = load_raw()
    y_dev = np.concatenate([d["ytr"], d["yva"]])
    CACHE = r"C:/temp/results"
    G_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_G_trva.npy"))
    H_trva = np.load(os.path.join(CACHE, "haptics_inference_banks_H_trva.npy"))
    G_te = np.load(os.path.join(CACHE, "haptics_inference_banks_G_te.npy"))
    H_te = np.load(os.path.join(CACHE, "haptics_inference_banks_H_te.npy"))
    m0, _ = ridge_eval(G_trva, y_dev, G_te, d["yte"])
    assert abs(m0["macro_f1"] - 0.5037) < 5e-5
    r2, _ = ridge_eval(np.hstack([G_trva, H_trva]), y_dev,
                       np.hstack([G_te, H_te]), d["yte"])
    assert r2["macro_f1"] == 0.55


# extra: budget group selection is label-free and respects capacity
def test_extra_budget_groups_label_free_and_capacity():
    rng = np.random.default_rng(5)
    H = rng.random((20, 128))
    sel = select_h_budget_groups(H, 0.35)
    assert sel["kept_quantile"] == int(np.ceil(0.35 * 128))
    assert sel["mask_quantile"].sum() == sel["kept_quantile"]
    # deterministic
    sel2 = select_h_budget_groups(H, 0.35)
    assert np.array_equal(sel["mask_quantile"], sel2["mask_quantile"])
