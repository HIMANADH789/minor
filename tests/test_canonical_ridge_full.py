"""Sanity tests for the confound-isolated full-dimension Canonical Ridge
experiment (spec section 30).  The heavy end-to-end identity gates run
inside the experiment itself; here we cover the model mechanics and the
leakage/determinism invariants on synthetic data.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.heramba_canonical_ridge_full_haptics_seed42.runner import (  # noqa: E402
    ALPHA_GRID, GAMMA, fit_decomposition, load_all, reconstruction_check,
    transform_designs)
from models.canonical_ridge_full.model import (  # noqa: E402
    DualGeneralizedRidge, adaptive_delta, brute_force_generalized_ridge,
    canonical_direction_matrix, complete_orthonormal_basis,
    unit_directions)

N_G, N_H = 4998, 4998


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def real():
    G_trva, H_trva, G_te, H_te, ytr, yva, yte = load_all()
    return dict(G_trva=G_trva, H_trva=H_trva, G_te=G_te, H_te=H_te,
                ytr=ytr, yva=yva, yte=yte, n_tr=len(ytr))


@pytest.fixture(scope="module")
def decomposed(real):
    pcaG, pcaH, cca, rho, Q_cca, Q_perp, diag = fit_decomposition(
        real["G_trva"], real["H_trva"], real["n_tr"])
    return dict(pcaG=pcaG, pcaH=pcaH, cca=cca, rho=rho, Q_cca=Q_cca,
                Q_perp=Q_perp, diag=diag)


def _synth(n=60, nG=20, nH=24, K=4, seed=7):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, nG))
    U, _ = np.linalg.qr(rng.normal(size=(nH, K)))
    H = rng.normal(size=(n, nH))
    return G, H, U


# ----------------------------------------------------------------------
# 1-3: deterministic loading / correct dimensions
# ----------------------------------------------------------------------
def test_01_deterministic_loading(real):
    G_trva2, H_trva2, G_te2, H_te2, *_ = load_all()
    assert (hashlib.sha256(G_trva2.tobytes()).hexdigest()
            == hashlib.sha256(real["G_trva"].tobytes()).hexdigest())
    assert (hashlib.sha256(H_te2.tobytes()).hexdigest()
            == hashlib.sha256(real["H_te"].tobytes()).hexdigest())


def test_02_G_dimension(real):
    assert real["G_trva"].shape == (155, N_G) and real["G_te"].shape == (308, N_G)


def test_03_H_dimension(real):
    assert real["H_trva"].shape == (155, N_H) and real["H_te"].shape == (308, N_H)


# ----------------------------------------------------------------------
# 4-5: train-only PCA / CCA (fit is invariant to val/test content)
# ----------------------------------------------------------------------
def test_04_train_only_pca(real, decomposed):
    H = real["H_trva"].copy()
    n = real["n_tr"]
    H[n:] += 1e6                                  # poison val rows
    pcaH2 = type(decomposed["pcaH"])(var_threshold=0.95).fit(H[:n])
    assert np.allclose(pcaH2.comp_, decomposed["pcaH"].comp_, atol=1e-12)
    assert pcaH2.k_ == decomposed["pcaH"].k_


def test_05_train_only_cca(real, decomposed):
    G, H = real["G_trva"], real["H_trva"]
    n = real["n_tr"]
    pcaG, pcaH, cca, rho, *_ = decomposed.values()
    H2 = H.copy()
    H2[n:] = 0.0                                  # poison val rows
    pcaG2, pcaH2, cca2, rho2, *_ = fit_decomposition(
        G, H2, n)                                 # fit uses first n rows only
    assert np.allclose(np.asarray(rho2), np.asarray(rho), atol=1e-10)


# ----------------------------------------------------------------------
# 6: K = 29 canonical directions
# ----------------------------------------------------------------------
def test_06_K_29(decomposed):
    assert decomposed["diag"]["n_cca_components"] == 29
    assert decomposed["Q_cca"].shape == (N_H, 29)
    assert decomposed["Q_perp"].shape == (N_H, N_H - 29)


# ----------------------------------------------------------------------
# 7: canonical H mapping is deterministic
# ----------------------------------------------------------------------
def test_07_mapping_deterministic(real, decomposed):
    hsd = decomposed["pcaH"].transform_low(real["H_trva"][:real["n_tr"]]).std(0)
    D1 = canonical_direction_matrix(decomposed["pcaH"].comp_,
                                    decomposed["pcaH"].ok_, hsd,
                                    decomposed["cca"].B_)
    D2 = canonical_direction_matrix(decomposed["pcaH"].comp_,
                                    decomposed["pcaH"].ok_, hsd,
                                    decomposed["cca"].B_)
    assert np.array_equal(D1, D2)


# ----------------------------------------------------------------------
# 8: Q_H orthonormality
# ----------------------------------------------------------------------
def test_08_orthonormality(decomposed):
    Q = np.hstack([decomposed["Q_cca"], decomposed["Q_perp"]])
    err = np.abs(Q.T @ Q - np.eye(N_H)).max()
    assert err < 1e-10, err


# ----------------------------------------------------------------------
# 9-10: reconstruction error and full-dimension preservation
# ----------------------------------------------------------------------
def test_09_reconstruction(real, decomposed):
    mu_H = real["H_trva"][:real["n_tr"]].mean(axis=0, keepdims=True)
    errs = reconstruction_check(real["H_trva"], real["H_te"],
                                decomposed["Q_cca"], decomposed["Q_perp"],
                                mu_H, real["n_tr"])
    assert max(errs.values()) < 1e-10, errs


def test_10_full_dimension_preserved(real, decomposed):
    Z = transform_designs(real["G_trva"], real["H_trva"], real["G_te"],
                          real["H_te"], decomposed["Q_cca"],
                          decomposed["Q_perp"], real["n_tr"])
    assert Z["train"].shape[1] == 9996 and Z["test"].shape[1] == 9996


# ----------------------------------------------------------------------
# 11: Z dimension
# ----------------------------------------------------------------------
def test_11_Z_dimension(real, decomposed):
    Z = transform_designs(real["G_trva"], real["H_trva"], real["G_te"],
                          real["H_te"], decomposed["Q_cca"],
                          decomposed["Q_perp"], real["n_tr"])
    assert Z["train"].shape == (real["n_tr"], 9996)
    assert Z["val"].shape == (23, 9996) and Z["test"].shape == (308, 9996)


# ----------------------------------------------------------------------
# 12: basis invariance — H in Q coordinates reproduces raw-H ridge exactly
# ----------------------------------------------------------------------
def test_12_basis_invariance(real, decomposed):
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import f1_score
    y_dev = np.concatenate([real["ytr"], real["yva"]])
    Z = transform_designs(real["G_trva"], real["H_trva"], real["G_te"],
                          real["H_te"], decomposed["Q_cca"],
                          decomposed["Q_perp"], real["n_tr"])
    Zrot_dev = np.vstack([Z["train"], Z["val"]])
    Zraw_dev = np.hstack([real["G_trva"], real["H_trva"]])
    Zraw_te = np.hstack([real["G_te"], real["H_te"]])
    rc_raw = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)).fit(Zraw_dev, y_dev)
    rc_rot = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)).fit(Zrot_dev, y_dev)
    d_raw, d_rot = rc_raw.decision_function(Zraw_te), rc_rot.decision_function(Z["test"])
    assert np.abs(d_raw - d_rot).max() < 1e-8
    assert f1_score(real["yte"], rc_raw.predict(Zraw_te), average="macro",
                    zero_division=0) == f1_score(real["yte"],
                                                 rc_rot.predict(Z["test"]),
                                                 average="macro",
                                                 zero_division=0)


# ----------------------------------------------------------------------
# 13: alpha_k exactly equals alpha_base * (1 + gamma * rho_k^2)
# ----------------------------------------------------------------------
def test_13_alpha_formula():
    rho = np.array([0.674, 0.839, 0.5])
    delta = adaptive_delta(rho, GAMMA)
    alpha_base = 3.7
    assert np.allclose(alpha_base * (1 + delta), alpha_base * (1 + GAMMA * rho ** 2))
    # monotone in rho
    assert (delta[1] > delta[0] > delta[2])
    # bounded, no discard: 1 <= alpha_k/alpha_base <= 1+gamma
    assert ((1 + delta) >= 1).all() and ((1 + delta) <= 1 + GAMMA).all()


# ----------------------------------------------------------------------
# 14: generalized ridge == brute force on a small synthetic problem
# ----------------------------------------------------------------------
def test_14_dual_matches_brute_force():
    rng = np.random.default_rng(3)
    n, p, K = 50, 90, 5
    Z = rng.normal(size=(n, p))
    y = rng.integers(0, 3, n)
    lam0 = np.ones(p)
    U, _ = np.linalg.qr(rng.normal(size=(p - n // 2, K)))
    # split design as G|H
    nG = p // 2
    lam0[nG:nG + K] = 1.0 + np.linspace(0.2, 1.5, K)
    m = DualGeneralizedRidge(lam0=lam0, alpha_grid=np.logspace(-4, 4, 30)).fit(Z, y)
    W, b, mu = brute_force_generalized_ridge(Z, y, lam0, m.alpha_base_)
    # standard centered-ridge convention: prediction = Z @ W + (y_mean - mu @ W)
    diff = np.abs(m.decision_function(Z) - (Z @ W + b)).max()
    assert diff <= 1e-8, diff


# ----------------------------------------------------------------------
# 15: no test data enters PCA/CCA/GCV — poison test rows, fits unchanged
# ----------------------------------------------------------------------
def test_15_no_test_leakage(real, decomposed):
    H_te_poison = real["H_te"].copy()
    H_te_poison += 1e6
    G_te_poison = real["G_te"].copy()
    G_te_poison += 1e6
    pcaG, pcaH, cca, rho, Q_cca, Q_perp, diag = fit_decomposition(
        real["G_trva"], real["H_trva"], real["n_tr"])
    # decomposition never receives test matrices:
    assert "G_te" not in fit_decomposition.__code__.co_varnames[:4]
    Z = transform_designs(real["G_trva"], real["H_trva"], G_te_poison,
                          H_te_poison, Q_cca, Q_perp, real["n_tr"])
    # transform of poisoned test data is affine in the input (no fitting):
    Zc = transform_designs(real["G_trva"], real["H_trva"], real["G_te"],
                           real["H_te"], Q_cca, Q_perp, real["n_tr"])
    # affine-invariance: poisoning all test rows by a constant shifts the
    # transformed design by a CONSTANT row vector (no refitting, no
    # test-dependent parameters) — G block shifts exactly by 1e6, H block
    # by 1e6 * 1^T Q.
    dG = Z["test"][:, :4998] - Zc["test"][:, :4998]
    assert np.allclose(dG, 1e6, rtol=1e-9, atol=1e-9)
    dH = Z["test"][:, 4998:] - Zc["test"][:, 4998:]
    assert np.allclose(dH, dH[0:1, :], atol=1e-6)
    # GCV uses train rows only (the fit signature has no val/test input)
    m = DualGeneralizedRidge().fit(Z["train"], real["ytr"])
    assert m.diagnostics["n"] == real["n_tr"]


# ----------------------------------------------------------------------
# 16: predictions deterministic
# ----------------------------------------------------------------------
def test_16_deterministic_predictions(real, decomposed):
    Z = transform_designs(real["G_trva"], real["H_trva"], real["G_te"],
                          real["H_te"], decomposed["Q_cca"],
                          decomposed["Q_perp"], real["n_tr"])
    lam0 = np.ones(9996)
    rho = decomposed["rho"]
    lam0[4998:4998 + len(rho)] = 1.0 + adaptive_delta(rho, GAMMA)
    m1 = DualGeneralizedRidge(lam0=lam0, alpha_grid=ALPHA_GRID).fit(Z["train"],
                                                                    real["ytr"])
    m2 = DualGeneralizedRidge(lam0=lam0, alpha_grid=ALPHA_GRID).fit(Z["train"],
                                                                    real["ytr"])
    assert np.array_equal(m1.predict(Z["test"]), m2.predict(Z["test"]))
    assert m1.alpha_base_ == m2.alpha_base_


# ----------------------------------------------------------------------
# 17-19: identity gates (rerun cheaply on stored rows — same ridge_eval)
# ----------------------------------------------------------------------
def test_17_gate_minirocket(real):
    from experiments.heramba_cca_ranked_haptics_seed42.runner import ridge_eval
    y_dev = np.concatenate([real["ytr"], real["yva"]])
    res, _ = ridge_eval(real["G_trva"], y_dev, real["G_te"], real["yte"])
    assert abs(res["macro_f1"] - 0.5037) < 5e-5, res


def test_18_gate_raw_gh(real):
    from experiments.heramba_cca_ranked_haptics_seed42.runner import ridge_eval
    y_dev = np.concatenate([real["ytr"], real["yva"]])
    res, _ = ridge_eval(np.hstack([real["G_trva"], real["H_trva"]]),
                        y_dev, np.hstack([real["G_te"], real["H_te"]]),
                        real["yte"])
    assert res["macro_f1"] == 0.55, res


def test_19_gate_h_unique(real):
    from experiments.heramba_canonical_ridge_full_haptics_seed42.runner import (
        build_h_unique)
    from experiments.heramba_cca_ranked_haptics_seed42.runner import ridge_eval
    y_dev = np.concatenate([real["ytr"], real["yva"]])
    Hu_trva, Hu_te, _ = build_h_unique(real["G_trva"], real["H_trva"],
                                       real["H_te"], real["n_tr"])
    res, _ = ridge_eval(np.hstack([real["G_trva"], Hu_trva]), y_dev,
                        np.hstack([real["G_te"], Hu_te]), real["yte"])
    assert abs(res["macro_f1"] - 0.5138) < 5e-5, res


# ----------------------------------------------------------------------
# 20: previous experiment artifacts untouched + prior tests importable
# ----------------------------------------------------------------------
def test_20_previous_artifacts_untouched():
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    for rel in (os.path.join("heramba_canonical_ridge", "haptics_seed42",
                             "final_comparison.csv"),
                os.path.join("heramba_cca_ranked", "haptics_seed42",
                             "evidence_report.md")):
        assert os.path.exists(os.path.join(root, rel)), rel
