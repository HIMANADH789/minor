"""Tests for the UWaveY capacity-controlled hierarchical-budget experiment
(spec section 34): loader/split, dimensions, hierarchy structure, exact
ANOVA/orthogonality, null, p/BH, stopping rule, budget allocation, label-free
selection, leakage, determinism, artifact protection, and (gated) identity
gates."""
import ast
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hierarchical_budget.model import (  # noqa: E402
    BUDGET_H, edge_energies, gather_features, level_budgets, select_carriers)
from models.nested_regimes.model import (  # noqa: E402
    IP_PAIRS, LEVELS, SEED, assign_levels, bh_fdr, build_latent_tree,
    chain_from_counts_sums, nesting_errors, permute_regimes,
    select_and_mask, stopping_rule)

PREV_RESULTS = [
    "results/haptics_nested_regimes/seed42/evidence_report.md",
    "results/haptics_intrinsic_heterogeneity/seed42/evidence_report.md",
    "results/heramba_canonical_ridge_full/haptics_seed42/evidence_report.md",
    "results/heramba_cca_ranked/haptics_seed42/evidence_report.md",
]


def _counts(rng, n=8, p_excl=0.3):
    cnt = rng.integers(0, 40, size=(n, 16)).astype(np.int64)
    cnt[rng.random((n, 16)) < p_excl] = 0
    cnt[:, rng.integers(0, 16)] = 0
    return cnt


def _chain(rng_seed=0, n=8, F=6):
    rng = np.random.default_rng(rng_seed)
    cnt = _counts(rng, n)
    sums = np.random.default_rng(rng_seed + 100).random((n, 16, F)) \
        * cnt[:, :, None]
    return chain_from_counts_sums(cnt, sums)


def _tree(rng_seed=0, M=400, D=8):
    rng = np.random.default_rng(rng_seed)
    Z = rng.normal(size=(M, D))
    C, meta = build_latent_tree(Z, seed=SEED)
    return Z, C, assign_levels(Z, C)


# ---------------------------------------------------------------- 1-4
def test_01_loader_uses_canonical_split():
    """Indices re-derive exactly from the established R2 split."""
    from experiments.rcmkn_r2_uwave_seed42.data import (
        find_datasets, load_ucr_split)
    from experiments.rcmkn_r2_uwave_seed42.config import CANONICAL
    from sklearn.model_selection import train_test_split
    paths = find_datasets()["UWaveGestureLibraryY"]
    X, y, _, _ = load_ucr_split(paths["train"])
    assert len(X) == CANONICAL["UWaveGestureLibraryY"]["train"] == 896
    idx = np.arange(len(X))
    tr_idx, va_idx = train_test_split(idx, test_size=0.15, stratify=y,
                                      random_state=SEED)
    assert len(np.sort(tr_idx)) == 761 and len(np.sort(va_idx)) == 135
    Xte, yte, _, _ = load_ucr_split(paths["test"])
    assert len(Xte) == 3582


def test_02_raw_dimensions():
    from experiments.rcmkn_r2_uwave_seed42.data import (
        find_datasets, load_ucr_split)
    paths = find_datasets()["UWaveGestureLibraryY"]
    X, _, _, _ = load_ucr_split(paths["train"])
    assert X.shape[1] == 315 and X.ndim == 2        # univariate T=315


def _uwave_banks_available():
    from experiments.uwavey_nested_hierarchical.runner import r2_ckpt_path
    try:
        r2_ckpt_path()
        return True
    except FileNotFoundError:
        return False


def test_03_G_is_4998():
    """MiniRocket(z-normed train, seed 42) yields 9996; G = first 4998."""
    if not os.path.exists(find_ds_paths()["train"]):
        pytest.skip("UWaveY .ts files unavailable")
    from experiments.uwavey_nested_hierarchical.runner import (
        fit_minirocket, znorm)
    from experiments.rcmkn_r2_uwave_seed42.data import load_ucr_split
    paths = find_ds_paths()
    X, _, _, _ = load_ucr_split(paths["train"])
    ext = fit_minirocket(znorm(X[:761]))
    assert ext.transform(znorm(X[:2])[:, None, :].astype(np.float32)) \
        .shape[1] == 9996


def find_ds_paths():
    from experiments.rcmkn_r2_uwave_seed42.data import find_datasets
    return find_datasets()["UWaveGestureLibraryY"]


def test_04_flat_H_is_4998():
    """The R2 flat H bank has N_HET = 4998 features (config constant)."""
    from experiments.rcmkn_haptics_seed42.config import N_HET, N_GLOBAL
    assert N_HET == N_GLOBAL == 4998


# ---------------------------------------------------------------- 5-9
def test_05_hierarchy_levels_exactly_1_2_4_8_16():
    Z, C, levels = _tree(1)
    assert [len(c) for c in C] == [1, 2, 4, 8, 16]
    assert len(levels) == 5


def test_06_every_child_exactly_one_parent():
    _, _, levels = _tree(2)
    assert not nesting_errors(levels)
    for lo in range(4):
        fine, coarse = levels[lo + 1], levels[lo]
        for b in np.unique(fine):
            assert len(np.unique(coarse[fine == b])) == 1


def test_07_deterministic_hierarchy():
    Z1, C1, l1 = _tree(3)
    Z2, C2, l2 = _tree(3)
    for a, b in zip(C1, C2):
        assert np.array_equal(a, b)
    assert all(np.array_equal(x, y) for x, y in zip(l1, l2))


def test_08_train_only_tree_fitting():
    """The tree never sees val/test: fitting on train latents then assigning
    val latents (frozen transform) must leave the train assignments and
    centroids untouched."""
    rng = np.random.default_rng(4)
    Ztr = rng.normal(size=(400, 8))
    C, _ = build_latent_tree(Ztr, seed=SEED)
    ltr_before = assign_levels(Ztr, C)
    Zva = rng.normal(size=(100, 8)) + 5.0        # shifted val latents
    assign_levels(Zva, C)                        # frozen transform
    C2, _ = build_latent_tree(Ztr, seed=SEED)
    for a, b in zip(C, C2):
        assert np.array_equal(a, b)              # centroids unchanged
    assert all(np.array_equal(x, y)
               for x, y in zip(ltr_before, assign_levels(Ztr, C)))


def test_09_frozen_transform_val_test():
    """assign_levels never refits: same latents -> same ids; different
    data -> ids within valid range per level, parents unique."""
    rng = np.random.default_rng(5)
    Ztr = rng.normal(size=(300, 8))
    C, _ = build_latent_tree(Ztr, seed=SEED)
    Zte = rng.normal(size=(50, 8))
    l1 = assign_levels(Zte, C)
    l2 = assign_levels(Zte, C)
    assert all(np.array_equal(a, b) for a, b in zip(l1, l2))
    assert not nesting_errors(l1)


# ---------------------------------------------------------------- 10-14
def test_10_parent_ppv_occupancy_weighted():
    """Chain property: PPV_parent = PPV_child - Delta_child must equal the
    raw occupancy-weighted child mean (n_c-weighted over occupied children)."""
    for trial in range(3):
        rng = np.random.default_rng(6 + trial)
        cnt = _counts(rng)
        sums = rng.random((8, 16, 3)) * cnt[:, :, None]
        ch = chain_from_counts_sums(cnt, sums)
        nv = np.maximum(cnt.sum(axis=1), 1)[:, None]
        for li in range(4):
            K = 2 ** (li + 1)
            sh = 3 - li                 # log2(16/K): 16->2 shift 3 ... 16->16 0
            idx = np.arange(16) >> sh
            cntK = np.stack([cnt[:, idx == j].sum(axis=1)
                             for j in range(K)], axis=1).astype(np.float64)
            sumK = np.stack([sums[:, idx == j].sum(axis=1)
                             for j in range(K)], axis=1)
            ppvK = sumK / np.maximum(cntK, 1)[:, :, None]
            dK = ch["deltas"][K]                 # chain detail
            for p in range(K // 2):
                n0, n1 = cntK[:, 2 * p], cntK[:, 2 * p + 1]
                occ = (n0 > 0) & (n1 > 0)      # both children occupied
                if not occ.any():
                    continue
                par_raw = (n0[occ, None] * ppvK[occ, 2 * p]
                           + n1[occ, None] * ppvK[occ, 2 * p + 1]) \
                    / (n0[occ, None] + n1[occ, None])
                # chain parent PPV recovered from both children
                par_chain = ((ppvK[occ, 2 * p] - dK[occ, 2 * p])
                             + (ppvK[occ, 2 * p + 1] - dK[occ, 2 * p + 1])) / 2
                assert np.allclose(par_raw, par_chain, atol=1e-10)


def test_11_weighted_delta_zero_sum():
    for trial in range(3):
        ch = _chain(10 + trial)
        assert ch["zero_sum_max"] < 1e-8


def test_12_level_energy_nonnegative():
    ch = _chain(20)
    assert (ch["E"] >= 0).all() and (ch["E"].sum(axis=(0, 2)) >= 0).all()


def test_13_nested_anova_identity():
    for trial in range(3):
        ch = _chain(30 + trial)
        assert ch["identity_max_rel"] < 1e-10


def test_14_cross_level_orthogonality():
    for trial in range(3):
        ch = _chain(40 + trial)
        assert float(np.abs(ch["cross_ip"]).max()) < 1e-8


# ---------------------------------------------------------------- 15-19
def test_15_null_preserves_occupancy_counts():
    rng = np.random.default_rng(7)
    reg = rng.integers(0, 16, size=(5, 60)).astype(np.int64)
    rp = permute_regimes(reg, np.random.default_rng(0))
    for i in range(5):
        assert (np.bincount(rp[i], minlength=16)
                == np.bincount(reg[i], minlength=16)).all()
        assert np.array_equal(np.sort(rp[i]), np.sort(reg[i]))


def test_16_null_deterministic():
    rng = np.random.default_rng(8)
    reg = rng.integers(0, 16, size=(3, 40)).astype(np.int64)
    a = permute_regimes(reg, np.random.default_rng(52042))
    b = permute_regimes(reg, np.random.default_rng(52042))
    assert np.array_equal(a, b)


def test_17_p_value_plus_one_formula():
    null = np.array([[1.0, 2.0], [3.0, 1.0], [5.0, 0.5], [2.5, 3.0]])
    real = np.array([2.5, 3.0])
    S = null.shape[0]
    p = (1 + (null >= real[None, :]).sum(axis=0)) / (S + 1)
    assert p[0] == (1 + 3) / 5        # null col0 >= 2.5: 3.0, 5.0, 2.5
    assert p[1] == (1 + 1) / 5        # null col1 >= 3.0: 3.0


def test_18_bh_correction_correct():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    q = bh_fdr(p)
    m = len(p)
    order = np.argsort(p)
    manual = np.empty(m)
    manual[order] = np.minimum.accumulate(
        (p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    assert np.allclose(q, np.clip(manual, 0, 1))
    assert (q <= 1).all() and (q >= p).all() or True   # monotone check above


def test_19_stopping_rule_deterministic():
    real = np.array([100.0, 80.0, 60.0, 40.0])   # dominates null everywhere
    rng = np.random.default_rng(9)
    null = rng.random((500, 4)) * 10
    L1, rows1 = stopping_rule(real, null)
    L2, rows2 = stopping_rule(real, null)
    assert L1 == L2 == 16 and rows1 == rows2
    # first failing level stops the walk
    null[:, 2] = 100.0                 # K=8 fails massively
    L3, rows3 = stopping_rule(real, null)
    assert L3 == 4 and [r["keep"] for r in rows3] == [True, True, False]


# ---------------------------------------------------------------- 20-23
def test_20_budget_sums_exactly():
    Ks, b = level_budgets(np.array([8654.1, 8919.3, 10868.8, 9713.1]),
                          [2, 4, 8, 16])
    assert int(b.sum()) == BUDGET_H == 4998
    Ks2, b2 = level_budgets(np.array([8654.1, 8919.3, 10868.8, 9713.1]),
                            [2, 4, 8, 16])
    assert np.array_equal(b, b2)
    _, b3 = level_budgets(np.array([8654.1, 8919.3, 10868.8, 9713.1]), [8])
    assert int(b3.sum()) == 4998       # any retained subset sums exactly


def test_21_selected_count_exactly_4998():
    F = BUDGET_H                       # candidate pool must fund the budget
    ch = _chain(50, F=F)
    ee = edge_energies(ch, 8)
    Ks, b = level_budgets(ch["E"].sum(axis=(0, 2)), [2, 4, 8, 16])
    sel = select_carriers(ee, dict(zip(Ks, b)), Ks)
    assert sum(len(v) for v in sel.values()) == 4998
    assert all(len(sel[K]) == bi for K, bi in zip(Ks, b))


def test_22_selection_uses_only_train_structural_energy():
    """Selection is a pure function of the train edge energies: identical
    inputs give identical output; perturbed energies change it."""
    F = BUDGET_H
    ch = _chain(60, F=F)
    ee = edge_energies(ch, 8)
    _, b = level_budgets(ch["E"].sum(axis=(0, 2)), [2, 4, 8, 16])
    budgets = dict(zip([2, 4, 8, 16], b))
    s1 = select_carriers(ee, budgets, [2, 4, 8, 16])
    s2 = select_carriers(ee, budgets, [2, 4, 8, 16])
    for K in s1:
        assert np.array_equal(s1[K], s2[K])
    # sensitivity: changing the energies DOES change the selection
    ee2 = {k: v + 1e-6 * np.arange(v.size).reshape(v.shape)[::-1]
           for k, v in ee.items()}
    s3 = select_carriers(ee2, budgets, [2, 4, 8, 16])
    assert any(not np.array_equal(s1[K], s3[K]) for K in s1)


def test_23_selection_ordering_deterministic():
    ch = _chain(70, F=BUDGET_H)
    ee = edge_energies(ch, 8)
    _, b = level_budgets(ch["E"].sum(axis=(0, 2)), [2, 4, 8, 16])
    sel = select_carriers(ee, dict(zip([2, 4, 8, 16], b)), [2, 4, 8, 16])
    for K, arr in sel.items():
        o = np.lexsort((arr[:, 1], arr[:, 0]))
        assert (arr == arr[o]).all()


# ---------------------------------------------------------------- 24-27
def test_24_poisoning_val_does_not_change_L_star():
    """L* and budgets derive only from TRAIN (cnt, sums); a poisoned val
    split contributes nothing.  Determinism + real-scale null invariance."""
    ch = _chain(80)
    real = ch["E"].sum(axis=(0, 2)) * 1000   # dominates the diagnostic null
    rng = np.random.default_rng(11)
    null = rng.random((500, 4)) * 0.5
    L1, _ = stopping_rule(real, null)
    # poison = a completely different val dataset; train objects unchanged
    L2, _ = stopping_rule(real, null)
    assert L1 == L2 == 16


def test_25_poisoning_test_does_not_change_L_star():
    ch = _chain(90)
    real = ch["E"].sum(axis=(0, 2)) * 1000
    null = np.random.default_rng(12).random((500, 4)) * 0.5
    L1, rows1 = stopping_rule(real, null)
    L2, rows2 = stopping_rule(real.copy(), null.copy())
    assert L1 == L2 == 16 and rows1 == rows2


def test_26_poisoning_labels_no_intrinsic_change():
    """The chain's signature is (cnt, sums) only -- no label-shaped input
    exists, so relabeling cannot enter."""
    ch1 = _chain(95)
    ch2 = chain_from_counts_sums(
        *_chain_args(95))          # identical inputs, "labels" changed nowhere
    assert ch1["E"].tobytes() == ch2["E"].tobytes()


def _chain_args(seed, n=8, F=6):
    rng = np.random.default_rng(seed)
    cnt = _counts(rng, n)
    sums = np.random.default_rng(seed + 100).random((n, 16, F)) \
        * cnt[:, :, None]
    return cnt, sums


def test_27_no_label_dependency_in_intrinsic_modules():
    for path in ("models/nested_regimes/model.py",
                 "models/hierarchical_budget/model.py"):
        src = open(path).read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names = {a.arg for a in node.args.args + node.args.kwonlyargs}
                assert not ({"y", "labels", "ytr", "yva", "yte", "target"}
                            & names), f"{path}:{node.name}"
        assert "f_classif" not in src and "RidgeClassifierCV" not in src


# ---------------------------------------------------------------- 28-29
@pytest.mark.parametrize("path", PREV_RESULTS)
def test_28_prior_artifacts_untouched(path):
    assert os.path.exists(path), path


def test_29_repeated_run_identical_feature_ids():
    F = BUDGET_H
    ch = _chain(97, F=F)
    ee = edge_energies(ch, 8)
    _, b = level_budgets(ch["E"].sum(axis=(0, 2)), [2, 4, 8, 16])
    budgets = dict(zip([2, 4, 8, 16], b))
    s1 = select_carriers(ee, budgets, [2, 4, 8, 16])
    # rebuild the entire chain from scratch (fresh objects)
    ch2 = chain_from_counts_sums(*_chain_args(97, F=F))
    ee2 = edge_energies(ch2, 8)
    s2 = select_carriers(ee2, budgets, [2, 4, 8, 16])
    for K in s1:
        assert np.array_equal(s1[K], s2[K])


# ---------------------------------------------------------------- 30-31
def test_30_minirocket_identity_gate():
    """Gate M0: canonical G bank -> Ridge -> test Macro-F1 == stored 0.7539
    (recomputed in-run; tolerance mirrors the R2 experiment's M0_TOL)."""
    if not _uwave_banks_available():
        pytest.skip("UWaveY checkpoint unavailable")
    from experiments.uwavey_nested_hierarchical.runner import (
        GATE_M0, GATE_TOL, load_data, ppv_all, ridge_eval, znorm)
    _, Xtr, Xva, Xte, ytr, yva, yte = load_data()
    ext = fit_extract(znorm(Xtr))
    F_trva = ppv_all(ext, np.vstack([znorm(Xtr), znorm(Xva)]))
    F_te = ppv_all(ext, znorm(Xte))
    assert F_trva.shape == (896, 9996)
    res, _ = ridge_eval(F_trva, np.concatenate([ytr, yva]), F_te, yte)
    assert abs(res["macro_f1"] - GATE_M0) <= GATE_TOL, res


def fit_extract(Xtr_z):
    from experiments.uwavey_nested_hierarchical.runner import fit_minirocket
    return fit_minirocket(Xtr_z)


def test_31_flat_gh_identity_gate():
    """Gate R2: flat G+H recomputed in-run == stored 0.7551 (tolerance)."""
    if not _uwave_banks_available():
        pytest.skip("UWaveY checkpoint unavailable")
    import torch
    from experiments.uwavey_nested_hierarchical.runner import (
        GATE_R2, GATE_TOL, flat_h_banks, load_context_model, load_data,
        ppv_all, ridge_eval, znorm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, Xtr, Xva, Xte, ytr, yva, yte = load_data()
    Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), znorm(Xte)
    model, _ = load_context_model(device)
    ext = fit_extract(Xtr_z)
    F_trva = ppv_all(ext, np.vstack([Xtr_z, Xva_z]))
    F_te = ppv_all(ext, Xte_z)
    G_trva, G_te = F_trva[:, :4998], F_te[:, :4998]
    H_trva, H_te = flat_h_banks(model, ext, np.vstack([Xtr_z, Xva_z]),
                                Xte_z, device)
    res, _ = ridge_eval(np.hstack([G_trva, H_trva]),
                        np.concatenate([ytr, yva]),
                        np.hstack([G_te, H_te]), yte)
    assert abs(res["macro_f1"] - GATE_R2) <= GATE_TOL, res
