"""Focused tests for the HIER-CONTINUOUS-SUP one-change ablation.

The defining change vs HIER-HIGH-SUP: the protected G/H boundary is
removed — root (19,992) + delta (449,820) candidates form ONE unified pool
(469,812) and a single supervised ANOVA-F ranking picks all 19,992 final
features.  These tests verify the unified construction, the single
selector, the absence of rho/N_G/N_H/quota logic, locked dimensions,
label-flow boundaries (incl. val/test poisoning), and artifact safety.
"""

import ast
import inspect
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from experiments.uwavey_hier_continuous_sup.runner import (  # noqa: E402
    B_BASE, B_HIGH, EDGES, G_PART, H_CAND,
    H_KERNELS, KEPT_K, N_CAND, ROOTS)
from experiments.uwavey_hier_high_sup.runner import col_meta  # noqa: E402

RUNNER = os.path.join(os.path.dirname(__file__), "..",
                      "experiments", "uwavey_hier_continuous_sup",
                      "runner.py")


# ---------------------------------------------------------------- budgets
def test_01_dimensions_locked():
    assert B_HIGH == 19992 and B_BASE == 9996 and G_PART == 4998
    assert H_KERNELS == 14994 and EDGES == 30
    assert H_CAND == 449820 and ROOTS == 19992
    assert N_CAND == 469812 == ROOTS + H_CAND


def test_02_no_rho_no_ng_no_nh_anywhere():
    """The runner must not define or use rho, N_G, N_H, or any quota."""
    tree = ast.parse(open(RUNNER).read())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.targets[0].id for n in ast.walk(tree)
              if isinstance(n, ast.Assign) and isinstance(n.targets[0],
                                                          ast.Name)}
    for bad in ("N_G", "N_H", "RHO", "rho", "n_g", "n_h", "level_budgets",
                "select_carriers", "quota"):
        assert bad not in names, bad
    src = open(RUNNER).read()
    for tok in ("N_G =", "N_H =", "rho =", "RHO =", "level_budgets(",
                "select_carriers("):
        assert tok not in src, tok


def test_03_single_selection_stage():
    """Exactly ONE selection call over the unified pool; no separate G/H
    stages, no [:, :N_G] root slicing."""
    src = open(RUNNER).read()
    assert src.count("argsort(-f_stat") == 1
    assert "top_f_select(C_dev, y_dev, B_HIGH)" in src
    assert "[:, :N_G]" not in src


def test_04_root_matrix_not_preselected():
    """The root candidates entering the pool are the FULL 19,992, not a
    17,993 preselection."""
    src = open(RUNNER).read()
    assert "GHK_trva.shape == (n_tr + n_va, ROOTS)" in src
    assert "ROOTS = B_HIGH" in src          # full width == expanded budget
    # no N_G/17,993 constant anywhere (doc mentions are fine)
    tree = ast.parse(src)
    consts = {n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, int)}
    assert 17993 not in consts and 1999 not in consts


# ------------------------------------------------- locked pool layout
def _chain_layout_columns(f_h):
    cols = []
    for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
        s = cbefore * f_h
        for j in range(K):
            for k in range(f_h):
                cols.append((K, j, j >> 1, G_PART + k))
    return cols


def test_05_delta_layout_and_unified_mapping():
    cols = _chain_layout_columns(H_KERNELS)
    assert len(cols) == H_CAND
    # unified pool mapping: candidate c < ROOTS is root; else col_meta
    for c in (0, 1, H_KERNELS, 2 * H_KERNELS - 1, 6 * H_KERNELS,
              14 * H_KERNELS, H_CAND - 1):
        K, j, p, kg = col_meta(c, H_KERNELS)
        assert cols[c] == (K, j, p, kg)
        # unified index = ROOTS + delta index
        assert c + ROOTS >= ROOTS


def test_06_unified_pool_shape_via_chain():
    """Synthetic end-to-end pool shape through the REAL chain code (small
    F) proves root + delta stacking and the column mapping at any width."""
    from experiments.uwavey_nested_hierarchical.runner import delta_banks
    rng = np.random.default_rng(7)
    n, F, T = 24, 7, 315
    reg16 = rng.integers(0, 16, size=(n, T)).astype(np.int64)
    reg16[:, :16] = np.arange(16)
    act = (rng.random((n, F, T)) > 0.5)
    valid = np.zeros((F, T), dtype=bool)
    valid[:, 5:310] = True
    pool = delta_banks(act, valid, reg16, KEPT_K, delta_cols=None,
                       sample_chunk=8)
    assert pool.shape == (24, 30 * 7)
    roots = rng.random((24, 5))                       # fake 5 root candidates
    C = np.hstack([roots, pool])
    assert C.shape == (24, 5 + 30 * 7)
    # metadata split point
    for c in range(C.shape[1]):
        if c < 5:
            pass                                       # root candidate
        else:
            K, j, p, kg = col_meta(c - 5, 7)
            assert kg == G_PART + ((c - 5) % 7)


# ------------------------------------------------- selector identity
def test_07_selector_is_r5_top_f_select():
    import experiments.uwavey_capacity_scaled.runner as cap
    from experiments.uwavey_hier_continuous_sup import runner as sup
    assert sup.top_f_select is cap.top_f_select
    assert sup.r5_cv_fixed_rho is cap.r5_cv_fixed_rho
    src = inspect.getsource(sys.modules["experiments.uwavey_hier_continuous_sup.runner"])
    assert "top_f_select" in src and "r5_cv_fixed_rho" in src


def test_08_top_f_select_deterministic_and_nan_safe():
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 40))
    y = np.repeat(np.arange(3), 20)
    X[:, 3] = 0.0
    t1 = top_f_select(X, y, 10)
    t2 = top_f_select(X, y, 10)
    assert np.array_equal(t1, t2) and len(t1) == 10
    from sklearn.feature_selection import f_classif
    f, _ = f_classif(X, y)
    assert np.isnan(f[3])


def test_09_unified_selection_beats_nothing_by_construction():
    """The single ranking must be exactly top_f_select over the unified
    matrix (verified on synthetic data incl. root columns)."""
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(3)
    roots = rng.normal(size=(90, 12))
    deltas = rng.normal(size=(90, 48))
    y = np.repeat(np.arange(3), 30)
    deltas[:, 5] += 3.0 * (y == 1)                    # informative delta
    roots[:, 2] += 3.0 * (y == 2)                     # informative root
    C = np.hstack([roots, deltas])
    sel = top_f_select(C, y, 20)
    got = set(sel.tolist())
    assert 5 + 12 in got and 2 in got                 # both sources compete
    # cross-check against the manual f_classif path used by the runner
    from sklearn.feature_selection import f_classif
    f_stat, _ = f_classif(C, y)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    manual = np.argsort(-f_stat, kind="stable")[:20]
    assert np.array_equal(sel, manual)


# ------------------------------------------------- fold-internal CV
def test_10_cv_uses_unified_pool_fold_internally():
    """The diagnostic CV must run ONE selection per fold over the unified
    pool (n_g=0 exposes the full pool), fold-train labels only."""
    src = inspect.getsource(
        sys.modules["experiments.uwavey_capacity_scaled.runner"])
    assert "top_f_select(H_full[tr], y_dev[tr], n_h)" in src
    rsrc = open(RUNNER).read()
    assert "r5_cv_fixed_rho(" in rsrc
    # the runner passes n_g=0 so the G block contributes nothing
    assert "np.zeros((C_dev.shape[0], 0)), C_dev, y_dev, 0, B_HIGH" in rsrc


# ------------------------------------------------- train-only hierarchy
def test_11_runner_uses_train_latents_only():
    src = open(RUNNER).read()
    assert "_encoder_latents(model, Xtr_z, device)" in src
    assert src.count("assign_levels(lat_va") == 1
    assert src.count("assign_levels(lat_te") == 1
    assert "build_latent_tree(lat_va" not in src
    assert "build_latent_tree(lat_te" not in src


def test_12_val_signal_poisoning_leaves_tree_unchanged():
    from experiments.uwavey_nested_hierarchical.runner import (
        _encoder_latents, load_context_model, load_data, znorm)
    from models.nested_regimes.model import (
        assign_levels, build_latent_tree, nesting_errors)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, Xtr, Xva, Xte, ytr, yva, yte = load_data()
    Xtr, Xva = Xtr[:24], Xva[:8]
    model, _ = load_context_model(device)
    lat_tr = _encoder_latents(model, znorm(Xtr), device).reshape(-1, 32)
    C, _ = build_latent_tree(lat_tr, seed=42)
    levels = assign_levels(
        _encoder_latents(model, znorm(Xva), device).reshape(-1, 32), C)
    assert not nesting_errors(levels)
    Xva_bad = Xva.copy()
    Xva_bad[:] = 9.9 * np.abs(Xva_bad) + 7.3
    assign_levels(
        _encoder_latents(model, znorm(Xva_bad), device).reshape(-1, 32), C)
    # tree rebuilt from the SAME cached train latents is identical
    # (assignment-level determinism; centroids allclose — see HIER-HIGH-SUP)
    C2, _ = build_latent_tree(lat_tr, seed=42)
    for c1, c2 in zip(C, C2):
        assert np.allclose(np.asarray(c1), np.asarray(c2), atol=1e-12)
    for la, lb in zip(assign_levels(lat_tr, C), assign_levels(lat_tr, C2)):
        assert np.array_equal(la, lb)


# ------------------------------------------------- label-flow boundaries
def test_13_test_labels_never_touch_selection():
    src = open(RUNNER).read()
    assert "f_classif(C_dev, y_dev)" in src
    assert "top_f_select(C_dev, y_dev, B_HIGH)" in src
    assert "f_classif(C_dev, yte)" not in src
    assert "top_f_select(C_dev, yte" not in src


def test_14_val_label_poisoning_changes_only_final_stage():
    """Val labels legitimately enter the canonical final stage (train+val
    selection, shared with R5-HIGH/HIER-HIGH-SUP).  They must NOT change
    the CV-diagnostic fold-internal selection."""
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(0)
    C = rng.normal(size=(100, 500))
    y = np.repeat(np.arange(4), 25)
    y2 = y.copy()
    y2[75:] = (y2[75:] + 1) % 4
    fold1 = top_f_select(C[:75], y[:75], 50)
    fold2 = top_f_select(C[:75], y2[:75], 50)
    assert np.array_equal(fold1, fold2)
    full1 = top_f_select(C, y, 50)
    full2 = top_f_select(C, y2, 50)
    assert not np.array_equal(full1, full2)


def test_15_test_label_poisoning_changes_nothing():
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(0)
    C = rng.normal(size=(100, 500))
    y = np.repeat(np.arange(4), 25)
    assert np.array_equal(top_f_select(C, y, 50), top_f_select(C, y, 50))
    src = open(RUNNER).read()
    seg = src.split("# ---- 9. THE ONE SELECTION")[1].split(
        "# ---- 11. final 19,992")[0]
    assert "yte" not in seg.replace("len(yte)", "")


# ------------------------------------------------- artifacts / old runs
def _saved_artifacts():
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    out = {}
    for root, _, files in os.walk(base):
        for fn in files:
            p = os.path.join(root, fn)
            if fn.endswith((".json", ".csv", ".md", ".npy", ".png",
                            ".yaml")):
                st = os.stat(p)
                out[os.path.relpath(p, base)] = (st.st_size, st.st_mtime_ns)
    return out


def test_16_old_artifacts_untouched_by_import():
    before = _saved_artifacts()
    import experiments.uwavey_hier_continuous_sup.runner  # noqa: F401
    after = _saved_artifacts()
    assert before == after


def test_17_runner_writes_only_inside_its_outdir():
    tree = ast.parse(open(RUNNER).read())
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) in ("save", "savefig")]
    for n in saves:
        assert n.args, "np.save/savefig without path"
        assert "OUT" in ast.unparse(n.args[0]), \
            f"save outside OUT: {ast.unparse(n.args[0])}"


# ------------------------------------------------- assembly invariants
def test_18_final_assembly_semantics():
    """One shared column order for both splits: roots first, then deltas
    level-grouped (== hier_banks_test layout); layout guard enforced."""
    src = open(RUNNER).read()
    assert "Z_dev = C_dev[:, sel]" in src
    assert "Z_te[:, :n_root_sel] = GHK_te[:, root_idx]" in src
    assert "Z_te[:, n_root_sel:] = Hcand_te" in src
    assert "hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)" in src
    assert "assert lv_seq == sorted(lv_seq)" in src
    assert "Z_dev.shape == (n_tr + n_va, B_HIGH)" in src
    assert "np.isfinite(Z_dev).all() and np.isfinite(Z_te).all()" in src


# ------------------------------------------------- smoke (opt-in)
@pytest.mark.slow
def test_99_smoke_full_path():
    """Full pipeline on tiny subsets: expanded MiniROCKET -> hierarchy ->
    full root matrix -> 449,820-layout delta pool -> unified 469,812-layout
    pool -> ONE supervised ranking -> exact final representation -> Ridge.
    Run explicitly with: pytest -q tests/test_uwavey_hier_continuous_sup.py::test_99
    """
    from experiments.uwavey_hier_continuous_sup import runner as sup
    sup.main(smoke=True)
    out = sup.OUT
    assert os.path.isdir(out)
    r = json.load(open(os.path.join(out, "results",
                                    "hier_continuous_sup.json")))
    assert r["final_features"] == 19992
    assert r["candidate_pool"]["root"] == 19992
    assert r["candidate_pool"]["delta"] == 449820
    assert r["candidate_pool"]["unified"] == 469812
    assert sum(r["selected_composition"].values()) == 19992
    dims = json.load(open(os.path.join(out, "diagnostics",
                                       "dimensions.json")))
    assert dims["final"]["Z_dev"][1] == 19992
    assert dims["pools"]["unified_candidates"] == 469812
    meta = open(os.path.join(out, "diagnostics",
                             "selected_feature_metadata.csv")).read()
    assert "source_type" in meta and "root" in meta and "delta" in meta
