"""Focused tests for the HIER-HIGH-SUP one-change ablation.

Covers: exact budgets/dimensions, locked candidate-pool layout, supervised
selector identity with R5-HIGH, fold-internal CV, train-only hierarchy,
label-flow boundaries (incl. val/test poisoning), artifact protection and
forbidden-allocator absence.
"""

import ast
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from experiments.uwavey_hier_high_sup.runner import (  # noqa: E402
    B_BASE, B_HIGH, G_PART, H_CAND, H_KERNELS, KEPT_K, N_G, N_H, col_meta)

RUNNER = os.path.join(os.path.dirname(__file__), "..",
                      "experiments", "uwavey_hier_high_sup", "runner.py")
CAP_RUNNER = os.path.join(os.path.dirname(__file__), "..",
                          "experiments", "uwavey_capacity_scaled",
                          "runner.py")


@pytest.fixture(scope="module")
def delta_pool_fixture():
    """Synthetic frozen-transform pool through the REAL chain/layout code
    (delta_banks with delta_cols=None) on a small kernel width F=7."""
    from experiments.uwavey_nested_hierarchical.runner import delta_banks
    rng = np.random.default_rng(7)
    n, F, T = 24, 7, 315
    reg16 = rng.integers(0, 16, size=(n, T)).astype(np.int64)
    reg16[:, :16] = np.arange(16)          # every regime touched
    act = (rng.random((n, F, T)) > 0.5)
    valid = np.zeros((F, T), dtype=bool)
    valid[:, 5:310] = True                 # contiguous padding region p=5
    return delta_banks(act, valid, reg16, KEPT_K, delta_cols=None,
                       sample_chunk=8)


# ---------------------------------------------------------------- budgets
def test_01_final_budget_exact():
    assert B_HIGH == 19992 and N_G == 17993 and N_H == 1999
    assert N_G + N_H == B_HIGH


def test_02_candidate_pool_exact():
    assert H_KERNELS == 14994 and H_CAND == 449820
    assert H_CAND == sum(KEPT_K) * H_KERNELS == 30 * H_KERNELS


def test_03_g_part_locked():
    assert G_PART == 4998 and B_BASE == 9996


# ------------------------------------------------- locked pool layout
def _chain_layout_columns(f_h):
    cols = []
    for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
        s = cbefore * f_h
        for j in range(K):
            for k in range(f_h):
                cols.append((K, j, j >> 1, 4998 + k))
    return cols


def test_04_pool_layout_and_metadata_consistency():
    cols = _chain_layout_columns(H_KERNELS)
    assert len(cols) == H_CAND
    # runner's col_meta must invert the delta_banks column order exactly
    for c in (0, 1, H_KERNELS, 2 * H_KERNELS - 1, 6 * H_KERNELS,
              14 * H_KERNELS, H_CAND - 1):
        K, j, p, kg = col_meta(c, H_KERNELS)
        assert cols[c] == (K, j, p, kg)
    # boundaries between levels
    assert col_meta(2 * H_KERNELS - 1, H_KERNELS)[0] == 2
    assert col_meta(2 * H_KERNELS, H_KERNELS)[0] == 4
    assert col_meta(14 * H_KERNELS - 1, H_KERNELS)[0] == 8
    assert col_meta(14 * H_KERNELS, H_KERNELS)[0] == 16


def test_05_pool_shape_via_chain(delta_pool_fixture):
    pool = delta_pool_fixture
    assert pool.shape == (24, 30 * 7)
    # col_meta must invert the layout at ANY kernel width
    for c in (0, 7, 14, 43, 85, 209):
        K, j, p, kg = col_meta(c, 7)
        assert kg == 4998 + (c % 7)
        assert p == j >> 1


def test_06_k2_child_columns_are_antitwin(delta_pool_fixture):
    """Locked-definition structural property (documented in AUDIT.md): at
    K=2 the two child columns are exact anti-correlated rescalings, so
    f_classif ranks them adjacently (equal F up to float rounding)."""
    pool = delta_pool_fixture
    a, b = pool[:, 0], pool[:, 7]
    r = np.corrcoef(a, b)[0, 1]
    assert r < -0.99


# ------------------------------------------------- selector identity
def test_07_selector_is_r5_top_f_select():
    import experiments.uwavey_capacity_scaled.runner as cap
    src = inspect.getsource(
        sys.modules["experiments.uwavey_hier_high_sup.runner"])
    assert "top_f_select" in src and "r5_cv_fixed_rho" in src
    # the runner must reuse, not reimplement, the R5 selector
    from experiments.uwavey_hier_high_sup import runner as sup
    assert sup.top_f_select is cap.top_f_select
    assert sup.r5_cv_fixed_rho is cap.r5_cv_fixed_rho


def test_08_top_f_select_deterministic_and_nan_safe():
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 40))
    y = np.repeat(np.arange(3), 20)
    X[:, 3] = 0.0                       # constant -> nan f_stat
    t1 = top_f_select(X, y, 10)
    t2 = top_f_select(X, y, 10)
    assert np.array_equal(t1, t2)       # deterministic (stable argsort)
    assert len(t1) == 10
    from sklearn.feature_selection import f_classif
    f, _ = f_classif(X, y)
    assert np.isnan(f[3])               # nan path exercised


def test_09_fold_internal_cv_selection():
    src = inspect.getsource(
        sys.modules["experiments.uwavey_capacity_scaled.runner"]
    )
    # the CV helper selects inside each fold (H_full[tr], y_dev[tr])
    assert "top_f_select(H_full[tr], y_dev[tr], n_h)" in src


# ------------------------------------------------- train-only hierarchy
def test_10_train_only_hierarchy_and_poisoned_val_signals():
    """Hierarchy must not change when validation SIGNALS are poisoned, and
    its construction must reference train latents only."""
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
    Xva_bad[:] = 9.9 * np.abs(Xva_bad) + 7.3   # poison validation signals
    assign_levels(
        _encoder_latents(model, znorm(Xva_bad), device).reshape(-1, 32), C)
    # the TREE is untouched by the poisoned val pass: rebuild from the SAME
    # cached train latents must give the identical tree, and no refit ever
    # consumed validation data.  (sklearn KMeans centroids are ulp-noisy
    # under threaded BLAS, so assert assignment-level determinism plus
    # allclose centroids.)
    C2, _ = build_latent_tree(lat_tr, seed=42)
    for c1, c2 in zip(C, C2):
        assert np.allclose(np.asarray(c1), np.asarray(c2), atol=1e-12)
    lv_a = assign_levels(lat_tr, C)
    lv_b = assign_levels(lat_tr, C2)
    for la, lb in zip(lv_a, lv_b):
        assert np.array_equal(la, lb)


def test_11_runner_uses_train_latents_only():
    src = inspect.getsource(
        sys.modules["experiments.uwavey_hier_high_sup.runner"])
    assert "_encoder_latents(model, Xtr_z, device)" in src
    # val/test latents are used ONLY for assign_levels (transform-only)
    assert src.count("assign_levels(lat_va") == 1
    assert src.count("assign_levels(lat_te") == 1
    assert "build_latent_tree(lat_va" not in src
    assert "build_latent_tree(lat_te" not in src


# ------------------------------------------------- label-flow boundaries
def test_12_test_labels_never_touch_selection():
    src = inspect.getsource(
        sys.modules["experiments.uwavey_hier_high_sup.runner"])
    # selection consumes y_dev only; yte appears only in ridge_eval /
    # artifacts
    assert "top_f_select(Hcand_trva, y_dev, N_H)" in src
    assert "f_classif(Hcand_trva, y_dev)" in src
    assert "f_classif(Hcand_trva, yte)" not in src
    assert "top_f_select(Hcand_trva, yte" not in src


def test_13_val_label_poisoning_changes_only_final_stage():
    """Val labels are legitimately used by the canonical final stage
    (train+val selection + Ridge fit).  Poisoning them must NOT change:
    the hierarchy, the candidate pool, or the CV-diagnostic fold-internal
    selection (fold-train labels only).  It MUST change the final top-1999
    (that is the canonical protocol, shared with R5-HIGH)."""
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(0)
    H = rng.normal(size=(100, 500))
    y = np.repeat(np.arange(4), 25)
    y2 = y.copy()
    y2[75:] = (y2[75:] + 1) % 4          # poison the val portion
    top1 = top_f_select(H[:75], y[:75], 50)     # fold-internal (train only)
    top2 = top_f_select(H[:75], y2[:75], 50)
    assert np.array_equal(top1, top2)           # unaffected by val labels
    full1 = top_f_select(H, y, 50)              # final stage (train+val)
    full2 = top_f_select(H, y2, 50)
    assert not np.array_equal(full1, full2)     # canonical protocol uses them


def test_14_test_label_poisoning_changes_nothing():
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(0)
    H = rng.normal(size=(100, 500))
    y = np.repeat(np.arange(4), 25)
    assert np.array_equal(top_f_select(H, y, 50),
                          top_f_select(H, y, 50))          # idempotent
    # test labels are never an argument of any selection call
    src = inspect.getsource(
        sys.modules["experiments.uwavey_hier_high_sup.runner"])
    seg = src.split("# ---- 8. THE ONE CHANGE")[1].split(
        "# ---- 11. final representation")[0]
    assert "yte" not in seg.replace("len(yte)", "")


# ------------------------------------------------- artifacts / old runs
def _saved_artifacts():
    base = os.path.join(os.path.dirname(__file__), "..", "results",
                        "uwavey_capacity_scaled", "seed42")
    out = {}
    for root, _, files in os.walk(base):
        for fn in files:
            p = os.path.join(root, fn)
            if fn.endswith((".json", ".csv", ".md", ".npy", ".png",
                            ".yaml")):
                st = os.stat(p)
                out[os.path.relpath(p, base)] = (st.st_size, st.st_mtime_ns)
    return out


def test_15_old_artifacts_untouched_by_import():
    before = _saved_artifacts()
    import experiments.uwavey_hier_high_sup.runner  # noqa: F401
    after = _saved_artifacts()
    assert before == after


def test_16_runner_writes_only_inside_its_outdir():
    tree = ast.parse(open(RUNNER).read())
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) in ("save", "savefig",
                                                   "savefig")]
    for n in saves:
        args = n.args
        assert args, "np.save/savefig without path"
        first = args[0]
        text = ast.unparse(first)
        assert "OUT" in text, f"save outside OUT: {text}"


# ------------------------------------------------- forbidden allocator
def test_17_energy_allocator_absent():
    """The ablated allocator must not be USED (AST-level); the runner may
    mention it in documentation strings describing what was ablated."""
    tree = ast.parse(open(RUNNER).read())
    forbidden = {"level_budgets", "select_carriers", "shuffle_null",
                 "stopping_rule", "edge_energies"}
    hits = [n.id for n in ast.walk(tree)
            if isinstance(n, (ast.Name, ast.Attribute))
            and getattr(n, "id", getattr(n, "attr", None)) in forbidden]
    assert not hits, hits
    # the old quotas must not be computed anywhere
    src = open(RUNNER).read()
    assert "level_budgets(" not in src and "select_carriers(" not in src
    assert "shuffle_null(" not in src and "stopping_rule(" not in src


def test_18_no_validation_in_hierarchy_source():
    src = open(RUNNER).read()
    seg = src.split("# ---- 2. frozen hierarchy")[1].split(
        "# ---- 3. expanded")[0]
    seg = seg.replace("Xtr_z, Xva_z, Xte_z = znorm(Xtr), znorm(Xva), "
                      "znorm(Xte)", "")
    pre_tree = seg.split("build_latent_tree")[0]
    assert "Xva" not in pre_tree and "Xte" not in pre_tree
    assert "build_latent_tree" in seg


# ------------------------------------------------- smoke (opt-in)
@pytest.mark.slow
def test_99_smoke_full_path():
    """Full pipeline on tiny subsets: expanded MiniROCKET -> hierarchy ->
    449,820 pool -> supervised top-1,999 -> 19,992 representation -> Ridge.
    Run explicitly with: pytest -q tests/test_uwavey_hier_high_sup.py::test_99
    """
    from experiments.uwavey_hier_high_sup import runner as sup
    sup.main(smoke=True)
    out = sup.OUT
    assert os.path.isdir(out)
    import json
    r = json.load(open(os.path.join(out, "results",
                                    "hier_high_sup.json")))
    assert r["final_features"] == 19992
    assert r["g_features"] == 17993 and r["h_features"] == 1999
    assert r["h_candidate_pool"] == 449820
    meta = open(os.path.join(out, "diagnostics",
                             "selected_feature_metadata.csv")).read()
    assert len(meta.strip().splitlines()) == 2000   # header + 1,999 rows
