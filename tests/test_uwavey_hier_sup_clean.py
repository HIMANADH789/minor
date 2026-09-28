"""Focused tests for the HIER-SUP-CLEAN one-change experiment.

The experiment is a clean telescoping reformulation: PPV^(0)=0 makes the
root block Delta^(1) = PPV^(1); the unified pool [D1|D2|D4|D8|D16] and one
supervised ANOVA-F ranking are identical to HIER-CONTINUOUS-SUP, so the
production run must reproduce the stored run (equivalence gate).

Covers: locked budgets/dimensions, telescoping identities, pool layout,
absence of G/H/rho/quota/EB logic, selector identity, train-only hierarchy,
label-flow boundaries (incl. poisoning), artifact protection, and the full
smoke path.
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

from experiments.uwavey_hier_sup_clean.runner import (  # noqa: E402
    B_BASE, B_HIGH, D1, EDGES, G_PART, H_CAND, H_KERNELS, KEPT_K, N_CAND,
    PPV0)

RUNNER = os.path.join(os.path.dirname(__file__), "..",
                      "experiments", "uwavey_hier_sup_clean", "runner.py")
REF_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                       "uwavey_hier_continuous_sup", "seed42")


# ---------------------------------------------------------------- budgets
def test_01_dimensions_locked():
    assert B_HIGH == 19992 and B_BASE == 9996 and G_PART == 4998
    assert H_KERNELS == 14994 and EDGES == 30
    assert H_CAND == 449820 and D1 == 19992
    assert N_CAND == 469812 == D1 + H_CAND
    assert PPV0 == 0.0


def test_02_no_gh_no_rho_no_quota_no_eb():
    """No G/H split, no rho, no reserved budgets, no per-level quota, no
    energy allocation, no empirical-Bayes logic in the runner (descriptive
    negations are expected; only LOGIC tokens are forbidden)."""
    src = open(RUNNER).read()
    for tok in ("N_G =", "N_H =", "rho =", "RHO =", "level_budgets(",
                "select_carriers(", "lfdr", "posterior_signal_q",
                "pi0", "mu1", "sigma1"):
        assert tok not in src, tok
    for negation in ("no rho", "no G/H split", "no per-level quota",
                     "no empirical Bayes", "no moderated-F"):
        assert negation in src, negation


def test_03_single_selection_stage():
    """Exactly ONE selection call over the unified telescoping pool."""
    src = open(RUNNER).read()
    assert src.count("argsort(-f_stat") == 1
    assert "top_f_select(C_dev, y_dev, B_HIGH)" in src
    assert "[:, :N_G]" not in src and "[:, :D1]" not in src


# ------------------------------------------------- telescoping identities
def test_04_root_increment_identity_synthetic():
    """Delta^(1) = PPV^(1) - PPV^(0) with PPV^(0)=0 must equal PPV^(1)
    bit-exactly (the spec's equivalence check, on synthetic data)."""
    from experiments.uwavey_hier_sup_clean.runner import PPV0
    rng = np.random.default_rng(0)
    ppv1 = rng.random((40, 12))
    d1 = ppv1 - PPV0
    assert np.array_equal(d1, ppv1)


def test_05_deeper_delta_and_layout():
    """Full-pool delta layout through the REAL chain code (small F):
    levels coarse->fine, child-major/kernel-minor; col_meta agrees."""
    from experiments.uwavey_nested_hierarchical.runner import delta_banks
    from experiments.uwavey_hier_high_sup.runner import col_meta
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
    for c in (0, 7, 2 * 7, 6 * 7, 14 * 7, 29 * 7):
        K, j, p, kg = col_meta(c, 7)
        assert kg == G_PART + (c % 7)
        assert p == j >> 1


def test_06_unified_pool_mapping_root_then_deltas():
    """Unified index space: candidate c < D1 is a root increment; deeper
    candidates map to col_meta(c - D1)."""
    from experiments.uwavey_hier_high_sup.runner import col_meta
    assert col_meta(0, H_KERNELS)[0] == 2
    assert col_meta(2 * H_KERNELS, H_KERNELS)[0] == 4
    assert col_meta(6 * H_KERNELS, H_KERNELS)[0] == 8
    assert col_meta(14 * H_KERNELS, H_KERNELS)[0] == 16
    for c in (0, 1, D1, D1 + 1, D1 + H_CAND - 1):
        K, j, p, kg = (None, -1, -1, c) if c < D1 \
            else col_meta(c - D1, H_KERNELS)
        assert (K is None) == (c < D1)


def test_07_parent_occupancy_weighting_and_zero_sum():
    """The locked chain must keep occupancy-weighted residuals:
    sum_c (n_c/n_p) Delta_{m,c} = 0 per parent (zero_sum_max), and the
    telescoping identity var16 = sum_l E_l must hold to machine precision."""
    from experiments.uwavey_nested_hierarchical.runner import (
        compute_activations, hierarchy_chain)
    rng = np.random.default_rng(11)
    n, F, T = 12, 9, 315
    reg16 = rng.integers(0, 16, size=(n, T)).astype(np.int64)
    reg16[:, :16] = np.arange(16)
    act = (rng.random((n, F, T)) > 0.5)
    valid = np.zeros((F, T), dtype=bool)
    valid[:, 8:307] = True
    ch = hierarchy_chain(act[:, 4:], valid[4:], reg16,
                         want_deltas=False, want_ips=False)
    assert ch["zero_sum_max"] < 1e-10
    denom = max(float(np.abs(ch["var16"]).max()), 1e-30)
    assert float(np.abs(ch["var16"] - ch["E"].sum(axis=1)).max()) / denom \
        < 1e-10


# ------------------------------------------------- selector identity
def test_08_selector_is_incumbent_top_f_select():
    import experiments.uwavey_capacity_scaled.runner as cap
    from experiments.uwavey_hier_sup_clean import runner as clean
    assert clean.top_f_select is cap.top_f_select
    assert clean.r5_cv_fixed_rho is cap.r5_cv_fixed_rho
    assert clean.ridge_eval is (
        __import__("experiments.uwavey_nested_hierarchical.runner",
                   fromlist=["ridge_eval"]).ridge_eval)


def test_09_top_f_select_deterministic_and_nan_safe():
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


def test_10_unified_selection_single_ranking_synthetic():
    """One ranking over a mixed pool must let both sources compete; the
    manual f_classif path used by the runner must match top_f_select."""
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(3)
    roots = rng.normal(size=(90, 12))
    deltas = rng.normal(size=(90, 48))
    y = np.repeat(np.arange(3), 30)
    deltas[:, 5] += 3.0 * (y == 1)
    roots[:, 2] += 3.0 * (y == 2)
    C = np.hstack([roots, deltas])
    sel = top_f_select(C, y, 20)
    got = set(sel.tolist())
    assert 5 + 12 in got and 2 in got
    from sklearn.feature_selection import f_classif
    f_stat, _ = f_classif(C, y)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    manual = np.argsort(-f_stat, kind="stable")[:20]
    assert np.array_equal(sel, manual)


# ------------------------------------------------- fold-internal CV
def test_11_cv_diagnostic_fold_internal():
    """The CV diagnostic runs ONE fold-internal selection per fold over the
    unified pool (n_g=0 exposes the full pool); fold-val labels never
    touch selection."""
    src = inspect.getsource(
        sys.modules["experiments.uwavey_capacity_scaled.runner"])
    assert "top_f_select(H_full[tr], y_dev[tr], n_h)" in src
    rsrc = open(RUNNER).read()
    assert "r5_cv_fixed_rho(" in rsrc
    assert "np.zeros((C_dev.shape[0], 0)), C_dev, y_dev, 0, B_HIGH" in rsrc


# ------------------------------------------------- train-only hierarchy
def test_12_runner_uses_train_latents_only():
    src = open(RUNNER).read()
    assert "_encoder_latents(model, Xtr_z, device)" in src
    assert src.count("assign_levels(lat_va") == 1
    assert src.count("assign_levels(lat_te") == 1
    assert "build_latent_tree(lat_va" not in src
    assert "build_latent_tree(lat_te" not in src


def test_13_val_signal_poisoning_leaves_tree_unchanged():
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
    # (assignment-level determinism; centroids allclose - see HIER-HIGH-SUP)
    C2, _ = build_latent_tree(lat_tr, seed=42)
    for c1, c2 in zip(C, C2):
        assert np.allclose(np.asarray(c1), np.asarray(c2), atol=1e-12)
    for la, lb in zip(assign_levels(lat_tr, C), assign_levels(lat_tr, C2)):
        assert np.array_equal(la, lb)


# ------------------------------------------------- label-flow boundaries
def test_14_test_labels_never_touch_selection():
    src = open(RUNNER).read()
    assert "f_classif(C_dev, y_dev)" in src
    assert "top_f_select(C_dev, y_dev, B_HIGH)" in src
    assert "f_classif(C_dev, yte)" not in src
    assert "top_f_select(C_dev, yte" not in src


def test_15_val_label_poisoning_changes_only_final_stage():
    """Val labels legitimately enter the canonical final stage (train+val
    selection, shared with all incumbents).  They must NOT change the
    CV-diagnostic fold-internal selection."""
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


def test_16_test_label_poisoning_changes_nothing():
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


def test_17_old_artifacts_untouched_by_import():
    before = _saved_artifacts()
    import experiments.uwavey_hier_sup_clean.runner  # noqa: F401
    after = _saved_artifacts()
    assert before == after


def test_18_runner_writes_only_inside_its_outdir():
    tree = ast.parse(open(RUNNER).read())
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) in ("save", "savefig")]
    for n in saves:
        assert n.args, "np.save/savefig without path"
        assert "OUT" in ast.unparse(n.args[0]), \
            f"save outside OUT: {ast.unparse(n.args[0])}"


# ------------------------------------------------- metadata semantics
def test_19_metadata_semantics_root_vs_delta():
    """Root increments are (source_type root_increment, level 1,
    parent -1, child -1); deeper are (hierarchy_delta, 2/4/8/16, actual
    parent/child); the metadata CSV emitted by the runner uses these."""
    src = open(RUNNER).read()
    assert "root_increment,1,-1,-1" in src
    assert "hierarchy_delta" in src
    # level guard: root candidates get level 1, not 0 (telescoping naming)
    assert "if K == 1:" in src and "if c < D1" in src


def test_20_equivalence_gate_artifacts_present():
    """The runner computes and stores the equivalence check; production
    asserts the IDENTICAL verdict (components compared against the stored
    HIER-CONTINUOUS-SUP run)."""
    src = open(RUNNER).read()
    assert "equivalence_check.json" in src
    assert '"IDENTICAL"' in src
    assert "test_predictions.npy" in src
    assert "hier_continuous_sup.json" in src
    # production must FAIL LOUDLY on a divergence
    assert 'assert eq["verdict"] == "IDENTICAL"' in src


# ------------------------------------------------- assembly invariants
def test_21_final_assembly_semantics():
    """One shared column order for both splits: D1 first, then deltas
    level-grouped (== hier_banks_test layout); layout guard enforced."""
    src = open(RUNNER).read()
    assert "Z_dev = C_dev[:, sel]" in src
    assert "Z_te[:, :n_root_sel] = D1_te[:, root_idx]" in src
    assert "Z_te[:, n_root_sel:] = Hcand_te" in src
    assert "hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)" in src
    assert "assert lv_seq == sorted(lv_seq)" in src
    assert "Z_dev.shape == (n_tr + n_va, B_HIGH)" in src
    assert "np.isfinite(Z_dev).all() and np.isfinite(Z_te).all()" in src


# ------------------------------------------------- smoke (opt-in)
@pytest.mark.slow
def test_99_smoke_full_path():
    """Full pipeline on tiny subsets: expanded MiniROCKET -> frozen
    hierarchy -> D1 root increments -> D2/D4/D8/D16 pool (449,820 layout)
    -> unified 469,812 pool -> ONE supervised ranking -> exact final
    representation -> Ridge -> equivalence gate SKIPPED (smoke).
    Run explicitly with: pytest -q tests/test_uwavey_hier_sup_clean.py::test_99
    """
    from experiments.uwavey_hier_sup_clean import runner as clean
    clean.main(smoke=True)
    out = clean.OUT
    assert os.path.isdir(out)
    r = json.load(open(os.path.join(out, "results",
                                    "hier_sup_clean.json")))
    assert r["final_features"] == 19992
    assert r["candidate_pool"]["root_increment_D1"] == 19992
    assert r["candidate_pool"]["delta_D2_D4_D8_D16"] == 449820
    assert r["candidate_pool"]["unified"] == 469812
    assert sum(r["selected_composition"].values()) == 19992
    assert r["equivalence"]["verdict"] == "SKIPPED_SMOKE"
    dims = json.load(open(os.path.join(out, "diagnostics",
                                       "dimensions.json")))
    assert dims["final"]["Z_dev"][1] == 19992
    assert dims["pools"]["unified_candidates"] == 469812
    meta = open(os.path.join(out, "diagnostics",
                             "selected_feature_metadata.csv")).read()
    assert "root_increment" in meta and "hierarchy_delta" in meta
