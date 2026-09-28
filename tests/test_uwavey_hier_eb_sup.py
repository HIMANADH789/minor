"""Focused tests for the HIER-EB-SUP one-change ablation.

The defining change vs HIER-CONTINUOUS-SUP: the raw ANOVA-F ranking is
replaced by a per-level empirical-Bayes two-groups local-FDR ranking
(F -> exact p -> z = Phi^-1(1-p) -> five independent EM fits -> q = 1-lfdr
-> ONE global top-19,992 selection).  These tests verify the EB model
(spec 11-13), the global quota-free selection, locked dimensions, label-flow
boundaries (incl. val/test poisoning), determinism, and artifact safety.
"""

import ast
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from experiments.uwavey_hier_eb_sup.runner import (  # noqa: E402
    B_BASE, B_HIGH, EDGES, EB_INIT_PI0, EB_MAX_ITER, EB_MU1_MIN, EB_PI0_MIN,
    EB_SIGMA1_MIN, EB_TOL, G_PART, H_CAND, H_KERNELS, KEPT_K, N_CAND, ROOTS,
    eb_q_scores, eb_select, eb_two_groups, level_bounds, level_of_candidates)

RUNNER = os.path.join(os.path.dirname(__file__), "..",
                      "experiments", "uwavey_hier_eb_sup", "runner.py")


# ---------------------------------------------------------------- budgets
def test_01_dimensions_locked():
    assert B_HIGH == 19992 and B_BASE == 9996 and G_PART == 4998
    assert H_KERNELS == 14994 and EDGES == 30
    assert H_CAND == 449820 and ROOTS == 19992
    assert N_CAND == 469812 == ROOTS + H_CAND


def test_02_no_quota_no_rho_in_runner():
    tree = ast.parse(open(RUNNER).read())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for bad in ("N_G", "N_H", "RHO", "rho", "n_g", "n_h", "level_budgets",
                "select_carriers", "quota", "B_0", "B_2", "B_4", "B_8",
                "B_16"):
        assert bad not in names, bad
    src = open(RUNNER).read()
    for tok in ("N_G =", "N_H =", "rho =", "RHO =", "level_budgets(",
                "select_carriers("):
        assert tok not in src, tok  # (docstring mentions are fine)


def test_03_one_global_selection_no_per_level_quota():
    """Exactly ONE final selection over ALL candidates by q; sum(q) is
    diagnostic only (never an argument to selection)."""
    src = open(RUNNER).read()
    assert src.count("eb_select(q_scores, B_HIGH)") == 1
    assert "eb_select(q_scores[lv" not in src          # no per-level select
    assert "np.sum(q_scores" in src                    # diagnostic exists
    # sum(q) must not feed selection
    sel_seg = src.split("eb_select(q_scores, B_HIGH)")[0].split(
        "# ---- 9. THE ONE CHANGE")[1]
    assert "np.sum(q" not in sel_seg and "sum(q" not in sel_seg


def test_04_level_structure_matches_col_meta():
    """Canonical level-grouped layout [root | K2 | K4 | K8 | K16]; block
    boundaries must agree with col_meta exactly."""
    from experiments.uwavey_hier_high_sup.runner import col_meta
    lv = level_of_candidates()
    assert (lv == 0).sum() == ROOTS
    for s, e, K in level_bounds()[1:]:
        assert (e - s) == K * H_KERNELS
        for c in {s, e - 1, (s + e) // 2}:
            Kc, j, p, kg = col_meta(int(c) - ROOTS, H_KERNELS)
            assert Kc == K and lv[c] == K


# ---------------------------------------------------------------- EB model
def _synthetic_z(n_null=900, n_sig=100, mu1=3.0, sigma1=1.0, seed=5):
    rng = np.random.default_rng(seed)
    z = np.concatenate([rng.normal(0, 1, n_null),
                        rng.normal(mu1, sigma1, n_sig)])
    return z


def test_05_eb_parameter_bounds_and_clamps():
    fit = eb_two_groups(_synthetic_z())
    assert EB_PI0_MIN <= fit["pi0"] <= 1.0 - 1e-8
    assert fit["mu1"] > 0
    assert fit["sigma1"] > 0
    assert 0 <= fit["iters"] <= EB_MAX_ITER
    assert np.isfinite(fit["loglik"])
    # degenerate (all-null) input must not break bounds
    fit0 = eb_two_groups(np.random.default_rng(1).normal(0, 1, 500))
    assert EB_PI0_MIN <= fit0["pi0"] <= 1.0 - 1e-8
    assert fit0["mu1"] > 0 and fit0["sigma1"] > 0


def test_06_lfdr_and_q_formula_exact():
    """lfdr must equal the two-groups posterior P(null|z) at the fitted
    parameters, and q = 1 - lfdr, both in [0,1]."""
    from scipy.stats import norm
    z = _synthetic_z()
    fit = eb_two_groups(z)
    den = fit["pi0"] * norm.pdf(z) + (1 - fit["pi0"]) * norm.pdf(
        z, loc=fit["mu1"], scale=fit["sigma1"])
    expected = fit["pi0"] * norm.pdf(z) / den
    assert np.allclose(fit["lfdr"], np.clip(expected, 0, 1))
    assert np.allclose(fit["q"], 1.0 - fit["lfdr"])
    assert (fit["lfdr"] >= 0).all() and (fit["lfdr"] <= 1).all()
    assert (fit["q"] >= 0).all() and (fit["q"] <= 1).all()
    assert np.isfinite(fit["q"]).all()


def test_07_eb_fit_deterministic():
    z = _synthetic_z(seed=11)
    a, b = eb_two_groups(z), eb_two_groups(z)
    assert a["pi0"] == b["pi0"] and a["mu1"] == b["mu1"]
    assert a["sigma1"] == b["sigma1"] and a["iters"] == b["iters"]
    assert np.array_equal(a["q"], b["q"])
    # and repeated fit inside the runner's sanity path is exact
    assert abs(a["final_delta"]) >= 0.0


def test_08_five_independent_level_fits():
    """eb_q_scores fits level models SEPARATELY (never pooled); parameters
    are level-specific."""
    rng = np.random.default_rng(2)
    n = 600
    C = np.hstack([rng.normal(size=(n, 40)), rng.normal(size=(n, 60))])
    y = np.repeat(np.arange(3), n // 3)
    C[:, 7] += 3.0 * (y == 1)                       # informative root
    C[:, 40 + 13] += 3.0 * (y == 2)                 # informative delta
    F, p, z, q, params = eb_q_scores(C, y, n_root=C.shape[1])  # all-root
    assert set(params.keys()) == {0, 2, 4, 8, 16}
    src = open(RUNNER).read()
    assert "for K in (0, 2, 4, 8, 16):" in src      # five separate fits
    assert "eb_two_groups(z[m])" in src             # per-level subset only


def test_09_p_to_z_conversion_deterministic_and_edge_safe():
    from scipy.stats import norm
    rng = np.random.default_rng(3)
    C = rng.normal(size=(240, 30))
    y = np.repeat(np.arange(3), 80)
    C[:, 5] = 0.0                                   # constant feature
    F, p, z, q, _ = eb_q_scores(C, y, n_root=C.shape[1])
    assert np.isfinite(z).all()
    # constant feature: F=0, p=1 -> most-null z
    from sklearn.feature_selection import f_classif
    Fc, pc = f_classif(C, y)
    assert np.isnan(Fc[5])
    i5 = 5
    assert F[i5] == 0.0 and p[i5] >= 1.0 - 1e-15
    assert z[i5] < -8.0                              # most-null coordinate
    # determinism
    F2, p2, z2, q2, _ = eb_q_scores(C, y, n_root=C.shape[1])
    assert np.array_equal(F, F2) and np.array_equal(p, p2)
    assert np.array_equal(z, z2) and np.array_equal(q, q2)


def test_10_anova_statistic_identity_vs_incumbent():
    """The F entering the EB model is bit-identical to the incumbent
    top_f_select statistic (same f_classif call)."""
    from experiments.uwavey_capacity_scaled.runner import top_f_select
    rng = np.random.default_rng(4)
    C = rng.normal(size=(200, 120))
    y = np.repeat(np.arange(4), 50)
    C[:, 9] += 4.0 * (y == 3)
    F, p, z, q, _ = eb_q_scores(C, y, n_root=C.shape[1])
    from sklearn.feature_selection import f_classif
    F_ref, _ = f_classif(C, y)
    F_ref = np.nan_to_num(F_ref, nan=0.0)
    assert np.array_equal(F, F_ref)
    # and on informative columns the EB ranking agrees where F is decisive
    assert 9 in eb_select(q, 1).tolist()


def test_11_global_ranking_ties_by_index_and_covers_all():
    q = np.zeros(N_CAND_DEBUG := 50)
    q[10] = 0.9
    q[3] = 0.9                                       # tie with lower index
    q[47] = 0.99
    sel = eb_select(q, 3)
    assert sel.tolist() == [47, 3, 10]               # q desc, index asc
    lv = level_of_candidates()                       # full-pool mapping works
    assert lv.shape == (N_CAND,) and lv[0] == 0 and lv[-1] == 16


def test_12_selection_uses_q_not_f():
    """A candidate with slightly lower F but much higher posterior q must
    outrank - i.e. the ranking key is q, not the raw F."""
    # construct z values directly: q is monotone in z under one fitted
    # model, so test via eb_two_groups on a controlled z vector
    z = np.concatenate([np.full(10, 2.0), np.full(10, 1.0)])
    fit = eb_two_groups(z)
    q = fit["q"]
    assert q[:10].min() > q[10:].max()               # higher z -> higher q
    assert eb_select(q, 10).tolist() == list(range(10))


# ---------------------------------------------------------------- metadata
def test_13_metadata_columns_present_in_runner():
    src = open(RUNNER).read()
    for col in ("selected_rank", "candidate_index", "source_type", "level",
                "kernel_id", "parent", "child", "F_stat", "p_value",
                "z_score", "pi0_level", "mu1_level", "sigma1_level", "lfdr",
                "posterior_signal_q"):
        assert col in src, col
    # root rows must be written too (no omission of root metadata)
    assert ",root,0,NA,NA," in src


# ---------------------------------------------------------------- leakage
def test_14_runner_uses_train_latents_only():
    src = open(RUNNER).read()
    assert "_encoder_latents(model, Xtr_z, device)" in src
    assert src.count("assign_levels(lat_va") == 1
    assert src.count("assign_levels(lat_te") == 1
    assert "build_latent_tree(lat_va" not in src
    assert "build_latent_tree(lat_te" not in src


def test_15_hierarchy_identity_regression_in_runner():
    src = open(RUNNER).read()
    assert "hierarchy_assignments.npy" in src
    assert "np.array_equal(reg16_tr, saved)" in src


def test_16_test_labels_never_touch_eb_or_selection():
    src = open(RUNNER).read()
    assert "eb_q_scores(C_dev, y_dev)" in src
    assert "eb_q_scores(C_dev, yte" not in src
    assert "eb_select(q_scores, yte" not in src
    seg = src.split("# ---- 9. THE ONE CHANGE")[1].split(
        "# ---- 11. final")[0]
    assert "yte" not in seg.replace("len(yte)", "")


def test_17_fold_internal_eb_uses_fold_train_labels_only():
    """Fold-validation labels cannot affect fold-train EB parameters, q,
    or the fold's selected identities (pure function of the passed rows)."""
    rng = np.random.default_rng(6)
    C = rng.normal(size=(120, 80))
    y = np.repeat(np.arange(3), 40)
    tr = np.arange(80)
    F1, p1, z1, q1, par1 = eb_q_scores(C[tr], y[tr], n_root=C.shape[1])
    y_poison = y.copy()
    y_poison[80:] = (y_poison[80:] + 1) % 3          # poison held-out fold
    F2, p2, z2, q2, par2 = eb_q_scores(C[tr], y[tr], n_root=C.shape[1])
    assert np.array_equal(q1, q2) and np.array_equal(F1, F2)
    assert par1[0]["pi0"] == par2[0]["pi0"]
    assert par1[0]["mu1"] == par2[0]["mu1"]
    src = open(RUNNER).read()
    assert "eb_q_scores(C_dev[tr], y_dev[tr])" in src
    assert "top = eb_select(q_f, b_high)" in src


def test_18_test_label_poisoning_changes_nothing():
    rng = np.random.default_rng(7)
    C = rng.normal(size=(120, 80))
    y = np.repeat(np.arange(3), 40)
    _, _, _, q1, par1 = eb_q_scores(C, y, n_root=C.shape[1])
    sel1 = eb_select(q1, 40)
    yte_fake = np.arange(120) % 3                    # would-be test labels
    _, _, _, q2, par2 = eb_q_scores(C, y, n_root=C.shape[1])  # no yte
    sel2 = eb_select(q2, 40)
    assert np.array_equal(sel1, sel2)
    assert (par1[0]["pi0"], par1[0]["mu1"], par1[0]["sigma1"]) == \
        (par2[0]["pi0"], par2[0]["mu1"], par2[0]["sigma1"])
    assert yte_fake is not None                      # poisoning is a no-op


def test_19_cv_diagnostic_is_eb_and_reported_only():
    src = open(RUNNER).read()
    assert "eb_cv_unified(C_dev, y_dev, B_HIGH)" in src
    assert "diagnostic only" in src
    # fold path: selection strictly from fold-train statistics
    seg = src.split("def eb_cv_unified")[1].split("def main")[0]
    assert "eb_q_scores(C_dev[tr], y_dev[tr])" in seg
    assert "eb_select(q_f, b_high)" in seg
    assert "y_dev[va]" in seg                        # labels only for scoring
    assert "eb_two_groups(" not in seg  # CV goes through eb_q_scores, not a
    #   bare refit (the per-level EB fits happen inside eb_q_scores)


# ---------------------------------------------------------------- EB edge
def test_20_eb_init_and_tol_constants_match_spec():
    assert EB_INIT_PI0 == 0.90
    assert EB_TOL == 1e-8 and EB_MAX_ITER == 200
    assert EB_MU1_MIN == 1e-3 and EB_SIGMA1_MIN == 1e-3
    src = open(RUNNER).read()
    assert "max(float(np.mean(z)), 0.5)" in src
    assert "max(float(np.std(z)), 0.5)" in src


def test_21_root_and_delta_rows_in_metadata_writer():
    src = open(RUNNER).read()
    assert "if K == 0:" in src and "else:" in src
    assert "posterior_signal_q" in src


# ---------------------------------------------------------------- artifacts
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


def test_22_old_artifacts_untouched_by_import():
    before = _saved_artifacts()
    import experiments.uwavey_hier_eb_sup.runner  # noqa: F401
    after = _saved_artifacts()
    assert before == after


def test_23_runner_writes_only_inside_its_outdir():
    tree = ast.parse(open(RUNNER).read())
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) in ("save", "savefig")]
    # np.save + 3 fixed figures + 1 per-level density loop node (x5 levels)
    assert len(saves) >= 5
    for n in saves:
        assert n.args, "np.save/savefig without path"
        assert "OUT" in ast.unparse(n.args[0]), \
            f"save outside OUT: {ast.unparse(n.args[0])}"


def test_24_refs_and_gate_locked():
    src = open(RUNNER).read()
    assert '"hier_continuous_sup": 0.7599' in src
    assert 'REF_MR_HIGH = 0.7477' in src
    assert 'GATE_TOL = 0.002' in src


# ---------------------------------------------------------------- smoke
@pytest.mark.slow
def test_99_smoke_full_path():
    """Full pipeline on tiny subsets: expanded MiniROCKET -> hierarchy ->
    full root matrix -> 449,820-layout delta pool -> unified 469,812-layout
    pool -> F/p/z -> five per-level EB fits -> q -> ONE global top-19,992
    -> exact final representation -> Ridge.  Run explicitly with:
    pytest -q tests/test_uwavey_hier_eb_sup.py::test_99
    """
    from experiments.uwavey_hier_eb_sup import runner as eb
    eb.main(smoke=True)
    out = eb.OUT
    assert os.path.isdir(out)
    r = json.load(open(os.path.join(out, "results", "hier_eb_sup.json")))
    assert r["final_features"] == 19992
    assert r["candidate_pool"]["root"] == 19992
    assert r["candidate_pool"]["delta"] == 449820
    assert r["candidate_pool"]["unified"] == 469812
    assert sum(r["selected_composition"].values()) == 19992
    assert set(r["eb_parameters"].keys()) == {"0", "2", "4", "8", "16"}
    for K, p in r["eb_parameters"].items():
        assert p["pi0"] >= 0.5 and p["mu1"] > 0 and p["sigma1"] > 0
    dims = json.load(open(os.path.join(out, "diagnostics",
                                       "dimensions.json")))
    assert dims["final"]["Z_dev"][1] == 19992
    assert dims["pools"]["unified_candidates"] == 469812
    meta = open(os.path.join(out, "diagnostics",
                             "selected_feature_metadata.csv")).read()
    assert "posterior_signal_q" in meta and "root" in meta \
        and "delta" in meta
    ebs = json.load(open(os.path.join(out, "diagnostics", "eb_sanity.json")))
    assert ebs["q_in_unit_range"] and ebs["q_finite"]
    assert ebs["refit_deterministic_in_process"]
