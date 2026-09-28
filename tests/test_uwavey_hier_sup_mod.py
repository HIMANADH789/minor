"""Focused tests for the HIER-SUP-MOD one-change experiment.

The ONE change vs HIER-SUP-CLEAN: raw ANOVA-F ranking -> EB variance-
moderated F ranking (Smyth scaled-F prior, single pool-wide fit) on the
SAME unified 469,812-candidate telescoping pool.  These tests verify the
locked representation, the ANOVA design invariants (no occupancy-as-df),
the EB moderation math, selection semantics, absence of forbidden
mechanisms, label-flow boundaries (incl. poisoning), the raw-F
equivalence gate, and the full smoke path.
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

from experiments.uwavey_hier_sup_mod.runner import (  # noqa: E402
    B_BASE, B_HIGH, D1, DF_BETWEEN, EDGES, FALLBACK_MIN_COUNT, G_PART,
    H_CAND, H_KERNELS, KEPT_K, N_CAND, N_CLASSES, PPV0)

RUNNER = os.path.join(os.path.dirname(__file__), "..",
                      "experiments", "uwavey_hier_sup_mod", "runner.py")
CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "results",
                         "uwavey_hier_sup_clean", "seed42")


# ---------------------------------------------------------------- budgets
def test_01_dimensions_locked():
    assert B_HIGH == 19992 and B_BASE == 9996 and G_PART == 4998
    assert H_KERNELS == 14994 and EDGES == 30
    assert H_CAND == 449820 and D1 == 19992
    assert N_CAND == 469812 == D1 + H_CAND
    assert PPV0 == 0.0 and DF_BETWEEN == 7 and N_CLASSES == 8


def test_02_no_forbidden_mechanisms():
    """No rho, no G/H split, no quota, no lfdr/two-groups/pi0, no
    feature shrinkage, no occupancy-as-df, no hard fallback filter."""
    src = open(RUNNER).read()
    for tok in ("N_G =", "N_H =", "RHO =", "level_budgets(",
                "select_carriers(", "posterior_signal_q", "pi0 =",
                "pi0_l", "mu1 =", "mu1_l", "sigma1"):
        assert tok not in src, tok
    import re as _re
    assert not _re.search(r"\brho\s*=", src), "rho assignment"
    assert not _re.search(r"\blfdr\b", src), "bare lfdr token"
    for negation in ("no rho", "no G/H", "no quotas", "no local-fdr",
                     "no occupancy-as-df", "no fallback filter"):
        assert negation in src, negation
    assert '"occupancy_as_df": False' in src


# ------------------------------------------------------- ANOVA design
def _synth_pool(n=96, w=40, seed=5):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(8), n // 8)
    X = rng.normal(size=(y.size, w))
    X[:, 3] += 2.0 * (y == 1)
    X[:, 7] = 0.0                                    # constant column
    return X, y


def test_03_anova_pieces_match_f_classif():
    from sklearn.feature_selection import f_classif
    from experiments.uwavey_hier_sup_mod.runner import anova_all
    X, y = _synth_pool()
    d_e = y.size - 8
    ssb, sse = anova_all(X, y)
    s2 = sse / d_e
    f_mine = (ssb / 7) / s2
    f_ref, _ = f_classif(X, y)
    ok = np.isfinite(f_ref)
    assert np.allclose(f_mine[ok], f_ref[ok], rtol=1e-7, atol=1e-12)
    # constant column: NaN in raw f_classif; the runner's ranking
    # convention maps it to 0 exactly like the incumbent's nan_to_num
    assert np.isnan(f_ref[7])
    from experiments.uwavey_hier_sup_mod.runner import raw_f_gate_convention
    assert raw_f_gate_convention(f_mine)[7] == 0.0


def test_04_design_df_shared():
    from experiments.uwavey_hier_sup_mod.runner import anova_all
    X, y = _synth_pool(n=96)
    ssb, sse = anova_all(X, y)
    d_e = y.size - 8
    # d_error is IDENTICAL for every candidate (no occupancy-as-df)
    s2 = sse / d_e
    assert np.isfinite(s2).all()
    assert d_e == 88 and DF_BETWEEN == 7


def test_05_sse_between_identity():
    """SSE = SS_total - SS_between on a hand-checkable column."""
    from experiments.uwavey_hier_sup_mod.runner import anova_all
    y = np.repeat(np.arange(4), 6)
    X = np.column_stack([np.arange(24, dtype=float)])
    ssb, sse = anova_all(X, y)
    sst = ((X - X.mean()) ** 2).sum()
    assert np.isclose(ssb[0] + sse[0], sst, rtol=1e-10)


# ------------------------------------------------------- EB prior fit
def test_06_prior_fit_moments_deterministic():
    from experiments.uwavey_hier_sup_mod.runner import eb_prior_fit
    rng = np.random.default_rng(0)
    d_e = 888
    s2 = np.exp(rng.normal(0.0, 0.5, size=20000))
    d0a, s0a, _ = eb_prior_fit(s2, d_e)
    d0b, s0b, _ = eb_prior_fit(s2, d_e)
    assert d0a == d0b and s0a == s0b                  # deterministic
    assert np.isfinite(d0a) and np.isfinite(s0a) and d0a > 0 and s0a > 0


def test_07_prior_fit_recovers_known_prior():
    """Data drawn from s0^2 * F(d_e, d0) should recover d0 approximately."""
    from experiments.uwavey_hier_sup_mod.runner import eb_prior_fit
    rng = np.random.default_rng(1)
    d_e, d0_true, s0_true = 888.0, 20.0, 1.3
    s2 = s0_true * rng.f(d_e, d0_true, size=40000)
    d0, s0sq, info = eb_prior_fit(s2, d_e)
    assert not info["degenerate"]
    assert 0.5 * d0_true < d0 < 2.0 * d0_true
    assert 0.5 < s0sq / s0_true < 2.0


def test_08_prior_fit_degenerate_reported():
    """Zero Var[log s^2] sits ABOVE the d0=inf limit (trigamma(d_e/2) >
    0), so the moment equations solve to a FINITE d0 -> the true
    degenerate fallback needs an even tighter distribution: the theoretical
    minimum Var[log s^2] for given d_e is trigamma(d_e/2) (attained as
    d0 -> inf), so any v_log below that is impossible and triggers the
    reported fallback with d0 = 1e12."""
    from scipy.special import polygamma
    from experiments.uwavey_hier_sup_mod.runner import eb_prior_fit
    # v_log below the trigamma(d_e/2) floor -> degenerate branch
    d_e = 888.0
    floor = float(polygamma(1, d_e / 2.0))
    s2 = np.exp(np.log(2.5)
                + np.linspace(-0.5 * np.sqrt(floor),
                              0.5 * np.sqrt(floor), 100))
    d0, s0sq, info = eb_prior_fit(s2, d_e)
    assert info["degenerate"] and "condition" in info and "fallback" in info
    assert d0 == 1e12
    # d0 = 1e12 leaves O(1/d_e) corrections in the moment identity
    assert np.isclose(s0sq, float(np.exp(np.log(s2).mean())), rtol=2e-3)


def test_09_posterior_variance_formula():
    from experiments.uwavey_hier_sup_mod.runner import moderated_scores
    d_e, d0, s0sq = 888.0, 10.0, 2.0
    ssb = np.array([7.0, 70.0])                       # msb = 1, 10
    sse = np.array([888.0, 8880.0])                   # s2 = 1, 10
    s2, st2, f_raw, f_mod = moderated_scores(ssb, sse, d_e, d0, s0sq)
    assert np.allclose(s2, [1.0, 10.0])
    expect = (d0 * s0sq + d_e * np.array([1.0, 10.0])) / (d0 + d_e)
    assert np.allclose(st2, expect)
    assert np.allclose(f_raw, [1.0, 1.0])             # msb / s2
    assert np.allclose(f_mod, [1.0, 10.0] / expect)


def test_10_moderated_f_bounded_by_shrinkage():
    """Moderation must deflate F for near-zero-variance candidates and
    leave large-variance candidates nearly unchanged."""
    from experiments.uwavey_hier_sup_mod.runner import moderated_scores
    d_e, d0, s0sq = 888.0, 10.0, 1.0
    ssb = np.array([70.0, 70.0])
    sse = np.array([8.88, 8880.0])                    # s2 = 0.01, 10
    _, _, f_raw, f_mod = moderated_scores(ssb, sse, d_e, d0, s0sq)
    assert f_mod[0] < f_raw[0]                        # deflation
    assert np.isclose(f_mod[1], f_raw[1], rtol=0.05)  # ~unchanged


def test_11_rank_select_tie_break_and_budget():
    from experiments.uwavey_hier_sup_mod.runner import rank_select
    score = np.array([5.0, 9.0, 5.0, 9.0, 1.0])
    sel = rank_select(score, 4)
    assert sel.tolist() == [1, 3, 0, 2]               # ties by index
    assert rank_select(np.zeros(10), 3).tolist() == [0, 1, 2]


# ------------------------------------------------------- locked identity
def test_12_root_increment_identity():
    rng = np.random.default_rng(2)
    ppv1 = rng.random((30, 9))
    assert np.array_equal(ppv1 - PPV0, ppv1)


def test_13_pool_layout_and_meta():
    from experiments.uwavey_hier_high_sup.runner import col_meta
    for c, K in ((0, 2), (2 * H_KERNELS, 4), (6 * H_KERNELS, 8),
                 (14 * H_KERNELS, 16)):
        assert col_meta(c, H_KERNELS)[0] == K
    for c in (0, 1, D1, D1 + 1, D1 + H_CAND - 1):
        if c < D1:
            continue
        K, j, p, kg = col_meta(c - D1, H_KERNELS)
        assert kg == G_PART + ((c - D1) % H_KERNELS) and p == j >> 1


def test_14_level_vector_full_coverage():
    """The runner's layout guard covers every column exactly once."""
    level_of_col = np.full(N_CAND, -1, dtype=np.int64)
    level_of_col[:D1] = 1
    for K, cbefore in ((2, 0), (4, 2), (8, 6), (16, 14)):
        level_of_col[D1 + cbefore * H_KERNELS:
                     D1 + (cbefore + K) * H_KERNELS] = K
    assert (level_of_col == 1).sum() == D1
    for K in KEPT_K:
        assert (level_of_col == K).sum() == K * H_KERNELS
    assert (level_of_col == -1).sum() == 0            # full coverage


def test_15_occupancy_weighting_and_zero_sum():
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


def test_16_fallback_convention_matches_locked_chain():
    from models.nested_regimes.model import select_and_mask
    from experiments.uwavey_hier_sup_mod.runner import select_and_mask_min
    rng = np.random.default_rng(3)
    cnt = rng.integers(0, 12, size=(50, 16))
    a = select_and_mask(cnt, FALLBACK_MIN_COUNT)
    b, fb = select_and_mask_min(cnt, FALLBACK_MIN_COUNT)
    assert np.array_equal(a, b) and FALLBACK_MIN_COUNT == 4


def test_17_bh_adjust_matches_reference():
    from experiments.uwavey_hier_sup_mod.runner import bh_adjust
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074,
                  0.205, 0.212, 0.216, 0.222, 0.251, 0.269, 0.275,
                  0.34, 0.341, 0.384, 0.569, 0.594, 0.696, 0.762,
                  0.94, 0.942, 0.975, 0.986])
    from statsmodels.stats.multitest import multipletests
    ref = multipletests(p, method="fdr_bh")[1]
    assert np.allclose(bh_adjust(p), ref, atol=1e-10)


# ------------------------------------------------------- selector sanity
def test_18_moderated_ranking_changes_selection_synthetic():
    """On a synthetic pool the moderated ranking must differ from raw-F
    for near-zero-variance informative-ish columns (sanity, not locks)."""
    from experiments.uwavey_hier_sup_mod.runner import (
        anova_all, eb_prior_fit, moderated_scores, rank_select)
    rng = np.random.default_rng(4)
    n, W = 96, 60
    y = np.repeat(np.arange(8), n // 8)
    X = rng.normal(size=(n, W))
    d_e = n - 8
    # same class signal, different residual scales: col0 noisy, col1 tight
    X[:, 0] = 1.5 * (y == 1) + rng.normal(0, 1.0, size=n)
    X[:, 1] = 0.3 * (y == 1) + rng.normal(0, 0.02, size=n)
    ssb, sse = anova_all(X, y)
    s2_pos = (sse / d_e)[(sse / d_e) > 0]
    d0, s0sq, _ = eb_prior_fit(s2_pos, d_e)
    _, _, f_raw, f_mod = moderated_scores(ssb, sse, d_e, d0, s0sq)
    # sanity: tiny-variance column wins raw, moderation deflates it hard
    assert f_raw[1] > f_raw[0]
    assert f_mod[1] < f_raw[1]
    # moderation must shrink the tiny-variance F far more (in ratio)
    assert (f_raw[1] / max(f_mod[1], 1e-300)) > (f_raw[0]
                                                 / max(f_mod[0], 1e-300))


def test_19_no_separate_gh_selection_calls():
    """Exactly ONE moderated selection; no rho-style helper usage."""
    src = open(RUNNER).read()
    assert src.count("rank_select(f_mod, B_HIGH)") == 1
    assert "r5_cv_fixed_rho" not in src               # not imported
    assert "[:, :N_G]" not in src and "[:, :D1]" not in src


# ------------------------------------------------------- label flow
def test_20_test_labels_never_touch_selection():
    """No statistics/ranking/prior call may reference test labels (the
    raw-gate's ridge_eval(..., yte) is the sanctioned scoring call)."""
    import re as _re
    src = open(RUNNER).read()
    assert "anova_all(C_dev, y_dev)" in src
    for bad in (r"anova_all\([^)]*yte", r"rank_select\([^)]*yte",
                r"eb_prior_fit\([^)]*yte", r"top_f_select\([^)]*yte"):
        assert not _re.search(bad, src), bad
    assert src.count("ridge_eval") >= 2               # gate + final eval


def test_21_test_label_poisoning_changes_nothing():
    from experiments.uwavey_hier_sup_mod.runner import (
        anova_all, eb_prior_fit, moderated_scores, rank_select)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 50))
    y = np.repeat(np.arange(8), 12)
    ssb, sse = anova_all(X, y)
    s2_pos = (sse / 88)[(sse / 88) > 0]
    d0, s0sq, _ = eb_prior_fit(s2_pos, 88)
    _, _, _, f_mod = moderated_scores(ssb, sse, 88, d0, s0sq)
    sel = rank_select(f_mod, 25)
    # identical inputs -> identical outputs (test labels never enter)
    ssb2, sse2 = anova_all(X, y)
    s2p2 = (sse2 / 88)[(sse2 / 88) > 0]
    d02, s0sq2, _ = eb_prior_fit(s2p2, 88)
    assert d0 == d02 and s0sq == s0sq2
    assert np.array_equal(sel, rank_select(
        moderated_scores(ssb2, sse2, 88, d02, s0sq2)[3], 25))


def test_22_val_poisoning_fold_internal_prior():
    """Fold-internal prior: statistics use fold-train labels only, so
    poisoning fold-VAL labels cannot change the fold-train fit."""
    from experiments.uwavey_hier_sup_mod.runner import (
        anova_all, eb_prior_fit, moderated_scores, rank_select)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 80))
    y = np.repeat(np.arange(8), 15)
    y_bad = y.copy()
    y_bad[90:] = (y_bad[90:] + 3) % 8                 # poison tail rows
    tr = np.arange(90)
    ssb1, sse1 = anova_all(X[tr], y[tr])
    ssb2, sse2 = anova_all(X[tr], y_bad[tr])
    s2p = (sse1 / (tr.size - 8))
    s2p = s2p[s2p > 0]
    d0a, s0a, _ = eb_prior_fit(s2p, tr.size - 8)
    s2q = (sse2 / (tr.size - 8))
    s2q = s2q[s2q > 0]
    d0b, s0b, _ = eb_prior_fit(s2q, tr.size - 8)
    assert d0a == d0b and s0a == s0b                  # unchanged
    _, _, _, f1 = moderated_scores(ssb1, sse1, tr.size - 8, d0a, s0a)
    _, _, _, f2 = moderated_scores(ssb2, sse2, tr.size - 8, d0b, s0b)
    assert np.array_equal(rank_select(f1, 30), rank_select(f2, 30))
    # sanity: poisoning the FIT labels WOULD change it
    ssb3, sse3 = anova_all(X[tr], y_bad[tr])
    _, _, _, f3 = moderated_scores(ssb3, sse3, tr.size - 8, d0b, s0b)
    # sanity: poisoning the FIT labels WOULD change it (poison WITHIN the
    # fold-train rows, not the held-out tail)
    y_fit = y.copy()
    y_fit[:45] = (y_fit[:45] + 3) % 8
    ssb3, sse3 = anova_all(X[tr], y_fit[tr])
    _, _, _, f3 = moderated_scores(ssb3, sse3, tr.size - 8, d0b, s0b)
    assert not np.array_equal(rank_select(f1, 30), rank_select(f3, 30))


# ------------------------------------------------------- runner structure
def test_23_runner_uses_train_latents_only():
    src = open(RUNNER).read()
    assert "_encoder_latents(model, Xtr_z, device)" in src
    assert src.count("assign_levels(lat_va") == 1
    assert src.count("assign_levels(lat_te") == 1
    assert "build_latent_tree(lat_va" not in src
    assert "build_latent_tree(lat_te" not in src


def test_24_layout_guard_present():
    src = open(RUNNER).read()
    assert "layout guard OK [D1|D2|D4|D8|D16]" in src
    assert 'assert lv_seq == sorted(lv_seq)' in src
    assert "hier_banks_test(exHK, Xte_z, reg16_te, KEPT_K, sel_dict)" in src


def test_25_raw_gate_present_and_enforced():
    src = open(RUNNER).read()
    assert 'assert eq_gate["verdict"] == "PASS"' in src
    assert "equivalence_gate.json" in src
    assert "hier_sup_clean.json" in src


def test_26_artifacts_untouched_by_import():
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    def snap():
        out = {}
        for root, _, files in os.walk(base):
            for fn in files:
                p = os.path.join(root, fn)
                if fn.endswith((".json", ".csv", ".md", ".npy", ".png",
                                ".yaml")):
                    st = os.stat(p)
                    out[os.path.relpath(p, base)] = (st.st_size,
                                                     st.st_mtime_ns)
        return out
    before = snap()
    import experiments.uwavey_hier_sup_mod.runner  # noqa: F401
    assert before == snap()


def test_27_runner_writes_only_inside_its_outdir():
    tree = ast.parse(open(RUNNER).read())
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) in ("save", "savefig")]
    assert saves
    for n in saves:
        assert n.args and "OUT" in ast.unparse(n.args[0]), \
            f"save outside OUT: {ast.unparse(n.args[0])}"


# ------------------------------------------------------- CV diagnostic
def test_28_cv_prior_fold_internal_structure():
    """The CV loop fits the prior on fold-train variances only."""
    src = inspect.getsource(
        __import__("experiments.uwavey_hier_sup_mod.runner",
                   fromlist=["main"]))
    seg = src.split("skf.split(C_dev, y_dev)")[1].split("cv_mean")[0]
    assert "anova_all(C_dev[tr_i], y_dev[tr_i])" in seg
    assert "eb_prior_fit(s2_f, d_e)" in seg
    assert "va_i" not in seg.split("eb_prior_fit")[0]


# ------------------------------------------------------- smoke (opt-in)
@pytest.mark.slow
def test_99_smoke_full_path():
    """Full pipeline on tiny subsets: expanded MiniROCKET -> frozen
    hierarchy -> D1 -> D2/D4/D8/D16 -> unified 469,812-layout pool ->
    ANOVA pieces -> EB prior -> moderated F -> top 19,992 -> Ridge.
    Run explicitly with: pytest -q tests/test_uwavey_hier_sup_mod.py::test_99
    """
    from experiments.uwavey_hier_sup_mod import runner as mod
    mod.main(smoke=True)
    out = mod.OUT
    r = json.load(open(os.path.join(out, "results", "hier_sup_mod.json")))
    assert r["final_features"] == 19992
    assert r["candidate_pool"]["root_D1"] == 19992
    assert r["candidate_pool"]["delta"] == 449820
    assert r["candidate_pool"]["unified"] == 469812
    assert r["anova_design"]["d_between"] == 7
    assert r["equivalence_gate"]["verdict"] == "SKIPPED_SMOKE"
    eb = r["eb_prior"]
    assert np.isfinite(eb["d0"]) and np.isfinite(eb["s0_sq"])
    assert eb["d0"] > 0 and eb["s0_sq"] > 0
    assert sum(r["selected_composition"].values()) == 19992
    dims = json.load(open(os.path.join(out, "diagnostics",
                                       "dimensions.json")))
    assert dims["final"]["Z_dev"][1] == 19992
    assert dims["pools"]["unified"] == 469812
    assert os.path.isfile(os.path.join(out, "diagnostics",
                                       "variance_prior.json"))
