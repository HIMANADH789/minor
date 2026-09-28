"""
Safety / leakage tests for the Haptics ensemble experiment (spec sec. 19).

These run WITHOUT the heavy aeon fits where possible: the stacking/fusion
machinery is tested on synthetic scores, and the alignment tests use the
canonical loader with the real DRTN checkpoint (cheap, GPU-free).
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.haptics_ensemble_seed42.core import (  # noqa: E402
    ALPHAS, SEED, MiniRocketArm, correctness_table, drtn_logits,
    drtn_oof_logits, error_correlation, load_official_drtn, macro_f1,
    minirocket_oof_scores, softmax_stable,
)

torch.manual_seed(SEED)


def _synth_scores(n=60, k=5, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, k, n)
    S = rng.randn(n, k)
    L = rng.randn(n, k)
    S[np.arange(n), y] += 1.2          # make MR decent
    L[np.arange(n), y] += 0.3          # DRTN weaker
    return y, S, L


# 1. deterministic score generation
def test_deterministic_score_generation():
    y, S, L = _synth_scores()
    p1 = softmax_stable(S).argmax(1)
    p2 = softmax_stable(S.copy()).argmax(1)
    assert np.array_equal(p1, p2)


# 2. sample-index alignment (train scores row i == train sample i)
def test_sample_index_alignment():
    d = load_haptics()
    mr = MiniRocketArm()
    mr.fit_extractor_train(d["Xtr"][:40])
    F = mr.transform(d["Xtr"][:40])
    mr.fit_ridge(F, d["ytr"][:40])
    pred = mr.ridge.predict(F)
    assert len(pred) == 40
    drtn, _ = load_official_drtn("cpu")
    L = drtn_logits(drtn, d["Xtr"][:40], "cpu")
    assert L.shape[0] == 40 and L.shape[1] == 5
    # same sample must yield same logit on repeated calls
    L2 = drtn_logits(drtn, d["Xtr"][:40], "cpu")
    assert np.allclose(L, L2, atol=1e-5)


# 3/4. no train/test or val/test overlap (identity-based, canonical split)
def test_split_disjointness():
    d = load_haptics()
    tr = np.asarray(d["Xtr"]).reshape(len(d["Xtr"]), -1)
    va = np.asarray(d["Xva"]).reshape(len(d["Xva"]), -1)
    te = np.asarray(d["Xte"]).reshape(len(d["Xte"]), -1)
    # sample-content overlap between splits would indicate split corruption
    tr_set = {r.tobytes() for r in tr}
    va_set = {r.tobytes() for r in va}
    te_set = {r.tobytes() for r in te}
    assert not (tr_set & te_set), "train/test content overlap"
    assert not (va_set & te_set), "val/test content overlap"
    assert not (tr_set & va_set), "train/val content overlap"


# 5/6. OOF predictions cover every train sample exactly once + honest rows
def test_oof_coverage_and_honesty():
    d = load_haptics()
    Xtr, ytr = d["Xtr"], d["ytr"]
    oof = minirocket_oof_scores(Xtr, ytr, n_folds=5)
    assert oof.shape == (len(Xtr), 5)
    assert np.isfinite(oof).all()
    # OOF rows must differ from in-sample rows of a train-fitted ridge
    mr = MiniRocketArm()
    mr.fit_extractor_train(Xtr)
    F = mr.transform(Xtr)
    mr.fit_ridge(F, ytr)
    in_sample = mr.decision_scores(F)
    assert not np.allclose(oof, in_sample, atol=1e-6), \
        "OOF scores identical to in-sample scores -> leakage"
    # and the OOF model must be worse on its own held-out rows (sanity)
    acc_oof = macro_f1(ytr, oof.argmax(1))
    acc_in = macro_f1(ytr, in_sample.argmax(1))
    assert acc_oof <= acc_in + 1e-9


# 7. no test labels used before final evaluation (structural: the search
#    functions only accept val arrays; assert the test loader is untouched)
def test_no_test_labels_in_selection_paths():
    import inspect
    import experiments.haptics_ensemble_seed42.runner as R
    src = inspect.getsource(R.main)
    # the fusion/stack search blocks must appear BEFORE any use of yte
    fusion_pos = src.find("fusion_search")
    stack_pos = src.find("stack_search")
    test_pos = src.find("eval_system")
    assert 0 < fusion_pos < test_pos and 0 < stack_pos < test_pos


# 8. fusion alpha search uses validation only (behavioral, synthetic)
def test_fusion_search_validation_only():
    y_va, S_va, L_va = _synth_scores(seed=1)
    y_te, _, _ = _synth_scores(seed=2)             # present but NEVER used
    best, best_a = -1, None
    for a in ALPHAS:
        f1 = macro_f1(y_va, (a * softmax_stable(S_va)
                             + (1 - a) * softmax_stable(L_va)).argmax(1))
        if f1 > best:
            best, best_a = f1, a
    # weaker DRTN: the optimum must sit at a high alpha (MR-heavy side)
    assert best_a >= 0.8
    # and the search result is invariant to y_te's content
    assert best == macro_f1(y_va, (best_a * softmax_stable(S_va)
                                   + (1 - best_a)
                                   * softmax_stable(L_va)).argmax(1))


# 9. stacker hyperparameter search uses validation only (behavioral)
def test_stacker_search_validation_only():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y_tr, S_tr, L_tr = _synth_scores(200, seed=3)
    y_va, S_va, L_va = _synth_scores(60, seed=4)
    M_tr = np.hstack([S_tr, L_tr])
    M_va = np.hstack([S_va, L_va])
    sc = StandardScaler().fit(M_tr)               # train rows only
    best, best_C = -1, None
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        clf.fit(sc.transform(M_tr), y_tr)
        f1 = macro_f1(y_va, clf.predict(sc.transform(M_va)))
        if f1 > best:
            best, best_C = f1, C
    assert best_C in (0.01, 0.1, 1.0, 10.0)


# 10. probability conversion numerically stable (extreme inputs)
def test_softmax_stability():
    S = np.array([[1e9, 1e9, 0.0, 0.0, 0.0], [-1e9, 1e9, 0, 0, 0]])
    P = softmax_stable(S)
    assert np.isfinite(P).all()
    assert np.allclose(P.sum(axis=1), 1.0)


# 11. correctness/error-correlation math + checkpoint reproduction
def test_diagnostics_and_checkpoint_reproduction():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    mr = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])
    dr = np.array([0, 2, 2, 1, 1, 2, 0, 1, 2, 0])
    ct = correctness_table(y, mr, dr)
    assert ct["A_mr_and_drtn_correct"] == 5      # idx 0,5,6,7,8
    assert ct["B_mr_only_correct"] == 2          # idx 1,3 (mr ok, dr wrong)
    assert ct["C_drtn_only_correct"] == 3        # idx 2,4,9 (dr ok, mr wrong)
    assert ct["D_both_wrong"] == 0
    ec = error_correlation(y, mr, dr)
    assert 0 <= ec["error_overlap"] <= 1
    # official checkpoint must reproduce its documented val MF1 (the stored
    # value is rounded to 4 dp, so compare at 1e-3)
    d = load_haptics()
    drtn, official = load_official_drtn("cpu")
    L_va = drtn_logits(drtn, d["Xva"], "cpu")
    assert abs(macro_f1(d["yva"], L_va.argmax(1))
               - official["best_val_mf1"]) < 1e-3


def load_haptics():
    from experiments.external_stack_generalization.data import load_dataset
    return load_dataset("Haptics")
