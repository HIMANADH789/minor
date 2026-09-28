"""Label-aware complementary-information analysis (no rho, no test use).

Question: does H_unique carry predictive information about Y beyond G?

Protocol (Haptics, seed 42):
  Base model      : RidgeClassifierCV(alphas=logspace(-4,4,20)) on G_train,
                    selected by LOO CV (sklearn default) -- the canonical
                    project classifier; validation split untouched.
  Augmented model : same classifier on [G_train, H_unique_train].
  Incremental statistic: difference in 5-fold stratified-CV macro-F1 on
                    TRAIN (dev-only, never test) between augmented and
                    base models, computed on the same folds (paired).
  Null            : H_unique carries no class-conditional information.
                    Implemented by permuting each H_unique column with a
                    column-wise seeded permutation (destroys association
                    with both Y and G while preserving marginal structure),
                    recomputing the paired CV increment, 2000 permutations
                    (predeclared), fixed RNG seed 42042.
  Empirical p     : one-sided P(null increment >= observed increment).
  Effect size     : mean CV macro-F1 increment + Cohen's d of paired
                    fold-score differences.

Both fits use TRAIN (train fold) labels only; the validation split stays
reserved for the canonical protocol; test is never touched here.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ALPHAS = np.logspace(-4, 4, 20)
K_FOLDS = 5
N_PERM = 2000
PERM_SEED = 42042
CV_SEED = 42


def _cv_scores(X, y, splitter):
    scores = []
    for tr_i, va_i in splitter.split(np.zeros(len(y)), y):
        clf = RidgeClassifierCV(alphas=ALPHAS)
        clf.fit(X[tr_i], y[tr_i])
        scores.append(f1_score(y[va_i], clf.predict(X[va_i]),
                               average="macro", zero_division=0))
    return np.array(scores)


def paired_cv_increment(G, Hu, y):
    """Paired 5-fold CV macro-F1 increment of [G, Hu] over G."""
    splitter = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                               random_state=CV_SEED)
    X_aug = np.hstack([G, Hu])
    s_base = _cv_scores(G, y, splitter)
    s_aug = _cv_scores(X_aug, y, splitter)
    d = s_aug - s_base
    obs = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    cohens_d = obs / sd if sd > 1e-12 else 0.0
    return {
        "cv_base_mean": round(float(s_base.mean()), 6),
        "cv_base_std": round(float(s_base.std()), 6),
        "cv_aug_mean": round(float(s_aug.mean()), 6),
        "cv_aug_std": round(float(s_aug.std()), 6),
        "observed_increment": round(obs, 6),
        "fold_deltas": [round(float(v), 6) for v in d],
        "cohens_d": round(cohens_d, 4),
        "splitter": f"StratifiedKFold({K_FOLDS}, shuffle, seed={CV_SEED})",
    }


def permutation_complementarity_test(G, Hu, y, n_perm=N_PERM,
                                     seed=PERM_SEED):
    """Column-permutation null for the incremental CV statistic."""
    rng = np.random.RandomState(seed)
    base = paired_cv_increment(G, Hu, y)
    obs = base["observed_increment"]
    null = np.empty(n_perm)
    for i in range(n_perm):
        Hup = Hu[rng.permutation(len(Hu))]
        null[i] = paired_cv_increment(G, Hup, y)["observed_increment"]
        if (i + 1) % 250 == 0:
            print(f"    perm {i+1}/{n_perm}: null mean so far "
                  f"{null[:i+1].mean():.4f}", flush=True)
    p = float((null >= obs).mean())
    return {
        "base": base,
        "observed_increment": obs,
        "null_mean": round(float(null.mean()), 6),
        "null_std": round(float(null.std()), 6),
        "null_p95": round(float(np.percentile(null, 95)), 6),
        "empirical_p_one_sided": round(p, 5),
        "n_permutations": n_perm,
        "permutation_seed": seed,
        "null_distribution": [round(float(v), 6) for v in null[::20]],
    }


def plus_one_permutation_test(G, Hu, y, n_perm=1000, seed=42042):
    """Additive (plus-one) permutation test for the paired CV increment.

    Null: H_unique adds no label-relevant information conditional on G.
    Permuting the rows of Hu independently of (G, y) destroys any
    association of Hu with both while preserving its marginal structure.
    p = (1 + #{null >= observed}) / (S + 1).
    """
    rng = np.random.RandomState(seed)
    base = paired_cv_increment(G, Hu, y)
    obs = base["observed_increment"]
    null = np.empty(n_perm)
    for i in range(n_perm):
        Hup = Hu[rng.permutation(len(Hu))]
        null[i] = paired_cv_increment(G, Hup, y)["observed_increment"]
        if (i + 1) % 100 == 0:
            print(f"    perm {i+1}/{n_perm}: null mean so far "
                  f"{null[:i+1].mean():.4f}", flush=True)
    p = float((1 + int((null >= obs).sum())) / (n_perm + 1))
    return {
        "base": base,
        "observed_increment": obs,
        "null_mean": round(float(null.mean()), 6),
        "null_std": round(float(null.std()), 6),
        "null_p95": round(float(np.percentile(null, 95)), 6),
        "empirical_p_one_sided": round(p, 5),
        "p_formula": "(1 + count(null >= observed)) / (S + 1)",
        "n_permutations": n_perm,
        "permutation_seed": seed,
    }


def dev_fit_eval(G, Hu, ytr, yva, G_va=None, Hu_va=None):
    """Canonical dev-set evaluation (train-fit -> val score, no test)."""
    models = {}
    Xtr = G if Hu is None else np.hstack([G, Hu])
    Xva = G_va if Hu_va is None else np.hstack([G_va, Hu_va])
    clf = RidgeClassifierCV(alphas=ALPHAS)
    clf.fit(Xtr, ytr)
    from sklearn.metrics import f1_score
    models["val_macro_f1"] = round(float(f1_score(
        yva, clf.predict(Xva), average="macro", zero_division=0)), 4)
    models["alpha"] = float(clf.alpha_)
    return models
