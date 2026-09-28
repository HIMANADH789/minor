"""Core math for the differential-Ridge control.

Objective (block-penalty Ridge, one-vs-all binarized targets Y):

    min over beta_G, beta_H:
        ||Y - G beta_G - H beta_H||^2
        + alpha_G ||beta_G||^2
        + alpha_G * gamma ||beta_H||^2

Exact implementation (scaling trick): substitute u = sqrt(gamma) * beta_H.
The model term H beta_H = (H/sqrt(gamma)) u and the penalty becomes
alpha_G (||beta_G||^2 + ||u||^2). Hence fitting ONE ordinary
RidgeClassifier(alpha_G) on [G || H/sqrt(gamma)] solves the block-penalty
problem EXACTLY, with prediction mapping beta_H_hat = u_hat / sqrt(gamma)
and identical margins/predictions in original coordinates:

    margin = G w_G + (H/sqrt(gamma)) u + b = G w_G + H (u/sqrt(gamma)) + b.

Direct verification solves the SAME objective in primal coordinates
(block-diagonal penalty matrix P = diag(I_G, gamma I_H)) and via its dual
W = (GG^T + gamma HH^T)^T (GG^T + gamma HH^T + alpha_G I)^{-1} [GY + gamma HY]
(the gamma factors enter from differentiating the squared penalty).
"""
import numpy as np
from sklearn.linear_model import RidgeClassifier

from experiments.rcmkn_haptics_dridge_seed42.config import ALPHA_G_CANON


def dridge_fit(G, H, y, alpha_G=ALPHA_G_CANON, gamma=1):
    """Fit [G || H/sqrt(gamma)] with ONE RidgeClassifier(alpha_G).

    Returns (est, scaler) where predictions in ORIGINAL coordinates are
    obtained via dridge_predict / dridge_decision_function."""
    scaler = 1.0 / np.sqrt(gamma)
    X = np.hstack([G, H * scaler])
    est = RidgeClassifier(alpha=alpha_G)
    est.fit(X, y)
    return est, scaler


def dridge_decision_function(est, scaler, G, H):
    """Margins in the ORIGINAL H coordinates: beta_H_hat = u_hat * scaler.

    RidgeClassifier.coef_ has shape (n_classes, n_features); class c margin
    is <w_c, x> + b_c, so we transpose the per-class weight rows."""
    pG = G.shape[1]
    wg = est.coef_[:, :pG]                    # (C, pG)
    u = est.coef_[:, pG:]                     # (C, pH)
    return G @ wg.T + H @ (u * scaler).T + est.intercept_


def dridge_predict(est, scaler, G, H):
    return np.argmax(dridge_decision_function(est, scaler, G, H), axis=1)


def _center(X, Y):
    return X - X.mean(0, keepdims=True), Y - Y.mean(0, keepdims=True)


def dridge_primal_direct(G, H, Y, alpha_G=ALPHA_G_CANON, gamma=1):
    """Direct primal solve of the block-penalty objective (centered)."""
    pG, pH = G.shape[1], H.shape[1]
    X, Yc = _center(np.hstack([G, H]), Y)
    P = np.zeros((pG + pH, pG + pH))
    P[:pG, :pG] = np.eye(pG)
    P[pG:, pG:] = gamma * np.eye(pH)
    b = X.T @ Yc
    return np.linalg.solve(X.T @ X + alpha_G * P, b)


def dridge_dual_direct(G, H, Y, alpha_G=ALPHA_G_CANON, gamma=1):
    """Dual solve in sample space (Hermitian n x n system).

    With X~ = [G || H/sqrt(gamma)] (the scaling-trick features), the
    solution in ORIGINAL coordinates is
        w = [G^T ; H^T/gamma] (X~ X~^T + alpha I)^{-1} Yc,
    i.e. kernel K = G G^T + (1/gamma) H H^T on the centered blocks."""
    X, Yc = _center(np.hstack([G, H]), Y)
    pG = G.shape[1]
    XG, XH = X[:, :pG], X[:, pG:]
    K = XG @ XG.T + (1.0 / gamma) * (XH @ XH.T)
    A = K + alpha_G * np.eye(X.shape[0])
    coef = np.linalg.solve(A, Yc)                       # (n, C)
    return np.vstack([XG.T, XH.T / gamma]) @ coef       # (p, C)


def dridge_reference_ridge(G, H, y, alpha_G=ALPHA_G_CANON):
    """gamma=1 sanity anchor: identical to a plain RidgeClassifier."""
    est, sc = dridge_fit(G, H, y, alpha_G=alpha_G, gamma=1)
    return est, sc


def score_macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def cv_macro_f1_for_gamma(G, H, y, gammas, k_folds, alpha_G=ALPHA_G_CANON):
    """Stratified k-fold CV of the differential Ridge at frozen alpha_G.

    Returns dict gamma -> (mean, std, fold scores). No test labels ever
    enter this function."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    out = {}
    for g in gammas:
        scores = []
        for tr, va in skf.split(np.zeros(len(y)), y):
            est, scaler = dridge_fit(G[tr], H[tr], y[tr],
                                     alpha_G=alpha_G, gamma=g)
            pred = dridge_predict(est, scaler, G[va], H[va])
            scores.append(score_macro_f1(y[va], pred))
        out[g] = (float(np.mean(scores)), float(np.std(scores)), scores)
    return out


def select_gamma(cv_results, tie_tol=0.001):
    """argmax mean CV Macro-F1; ties within tie_tol -> smaller gamma."""
    best = max(m for m, _, _ in cv_results.values())
    return min(g for g, (m, _, _) in cv_results.items()
               if m >= best - tie_tol)
