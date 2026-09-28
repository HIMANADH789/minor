"""Confound-isolated FULL-DIMENSION CCA-adaptive generalized ridge.

PROPOSED experimental formulation (this project) — NOT a literature-
established "Canonical Ridge" equation.  The direction-specific shrinkage

    alpha_k = alpha_base * (1 + gamma * rho_k^2)

on the k-th CCA-identified H coordinate (alpha_base everywhere else) is
our own smooth bounded rule, fixed with gamma = 1.0.

Model
-----
The final predictive design keeps ALL original features:

    Z = [G_full(4998) || H_cca(29) || H_perp(4969)]   (9996 columns)

where H_cca/H_perp are the train-centered H features expressed in the
frozen orthonormal basis Q = [Q_cca | Q_perp] obtained by completing the
29 CCA-identified raw-space H directions to a full basis of R^4998
(change of basis ONLY — no H information is discarded).

    min_W ||Y - Z_c W||^2 + W^T (alpha_base * diag(lam0)) W

with lam0 = [1_G(4998), 1+delta_1..29, 1_perp(4969)], delta_k = rho_k^2.
The uniform model is the exact delta = 0 special case (basis-invariance
gate against ordinary full ridge).  No mixing ratio, no rho threshold,
no gamma/rank/tau search anywhere.

PCA (95%-variance, train-only) and CCA are used ONLY to discover the 29
directions; G keeps all 4998 original features in the final design.

Dual (n-space) solver
---------------------
With S = Z_c diag(1/lam0) Z_c^T (n x n; p up to 9996 but only n x n
systems are ever diagonalized):

    W       = diag(1/lam0) Z_c^T (alpha I_n + S)^{-1} Y_c
    hat     = S (alpha I + S)^{-1},   df = sum_k theta_k/(theta_k+alpha)
    RSS     = sum_k (alpha/(theta_k+alpha))^2 ||v_k^T Y_c||^2
    GCV     = n * RSS / (n - df)^2

where theta/v is the eigendecomposition of S.  ONE eigendecomposition
serves the whole alpha grid.  No p x p system, no explicit inverse,
no iterative optimizer.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "GCV_GRID",
    "adaptive_delta",
    "canonical_direction_matrix",
    "unit_directions",
    "complete_orthonormal_basis",
    "DualGeneralizedRidge",
    "brute_force_generalized_ridge",
]

GCV_GRID = np.logspace(-4, 4, 81)
_EPS = np.finfo(np.float64).eps


# ----------------------------------------------------------------------
# canonical direction mapping  (H_low space -> raw H_full space)
# ----------------------------------------------------------------------
def canonical_direction_matrix(comp_H: np.ndarray, ok_mask: np.ndarray,
                               hsd: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Raw-space directions D (d_H x K) with

        canonical score_k = (H - mu_H) @ D[:, k] + const_k.

    Derivation: H_low = ((H - mu_H) * ok) @ comp_H.T (RankPCA.transform_low,
    mean-centering only, dead columns zeroed); CCAFit whitens H_low with
    (H_low - hmu)/hsd and projects via B; folding the affine parts gives
    D = diag(ok) @ comp_H.T @ diag(1/hsd) @ B.  Per-direction constants
    are removed by centering.
    """
    comp_H = np.asarray(comp_H, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    hsd_safe = np.where(np.asarray(hsd) > 1e-12, hsd, 1.0)
    D = comp_H.T @ (B / hsd_safe[:, None])          # (d_H, K)
    D = D * np.asarray(ok_mask, dtype=np.float64)[:, None]
    return D


def unit_directions(D: np.ndarray, tol_rel: float = 1e-8) -> np.ndarray:
    """Normalize columns to unit norm; drop numerically vanishing ones."""
    D = np.asarray(D, dtype=np.float64)
    norms = np.linalg.norm(D, axis=0)
    keep = norms > max(norms.max(initial=1.0), 1.0) * tol_rel
    return D[:, keep] / norms[keep][None, :]


def adaptive_delta(rho: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """delta_k = gamma * rho_k^2  (predeclared shrinkage increments)."""
    rho = np.asarray(rho, dtype=np.float64)
    return gamma * rho ** 2


# ----------------------------------------------------------------------
# orthonormal basis completion (change-of-basis only)
# ----------------------------------------------------------------------
def complete_orthonormal_basis(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q_cca = orthonormal basis of span(U) (columns aligned with U's
    leading spans via unpivoted QR), Q_perp completing R^d.

    Full QR factorization (LAPACK reflectors), deterministic sign
    convention (largest-|.| entry of every column made positive).
    """
    d, K = U.shape
    Qfull, _ = np.linalg.qr(U, mode="complete")
    Q_cca, Q_perp = Qfull[:, :K], Qfull[:, K:]
    for M in (Q_cca, Q_perp):
        for j in range(M.shape[1]):
            i = int(np.argmax(np.abs(M[:, j])))
            if M[i, j] < 0:
                M[:, j] *= -1.0
    return Q_cca, Q_perp


# ----------------------------------------------------------------------
# dual generalized ridge classifier (per-coordinate lam0 vector)
# ----------------------------------------------------------------------
class DualGeneralizedRidge:
    """One-hot generalized ridge on the full design, solved in dual form.

    Z: (n, p), p up to 9996.  lam0: (p,) alpha-unit penalty factors for
        W^T (c * diag(lam0)) W;  None or all-ones -> uniform ridge.
    The single selected scalar c = alpha_base comes from train-only GCV:
        GCV(c) = n * RSS(c) / (n - df(c))^2,  exact df from the hat matrix.
    """

    def __init__(self, lam0: np.ndarray | None = None,
                 alpha_grid: np.ndarray = GCV_GRID):
        self.lam0 = None if lam0 is None else np.asarray(lam0, dtype=np.float64)
        self.alpha_grid = np.asarray(alpha_grid, dtype=np.float64)
        self.classes_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.diagnostics: dict = {}

    # ------------------------------------------------------------------
    def fit(self, Z: np.ndarray, y: np.ndarray) -> "DualGeneralizedRidge":
        Z = np.asarray(Z, dtype=np.float64)
        n, p = Z.shape
        assert np.isfinite(Z).all(), "non-finite design matrix"
        lam0 = (np.ones(p) if self.lam0 is None
                else np.broadcast_to(self.lam0, (p,)).astype(np.float64).copy())
        assert lam0.shape == (p,)
        assert (lam0 > 0).all(), "lam0 must be positive"
        uniform = bool(np.allclose(lam0, 1.0, rtol=0, atol=1e-12))

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        Y = np.zeros((n, len(self.classes_)))
        Y[np.arange(n), y_idx] = 1.0
        self.y_mean_ = Y.mean(axis=0, keepdims=True)
        Yc = Y - self.y_mean_

        self.mu_ = Z.mean(axis=0, keepdims=True)
        Zc = Z - self.mu_
        ZL = Zc / lam0[None, :]                    # Z diag(1/lam0)
        S = ZL @ Zc.T                              # (n, n)
        S = (S + S.T) / 2.0

        theta, V = np.linalg.eigh(S)
        theta = np.clip(theta, 0.0, None)
        order = np.argsort(theta)[::-1]
        theta, V = theta[order], V[:, order]
        self.theta_ = theta
        self._fit_cache = {"Zc": Zc, "ZL": ZL, "Yc": Yc, "theta": theta,
                           "V": V, "S": S}
        rank_S = int((theta > max(theta[0], 1.0) * n * _EPS).sum())
        ytilde2 = ((V.T @ Yc) ** 2).sum(axis=1)    # (n,)

        # ---- train-only GCV over the whole grid (vectorized) ----------
        c = self.alpha_grid[None, :]
        hat = theta[:, None] / (theta[:, None] + c)
        df = hat.sum(axis=0)
        rss = ((c / (theta[:, None] + c)) ** 2 * ytilde2[:, None]).sum(axis=0)
        denom = np.maximum(n - df, 1e-12)
        gcv = n * rss / denom ** 2
        best = int(np.argmin(gcv))
        self.alpha_base_ = float(self.alpha_grid[best])

        # ---- solution at the selected alpha ---------------------------
        cc = self.alpha_base_
        u = V @ ((V.T @ Yc) / (theta + cc)[:, None])   # (alpha I + S)^{-1} Yc
        self.coef_ = (ZL.T @ u)[:, None] if Yc.shape[1] == 1 \
            else ZL.T @ u                              # diag(1/lam0) Z^T u
        self.coef_ = ZL.T @ u
        self.intercept_ = self.y_mean_ - self.mu_ @ self.coef_

        # ---- numerical diagnostics at the selected alpha --------------
        eig_pos = theta[theta > 0]
        A = S + cc * np.eye(n)
        self.diagnostics = {
            "n": int(n), "p": int(p),
            "uniform_lam0": uniform,
            "rank_Z_train": rank_S,
            "alpha_base": self.alpha_base_,
            "df": float(df[best]), "rss": float(rss[best]),
            "gcv_min": float(gcv[best]),
            "gcv_grid": self.alpha_grid.tolist(),
            "gcv_values": gcv.tolist(),
            "eig_S_max": float(theta[0]),
            "eig_S_min_nonzero": float(eig_pos.min()) if eig_pos.size else 0.0,
            "min_eig_regularized": float(theta[-1] + cc),
            "max_eig_regularized": float(theta[0] + cc),
            "cond_regularized": float((theta[0] + cc) /
                                      max(eig_pos.min() + cc, _EPS)),
            "rss_check": float(((Yc - S @ u) ** 2).sum()),
        }
        try:
            np.linalg.cholesky(A)
            self.diagnostics["cholesky_ok"] = True
            self.diagnostics["fallback_used"] = False
        except np.linalg.LinAlgError:
            self.diagnostics["cholesky_ok"] = False
            self.diagnostics["fallback_used"] = True
        self.diagnostics["finite_coef"] = bool(np.isfinite(self.coef_).all())
        return self

    # ------------------------------------------------------------------
    def refit_at(self, c: float) -> None:
        """Re-solve the dual system at a given alpha (no refitting of any
        representation object; uses the cached train-side quantities).
        Used for the train+val final-fit arm with alpha FROZEN at the
        train-only GCV value."""
        assert self._fit_cache is not None, "fit() first"
        ZL = self._fit_cache["ZL"]
        Yc = self._fit_cache["Yc"]
        theta, V = self._fit_cache["theta"], self._fit_cache["V"]
        n = ZL.shape[0]
        u = V @ ((V.T @ Yc) / (theta + c)[:, None])
        self.coef_ = ZL.T @ u
        self.intercept_ = self.y_mean_ - self.mu_ @ self.coef_
        self.alpha_base_ = float(c)
        self.diagnostics["n_fit_rows"] = int(n)
        self.diagnostics["refit_alpha_frozen"] = True

    def decision_function(self, Z: np.ndarray) -> np.ndarray:
        """Z @ W + intercept, with intercept = y_mean - mu_train @ W
        (standard centered-ridge convention; the centering correction is
        applied exactly once)."""
        return np.asarray(Z, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(Z), axis=1)]


# ----------------------------------------------------------------------
# brute-force primal reference (small p only, for unit tests)
# ----------------------------------------------------------------------
def brute_force_generalized_ridge(Z: np.ndarray, y: np.ndarray,
                                  lam0: np.ndarray | None, c: float
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct primal solve of (Z_c^T Z_c + c*diag(lam0)) W = Z_c^T Y_c.

    Returns (W, intercept, mu).  Only for small p (unit tests).
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, p = Z.shape
    classes, y_idx = np.unique(y, return_inverse=True)
    Y = np.zeros((n, len(classes)))
    Y[np.arange(n), y_idx] = 1.0
    y_mean = Y.mean(axis=0, keepdims=True)
    Yc = Y - y_mean
    mu = Z.mean(axis=0, keepdims=True)
    Zc = Z - mu
    lam = np.ones(p) if lam0 is None else np.asarray(lam0, dtype=np.float64)
    A = Zc.T @ Zc + c * np.diag(lam)
    W = np.linalg.solve(A, Zc.T @ Yc)
    return W, y_mean - mu @ W, mu
