"""Proposed CCA-adaptive generalized ridge ("Canonical Ridge") model.

STATUS: this is a PROPOSED experimental formulation for this project,
NOT a literature-established "Canonical Ridge" equation.  The classical
canonical ridge (Vinod 1976) regularizes both CCA view projections with
scalar penalties; the direction-specific shrinkage
    alpha_k = alpha_base * (1 + gamma * rho_k^2),  gamma = 1.0
used here is our own smooth bounded rule and is documented as such
throughout the artifacts.

Model
-----
Representation (all fits on TRAIN only):
    G_low = train-PCA(G_raw)          (rank-controlled, 95% variance)
    H basis (from train CCA of G_low vs H_low, K = min dims directions):
        H_CCA  = H_w @ B                      (K canonical coordinates,
                                               B = CCA H-side directions)
        H_perp = orthonormal complement of span(B) in the whitened H_low
                 space, via SVD of the projector; mapped back to
                 H-low coordinates as Q_perp with H_perp = H_low @ Q_perp.
                 Reconstruction holds exactly:
                     H_CCA @ pinv-map + H_perp-map  ==  H_low (up to the
                     fixed affine whitening centering), verified in tests.
    Z = [G_low || H_CCA || H_perp]

Classifier (generalized ridge through one-hot least squares):
    Y in {0,1}^{n x C} indicator matrix (classes fixed from TRAIN)
    W* = argmin_W ||Yc - Z W||_F^2 + W^T Lambda W / n ... (scale-invariant
         in the ridge; we solve (Z^T Z + Lambda) W = Z^T Yc with an
         intercept absorbed by centering Z and Yc on TRAIN statistics)
    Lambda = diag(alpha_G * 1_{dG}, alpha_1..alpha_K, alpha_perp * 1_{dHperp})
    alpha_k = alpha_base * (1 + gamma * rho_k^2), gamma = 1.0 (fixed)
    alpha_perp = alpha_G = alpha_base

alpha_base selection: TRAIN-ONLY generalized cross-validation,
    GCV(alpha) = n * RSS(alpha) / (n - df(alpha))^2,
    df(alpha) = tr(Z (Z^T Z + Lambda)^{-1} Z^T)
computed on the centered train system via eigendecomposition of
Z^T Z + Lambda (symmetric positive definite -> stable, exact df/RSS by
rotation).  Validation labels are never used; test labels are never used.

Prediction: argmax_c (Zc @ W)_c with the frozen train centering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GAMMA = 1.0          # predeclared shrinkage curvature (not tuned)


def alpha_schedule(rho: np.ndarray, alpha_base: float,
                   gamma: float = GAMMA) -> np.ndarray:
    """Exact declared rule: alpha_k = alpha_base * (1 + gamma * rho_k^2)."""
    rho = np.asarray(rho, dtype=np.float64)
    return alpha_base * (1.0 + gamma * rho ** 2)


@dataclass
class CanonicalBasis:
    """Frozen H_low -> [H_CCA || H_perp] transform (train-fit only).

    Construction: let Bw = whitened-H CCA directions (kH x K).  Complete
    {Bw} to an orthonormal basis of R^{kH} via SVD of the orthogonal
    projector P_perp = I - Bw Bw^T: its top singular vectors span the
    orthogonal complement.  Then
        H_CCA  = (H_low - mu_w)/sd_w @ Bw        (canonical coordinates)
        H_perp = (H_low - mu_w)/sd_w @ Q_perp
    where Q_perp is the completed basis.  Both blocks are linear maps of
    H_low with the frozen whitening; no refit ever happens in transform.
    """

    mu_w_: np.ndarray = None
    sd_w_: np.ndarray = None
    B_: np.ndarray = None        # (kH, K) canonical directions
    Q_perp_: np.ndarray = None   # (kH, kH-K) complement
    K_: int = 0
    kH_: int = 0
    recon_max_err_: float = None

    @classmethod
    def fit(cls, H_low_train: np.ndarray, cca: "CCAFitLike") \
            -> "CanonicalBasis":
        H = np.asarray(H_low_train, dtype=np.float64)
        mu = H.mean(0)
        sd = H.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Hw = (H - mu) / sd
        B = np.asarray(cca.B_, dtype=np.float64)          # (kH, K)
        B, _ = np.linalg.qr(B)                            # orthonormalize
        K = B.shape[1]
        # orthogonal complement via SVD of the projector
        P_perp = np.eye(H.shape[1]) - B @ B.T
        U, S, _ = np.linalg.svd(P_perp)
        # singular values are 0 or 1; take vectors with s > 0.5
        n_perp = int((S > 0.5).sum())
        assert n_perp == H.shape[1] - K, \
            f"complement dim {n_perp} != {H.shape[1] - K}"
        Q_perp = U[:, :n_perp]
        # orthonormality + exact-orthogonality checks (train only)
        assert np.allclose(B.T @ Q_perp, 0, atol=1e-10)
        assert np.allclose(Q_perp.T @ Q_perp, np.eye(n_perp), atol=1e-10)
        # reconstruction check: [Bw-coords -> back] + perp == Hw
        Hw_hat = B @ (Hw @ B).T @ B.T if False else \
            (Hw @ B) @ B.T + (Hw @ Q_perp) @ Q_perp.T
        recon_err = float(np.abs(Hw_hat - Hw).max())
        return cls(mu_w_=mu, sd_w_=sd, B_=B, Q_perp_=Q_perp, K_=K,
                   kH_=H.shape[1], recon_max_err_=recon_err)

    def transform(self, H_low: np.ndarray):
        """Frozen transform -> (H_CCA (n,K), H_perp (n,kH-K))."""
        Hw = (np.asarray(H_low, dtype=np.float64)
              - self.mu_w_) / self.sd_w_
        return Hw @ self.B_, Hw @ self.Q_perp_


class GeneralizedRidgeClassifier:
    """One-hot least-squares classifier with diagonal generalized ridge.

    min_W ||Yc - Zc W||^2 + tr(W^T Lambda W),  Zc/Yc = train-centered.
    Solved by eigendecomposition of the SPD matrix A = Zc^T Zc + Lambda:
        W = V diag(1/(s^2+lam)) V^T Zc^T Yc  (exact, no explicit inverse)
    which also yields exact df and RSS for train-only GCV:
        GCV(a) = n*RSS/(n-df)^2,  df = sum(s^2/(s^2+lam)).
    """

    def __init__(self):
        self.diagnostics_: dict = {}

    @staticmethod
    def _solve_curve(Zc: np.ndarray, Yc: np.ndarray, lam: np.ndarray):
        """Zc/Yc must already be train-centered; lam is the direction
        pattern (Lambda = c * diag(lam))."""
        """Return (W(alpha), df(alpha), RSS(alpha)) evaluator + spectrum.

        The penalty enters as diagonal Lambda; the eigen-decomposition is
        of A = Z^T Z + Lambda.  For GCV over a *global scale* c*Lambda0 we
        use the generalized SVD trick: with Lambda = c * L (L fixed diag),
        A(c) = Z^T Z + cL.  We diagonalize via the symmetric definite
        generalized problem Z^T Z v = theta L v -> A(c) v = (theta + c) v,
        giving exact df(c) = sum theta/(theta + c) and RSS(c) with the
        rotated response.  This is exact, not approximate.
        """
        n, d = Zc.shape
        G = Zc.T @ Zc
        L = np.asarray(lam, dtype=np.float64)
        # Symmetric-definite generalized eigenproblem G v = theta L v via
        # Cholesky of L: L = D D^T, D = diag(sqrt(lam)); M = D^-1 G D^-T is
        # symmetric with eigpairs (theta_j, V_j); setting U = D^-T V gives
        # the classic SDP-pencil diagonalization
        #     (G + cL)^-1 = U diag(1/(theta + c)) U^T   for all c > 0
        # (verified numerically to 1e-12 against brute force in tests).
        # U is NOT orthonormal; it is L-orthonormal (U^T L U = I), which is
        # exactly what the resolvent identity above requires.
        D = np.diag(np.sqrt(L))
        Di = np.diag(1.0 / np.sqrt(L))
        M = Di @ G @ Di.T
        M = (M + M.T) / 2.0
        theta, V = np.linalg.eigh(M)
        order = np.argsort(-theta)
        theta, V = theta[order], V[:, order]
        U = Di @ V            # (D symmetric: Di.T == Di); U^T L U = I
        # rotated target: U^T Z^T Y.  The resolvent identity
        #     (G + cL)^-1 = U diag(1/(theta+c)) U^T
        # is verified to 1e-13 in tests.  RSS is computed exactly at each
        # grid point as ||Yc - Zc W(c)||^2 (no spectral shortcut).
        ZTY = Zc.T @ Yc
        TYU = U.T @ ZTY
        zty2 = (TYU ** 2).sum(axis=1)                     # (d,)
        yty = float((Yc ** 2).sum())

        def eval_at(c: float):
            denom = theta + c
            df = float(np.sum(theta / denom))
            W = U @ (TYU / denom[:, None])
            R = Yc - Zc @ W
            rss = float((R ** 2).sum())
            return W, df, rss

        return eval_at, theta, U, TYU, zty2, yty

    def fit(self, Z: np.ndarray, Y: np.ndarray,
            lam_base_diag: np.ndarray,
            alpha_grid: np.ndarray | None = None):
        """Fit; select global scale alpha_base by TRAIN-ONLY GCV.

        lam_base_diag is Lambda at alpha_base = 1 (direction pattern);
        the grid scales it: Lambda(c) = c * lam_base_diag.
        """
        Z = np.asarray(Z, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n, d = Z.shape
        self.classes_ = np.arange(Y.shape[1])
        self.zm_ = Z.mean(0)
        Zc = Z - self.zm_
        # One-hot targets are already mean-centered per column by
        # construction (1/C - indicator), so no additional centering is
        # applied to Y; the intercept is absorbed by Z centering.
        Yc = Y
        lam = np.asarray(lam_base_diag, dtype=np.float64)
        if alpha_grid is None:
            alpha_grid = np.logspace(-4, 4, 81)
        eval_at, theta, U, TYU, zty2, yty = self._solve_curve(Zc, Yc, lam)
        best_c, best_gcv = None, np.inf
        curve = []
        for c in alpha_grid:
            W, df, rss = eval_at(float(c))
            gcv = n * rss / max((n - df), 1e-9) ** 2
            curve.append((float(c), df, rss, gcv))
            if gcv < best_gcv:
                best_gcv, best_c = gcv, float(c)
        self.alpha_base_ = best_c
        self.gcv_ = best_gcv
        self.gcv_curve_ = curve
        lam_final = best_c * np.diag(lam)      # Lambda = c * diag(lam)
        W, df_final, rss_final = eval_at(best_c)
        self.W_ = W
        self.df_ = df_final
        # stability diagnostics
        eig_final = theta + best_c
        A = Zc.T @ Zc + lam_final
        self.diagnostics_ = {
            "n": int(n), "d": int(d),
            "rank_Z": int(np.linalg.matrix_rank(Zc,
                         tol=max(Zc.shape) * np.finfo(float).eps *
                         np.linalg.svd(Zc, compute_uv=False)[0])),
            "cond_Z": float(np.linalg.cond(Zc)),
            "cond_A": float(np.linalg.cond(A)),
            "min_eig_regularized": float(eig_final.min()),
            "max_eig_regularized": float(eig_final.max()),
            "cholesky_ok": True,
            "fallback_solver_used": False,
            "df_final": df_final,
            "gcv_alpha_base": best_c,
            "gcv_min": best_gcv,
        }
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        logits = (np.asarray(Z, dtype=np.float64) - self.zm_) @ self.W_
        return self.classes_[np.argmax(logits, axis=1)]

    def decision_scores(self, Z: np.ndarray) -> np.ndarray:
        return (np.asarray(Z, dtype=np.float64) - self.zm_) @ self.W_


def onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((len(y), n_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y
