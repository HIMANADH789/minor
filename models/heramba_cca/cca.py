"""Numerically stable CCA + shared-subspace machinery for HERAMBA-CCA.

Design (matching the Haptics seed-42 reality: N_train = 132 << d = 4998):

  Pre-CCA reduction: deterministic, label-free, TRAIN-ONLY truncated PCA
  to the numerical rank of each representation (energy 1-1e-10, i.e. the
  SVD numerical-rank rule).  For G (eff. rank ~6) and H (eff. rank ~67)
  this keeps essentially all energy in a handful of components, so CCA is
  computed on k_G x k_H small matrices and is exact, not approximate.

  CCA: solve via whitening + SVD of the regularized cross-covariance
  (Numerical Recipes / sklearn-style formulation) on the reduced spaces.

  Shared-subspace rule (PREDECLARED, never tuned on val/test):
      component j of the CCA solution is SHARED iff
      rho_j >= TAU_SHARED with TAU_SHARED = 0.5
      (a canonical pair whose two reduced views correlate >= 0.5 on
      TRAIN represents a linear direction common to both banks).
      The subspace is spanned by the G-side canonical direction vectors.

  H_unique: H residualized against the shared G-side subspace by
  ordinary least-squares projection (fitted on TRAIN, frozen, then
  applied identically to val/test).  Orthogonality of the residual to
  the shared directions is verified numerically on TRAIN.

No label of any kind enters any fit here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def numerical_rank(S: np.ndarray, energy: float = 1 - 1e-10) -> int:
    """Rank of S by cumulative singular-value energy (>= energy)."""
    s = np.linalg.svd(S - S.mean(0), compute_uv=False)
    if s.size == 0 or s[0] <= 0:
        return 0
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(cum, energy) + 1)


def effective_rank(S: np.ndarray) -> float:
    """Entropy-based effective rank (existing repository definition)."""
    X = np.asarray(S, dtype=np.float64)
    X = X - X.mean(0)
    # subsample columns if extremely wide (same convention as the
    # repository's inference analysis: stride keeps the spectrum stable)
    if X.shape[1] > 4000:
        X = X[:, ::4]
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def _standardize_fit(X: np.ndarray):
    mu = X.mean(0)
    sd = X.std(0)
    ok = np.isfinite(sd) & (sd > 1e-12)
    return mu, sd, ok


def _standardize_apply(X: np.ndarray, mu, sd, ok):
    Z = (X - mu) / np.where(ok, sd, 1.0)
    Z[:, ~ok] = 0.0
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------
# truncated PCA (label-free, train-only, deterministic)
# ----------------------------------------------------------------------
@dataclass
class TrainPCA:
    """Truncated PCA to numerical rank, fitted on TRAIN only."""

    energy: float = 1 - 1e-10
    mu_: np.ndarray | None = None
    comp_: np.ndarray | None = None   # (k, d) components
    var_ratio_: np.ndarray | None = None
    k_: int = 0
    full_dim_: int = 0

    def fit(self, X: np.ndarray) -> "TrainPCA":
        X = np.asarray(X, dtype=np.float64)
        self.full_dim_ = X.shape[1]
        mu, sd, ok = _standardize_fit(X)
        Z = _standardize_apply(X, mu, sd, ok)
        # economy SVD of the (small-N x wide-d) matrix
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        s2 = S ** 2
        tot = s2.sum()
        cum = np.cumsum(s2) / max(tot, 1e-300)
        k = int(np.searchsorted(cum, self.energy) + 1)
        k = max(1, min(k, len(S)))
        self.mu_, self.sd_, self.ok_ = mu, sd, ok
        self.comp_ = Vt[:k]
        self.k_ = k
        self.var_ratio_ = (s2[:k] / max(tot, 1e-300))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = _standardize_apply(np.asarray(X, dtype=np.float64),
                               self.mu_, self.sd_, self.ok_)
        return Z @ self.comp_.T


# ----------------------------------------------------------------------
# CCA on the reduced views
# ----------------------------------------------------------------------
@dataclass
class CCAFit:
    """Canonical correlation analysis fitted on reduced TRAIN views."""

    rho_: np.ndarray | None = None          # canonical correlations (desc)
    A_: np.ndarray | None = None            # G-side directions (kG x k)
    B_: np.ndarray | None = None            # H-side directions (kH x k)
    Gwhite_mu_: np.ndarray | None = None
    Gwhite_sd_: np.ndarray | None = None
    Hwhite_mu_: np.ndarray | None = None
    Hwhite_sd_: np.ndarray | None = None
    k_shared_: int = 0
    tau_shared_: float = 0.5
    meta_: dict = field(default_factory=dict)

    # -- fitting -------------------------------------------------------
    def fit(self, G_red: np.ndarray, H_red: np.ndarray,
            tau_shared: float = 0.5) -> "CCAFit":
        G = np.asarray(G_red, dtype=np.float64)
        H = np.asarray(H_red, dtype=np.float64)
        n = G.shape[0]
        # whiten each reduced view (train statistics)
        gmu, gsd = G.mean(0), G.std(0)
        hmu, hsd = H.mean(0), H.std(0)
        gsd = np.where(gsd > 1e-12, gsd, 1.0)
        hsd = np.where(hsd > 1e-12, hsd, 1.0)
        Gw = (G - gmu) / gsd
        Hw = (H - hmu) / hsd
        self.Gwhite_mu_, self.Gwhite_sd_ = gmu, gsd
        self.Hwhite_mu_, self.Hwhite_sd_ = hmu, hsd

        kG, kH = Gw.shape[1], Hw.shape[1]
        # SVD of the cross-covariance: C = U S Vt with C = Gw^T Hw / n
        U, S, Vt = np.linalg.svd(Gw.T @ Hw / n, full_matrices=False)
        # canonical correlations with ridge-stabilized denominators
        lam = 1e-8
        dG = np.sqrt((U ** 2).sum(0) * (S ** 2) / n + lam)  # placeholder
        # Exact canonical correlations: rho_j = s_j / (sd_gj * sd_hj)
        # where sd_gj, sd_hj are the stds of the projected unit vectors.
        # Simplest exact route: project and normalize.
        A = U                        # kG x k
        B = Vt.T                     # kH x k
        ag = Gw @ A                  # n x k
        bh = Hw @ B
        sg = ag.std(0)
        sh = bh.std(0)
        sg = np.where(sg > 1e-12, sg, 1e-12)
        sh = np.where(sh > 1e-12, sh, 1e-12)
        rho = S / (sg * sh)
        rho = np.clip(rho / np.max(np.clip(rho, 1e-12, None)), -1, 1) \
            if False else np.clip(rho, -1, 1)
        order = np.argsort(-rho)
        rho, A, B = rho[order], A[:, order], B[:, order]
        ag, bh = ag[:, order], bh[:, order]
        # sign convention: positive correlation with the H view
        for j in range(A.shape[1]):
            if rho[j] < 0:
                B[:, j] *= -1
                bh[:, j] *= -1
        rho = np.abs(rho)

        self.rho_ = rho
        self.A_, self.B_ = A, B
        self.tau_shared_ = tau_shared
        self.k_shared_ = int((rho >= tau_shared).sum())
        self.meta_ = {
            "n_train": int(n), "k_G_reduced": int(kG),
            "k_H_reduced": int(kH), "n_components": int(len(rho)),
            "tau_shared": tau_shared,
        }
        return self

    # -- projections ---------------------------------------------------
    def project_G(self, G_red: np.ndarray) -> np.ndarray:
        """Canonical scores of a reduced G view (n x k)."""
        G = np.asarray(G_red, dtype=np.float64)
        return ((G - self.Gwhite_mu_) / self.Gwhite_sd_) @ self.A_

    def project_H(self, H_red: np.ndarray) -> np.ndarray:
        """Canonical scores of a reduced H view (n x k)."""
        H = np.asarray(H_red, dtype=np.float64)
        return ((H - self.Hwhite_mu_) / self.Hwhite_sd_) @ self.B_


# ----------------------------------------------------------------------
# shared-subspace residualization (the H_unique transform)
# ----------------------------------------------------------------------
@dataclass
class SharedResidualizer:
    """Remove the shared G-side canonical subspace from H.

    Fitted on TRAIN only.  The shared directions are the G-side canonical
    vectors A_s (kG x k_shared) expressed in the *reduced-G* space; for a
    raw H vector we regress the whitened raw H on the *raw* reconstructed
    shared directions computed through the cross-view mapping.
    """

    direction_raw_: np.ndarray | None = None   # (d_H_raw) shared directions
    mu_: np.ndarray | None = None
    coef_: np.ndarray | None = None            # (d_shared) OLS coefs

    def fit(self, H_raw: np.ndarray, H_shared_scores: np.ndarray) \
            -> "SharedResidualizer":
        """Regress each raw H feature on the shared scores (TRAIN only).

        H_shared_hat = scores @ coef_; residual = H_raw - H_shared_hat.
        This removes from every raw H feature the part linearly explained
        by the shared canonical scores (which are, by construction, the
        components of H aligned with the shared G subspace).
        """
        H = np.asarray(H_raw, dtype=np.float64)
        S = np.asarray(H_shared_scores, dtype=np.float64)
        self.mu_ = S.mean(0)
        Sc = S - self.mu_
        # OLS: coef = (Sc^T Sc)^-1 Sc^T H  (tall, small k_shared x d_H)
        coef, *_ = np.linalg.lstsq(Sc, H, rcond=None)
        self.coef_ = coef
        self.S_center_ = self.mu_
        self._H_center = H.mean(0)          # raw-H mean, frozen
        # orthogonality check on train
        resid = H - (Sc @ coef + H.mean(0))
        C = np.corrcoef(resid.T, Sc.T)[:resid.shape[1], resid.shape[1]:]
        self.train_max_abs_corr_ = float(np.nanmax(np.abs(C)))
        self.train_mean_abs_corr_ = float(np.nanmean(np.abs(C)))
        return self

    def transform(self, H_raw: np.ndarray, H_shared_scores: np.ndarray) \
            -> np.ndarray:
        """Apply the frozen residualization to any split."""
        H = np.asarray(H_raw, dtype=np.float64)
        S = np.asarray(H_shared_scores, dtype=np.float64) - self.mu_
        H_shared_hat = S @ self.coef_ + self._H_center
        return H - H_shared_hat


# ----------------------------------------------------------------------
# one frozen end-to-end pipeline (fit on TRAIN, apply to val/test)
# ----------------------------------------------------------------------
@dataclass
class HerambaCCAProjection:
    """Complete frozen transform: raw H -> H_unique (and diagnostics)."""

    tau_shared: float = 0.5
    pca_energy: float = 1 - 1e-10

    def fit(self, G_train: np.ndarray, H_train: np.ndarray) \
            -> "HerambaCCAProjection":
        self.pca_G_ = TrainPCA(self.pca_energy).fit(G_train)
        self.pca_H_ = TrainPCA(self.pca_energy).fit(H_train)
        G_red = self.pca_G_.transform(G_train)
        H_red = self.pca_H_.transform(H_train)
        self.cca_ = CCAFit().fit(G_red, H_red, self.tau_shared)
        k = self.cca_.k_shared_
        # shared scores of H: canonical scores restricted to shared comps
        H_shared_scores = self.cca_.project_H(H_red)[:, :k]
        # residualize in the reduced-H space: H_unique_red =
        #   H_red - Reconstruct_reduced(shared scores)
        Sc = H_shared_scores
        # regress reduced-H on shared canonical scores (train only)
        coef, *_ = np.linalg.lstsq(Sc - Sc.mean(0), H_red, rcond=None)
        self._red_coef_ = coef
        self._sc_mean_ = Sc.mean(0)
        resid = H_red - ((Sc - self._sc_mean_) @ coef + H_red.mean(0))
        # numerical orthogonality check: residual (reduced-H space) vs the
        # shared canonical scores (the OLS regressors) -- orthogonal by
        # construction; verified to machine precision.
        Scc = Sc - Sc.mean(0)
        rc = resid - resid.mean(0)
        denom_s = np.sqrt((Scc ** 2).sum(0))
        denom_r = np.sqrt((rc ** 2).sum(0))
        idx_r = np.where(denom_r > 1e-8)[0]   # ignore zero-variance dirs
        idx_s = np.where(denom_s > 1e-8)[0]
        if idx_r.size and idx_s.size:
            Ccos = (rc[:, idx_r].T @ Scc[:, idx_s]) / \
                np.outer(denom_r[idx_r], denom_s[idx_s])
        else:
            Ccos = np.zeros((1, 1))
        self.orth_max_abs_corr_ = float(np.nanmax(np.abs(Ccos)))
        self.orth_mean_abs_corr_ = float(np.nanmean(np.abs(Ccos)))
        self.k_shared_ = k
        return self

    # -- transforms ----------------------------------------------------
    def transform_shared_scores(self, G: np.ndarray, H: np.ndarray):
        G_red = self.pca_G_.transform(G)
        H_red = self.pca_H_.transform(H)
        return self.cca_.project_G(G_red), self.cca_.project_H(H_red)

    def transform_H_unique(self, H: np.ndarray) -> np.ndarray:
        """Frozen H -> H_unique (raw-dim residual in reduced-H space)."""
        H_red = self.pca_H_.transform(H)
        k = self.cca_.k_shared_
        Sc = self.cca_.project_H(H_red)[:, :k]
        return H_red - ((Sc - self._sc_mean_) @ self._red_coef_
                        + H_red.mean(0))

    def summary(self) -> dict:
        rho = self.cca_.rho_
        return {
            "tau_shared": self.tau_shared,
            "G_full_dim": self.pca_G_.full_dim_,
            "H_full_dim": self.pca_H_.full_dim_,
            "G_reduced_dim": self.pca_G_.k_,
            "H_reduced_dim": self.pca_H_.k_,
            "G_energy_kept": float(self.pca_G_.var_ratio_.sum()),
            "H_energy_kept": float(self.pca_H_.var_ratio_.sum()),
            "n_cca_components": int(len(rho)),
            "n_shared_components": self.k_shared_,
            "canonical_correlations": [round(float(r), 6) for r in rho],
            "shared_correlations": [round(float(r), 6)
                                    for r in rho[:self.k_shared_]],
            "cum_sq_corr_shared": round(
                float(np.sum(rho[:self.k_shared_] ** 2)), 6),
            "orth_max_abs_corr_Hunique_vs_shared":
                round(self.orth_max_abs_corr_, 6),
            "orth_mean_abs_corr_Hunique_vs_shared":
                round(self.orth_mean_abs_corr_, 6),
            "cca_meta": self.cca_.meta_,
        }


def save_summary(proj: HerambaCCAProjection, path: str):
    with open(path, "w") as f:
        json.dump(proj.summary(), f, indent=1)
