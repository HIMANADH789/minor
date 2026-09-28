"""TURS-MGB dual linear-kernel Ridge readout (Phase 7).

Because the concatenated feature dimension d (up to ~4M x 3 stats) is much
larger than n (10^3), the ridge is solved in the DUAL:

    K = Z Z^T                      [n x n]
    alpha = (K + lambda I)^{-1} Y  (one-vs-rest; Y is one-hot [n, C])
    f(x) = k(x, X_train) alpha,  k(x, X_train) = Z x Z_train^T

This is EXACTLY equivalent to primal linear-kernel Ridge
    beta = (Z^T Z + lambda I)^{-1} Z^T Y
with predictions Z_test beta (Phase 29 verifies this numerically).

Terminology: this is a "dual linear-kernel Ridge on fixed multi-geometry
features" — NOT a separately engineered nonlinear (RBF) kernel.

Numerical stability: torch.linalg.cholesky + cholesky_solve on the PSD
system (K + lambda I); never an explicit matrix inverse. The Gram matrix
and alpha are cached for inference (Phase 30).

Group contribution (Phase 17): because the primal beta is recoverable as
    beta = Z_train^T alpha,
per-geometry logit contributions are f_j(x) = Z_j(x) @ beta_j exactly.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _onehot(y, C, dtype=torch.float32):
    return F.one_hot(torch.as_tensor(y).long(), C).to(dtype)


class DualRidge:
    """Dual linear-kernel Ridge classifier (one-vs-rest logits)."""

    def __init__(self, n_classes, lam=1.0, eps=1e-6, device=None):
        self.C = int(n_classes)
        self.lam = float(lam)
        self.eps = float(eps)
        self.device = device or torch.device("cpu")
        self.alpha = None          # [n, C]
        self.Z_train = None        # [n, d] (kept on device; needed for k(x))
        self.beta = None           # [d, C] primal solution (recoverable)

    # ------------------------------------------------------------ fit
    def fit(self, Z_train, y_train, backend="cholesky"):
        Z = torch.as_tensor(Z_train).float().to(self.device)
        Y = _onehot(y_train, self.C).to(self.device)
        K = Z @ Z.T                                             # [n, n]
        A = K + (self.lam + self.eps) * torch.eye(K.shape[0],
                                                 device=self.device)
        if backend == "cholesky":
            try:
                L = torch.linalg.cholesky(A)
                self.alpha = torch.cholesky_solve(Y, L)
            except torch._C._LinAlgError:
                # tiny-lambda + float32 Gram can be numerically indefinite;
                # fall back to the general (LU) solver, same solution
                self.alpha = torch.linalg.solve(A, Y)
        elif backend == "solve":
            self.alpha = torch.linalg.solve(A, Y)
        else:
            raise ValueError(backend)
        # cache for inference
        self.Z_train = Z
        self.K_train = K
        self.Y_train = Y
        # primal coefficients (exact for the linear kernel): beta = Z^T alpha
        self.beta = Z.T @ self.alpha                            # [d, C]
        return self

    # ------------------------------------------------------- inference
    def logits(self, Z):
        Z = torch.as_tensor(Z).float().to(self.device)
        return Z @ self.Z_train.T @ self.alpha                  # k(x,X) alpha

    def logits_from_k(self, k_test):
        """k_test: [N, n] precomputed cross Gram matrix."""
        k = torch.as_tensor(k_test).float().to(self.device)
        return k @ self.alpha

    def predict_proba(self, Z):
        logits = self.logits(Z)
        return F.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, Z):
        return self.logits(Z).argmax(-1).cpu().numpy()

    # --------------------------------------------- group contributions
    def group_logits(self, Z, block_offsets):
        """Exact per-group logit contributions f_j(x) = Z_j beta_j.

        Z: [N, d] (standardized, same space as training);
        block_offsets: [G+1] cumulative block boundaries.
        Returns list of [N, C] tensors (numpy).
        """
        Zt = torch.as_tensor(Z).float().to(self.device)
        out = []
        for j in range(len(block_offsets) - 1):
            lo, hi = int(block_offsets[j]), int(block_offsets[j + 1])
            out.append((Zt[:, lo:hi] @ self.beta[lo:hi]).cpu().numpy())
        return out

    # ------------------------------------------------------- state
    def state(self):
        return dict(lam=self.lam, n_classes=self.C,
                    alpha=self.alpha.cpu().numpy(),
                    beta=self.beta.cpu().numpy(),
                    Z_train=self.Z_train.cpu().numpy(),
                    y_train=self.Y_train.cpu().numpy())

    def save(self, path):
        np.savez_compressed(path, **self.state())

    @classmethod
    def load(cls, path, device=None):
        z = np.load(path)
        obj = cls(n_classes=int(z["n_classes"]), lam=float(z["lam"]),
                  device=device)
        obj.alpha = torch.from_numpy(z["alpha"]).float().to(obj.device)
        obj.beta = torch.from_numpy(z["beta"]).float().to(obj.device)
        obj.Z_train = torch.from_numpy(z["Z_train"]).float().to(obj.device)
        obj.Y_train = torch.from_numpy(z["y_train"]).float().to(obj.device)
        obj.K_train = obj.Z_train @ obj.Z_train.T
        return obj


def dual_primal_check(Z_train, y_train, lam, Z_test, atol=1e-3, device=None):
    """Phase 29: verify dual == primal within tolerance.

    Returns dict with max abs logit diff and a pass flag. Primal solve is
    only feasible for small d (used in the validation harness with reduced
    feature dimensionality or a sample subset).
    """
    device = device or torch.device("cpu")
    # float64 for a fair numerical comparison (production fits are float32;
    # the tolerance reflects that reality — see validation record)
    Z = torch.as_tensor(Z_train).double().to(device)
    Zt = torch.as_tensor(Z_test).double().to(device)
    C = int(np.max(y_train)) + 1
    Y = _onehot(y_train, C).to(torch.float64).to(device)
    n = Z.shape[0]

    # dual
    K = Z @ Z.T
    A = K + (lam + 1e-6) * torch.eye(n, device=device)
    alpha = torch.linalg.solve(A, Y)
    dual_logits = Zt @ Z.T @ alpha

    # primal
    A_p = Z.T @ Z + (lam + 1e-6) * torch.eye(Z.shape[1], device=device)
    beta = torch.linalg.solve(A_p, Z.T @ Y)
    primal_logits = Zt @ beta

    diff = (dual_logits - primal_logits).abs().max().item()
    return dict(max_abs_logit_diff=diff, atol=atol,
                passed=bool(diff < atol), n=int(n), d=int(Z.shape[1]),
                lam=float(lam))
