"""R4 differentiable-Ridge regime gate (replaces the softmax classifier).

Central computational graph (spec section 31):

    theta (8,) -> sigmoid -> v_k
        -> H_gated(v) = sum_k v_k H_{m,k}          (frozen contributions)
        -> X(v) = [G || H_gated(v)]
        -> closed-form dual Ridge solve (TRAIN rows ONLY)
              K(v) = Xc_tr(v) Xc_tr(v)^T + alpha*I
              beta(v) = K(v)^-1 Xc_tr(v)^T Yc
        -> f_val(v) = Xc_val(v) beta(v) + intercept
        -> L = MSE(f_val, Y_val_onehot) + lambda_v sum_k theta_k^2
        -> grad -> theta (the ONLY optimized parameters)

Dual form is used throughout (n << p: p = 9996, n_train = 132 / 5226).
Everything is float64.  sklearn-Ridge semantics mirrored exactly: one-hot
{0,1} targets, fit_intercept=True (center X and Y with TRAIN statistics),
decision = X beta + intercept, prediction = argmax(decision).

Efficiency: with H_gated(v) = einsum(Hk, v), the Gram decomposes
    Xc_tr(v) Xc_tr(v)^T = sum_{k,l} v_k v_l (Hk_k Hk_l^T) + const
so the per-step solve is over cached K x K regime-block Grams -- no
n x F x K tensor is ever materialized.  Dual solves use Cholesky
(torch.linalg.solve uses LU; the Cholesky path is chosen for stability
on the SPD system and, crucially, gives gradients that reuse the same
factorization).  The old Adam-trained softmax classifier no longer
exists anywhere in this module (audits 4/5).
"""

import numpy as np
import torch

K_CODES = 8
THETA_INIT = -2.0                      # v_init = sigmoid(-2) ~ 0.1192 (<0.5)
LAMBDA_V = 1e-3                        # gate penalty lambda_v * sum theta^2
LR = 1e-2                              # outer Adam lr (spec section 15)
MAX_STEPS = 500
PATIENCE = 50


class DualRidgeGateModel:
    """Closed-form dual Ridge with 8 regime gates; float64 throughout."""

    def __init__(self, alpha, k_codes=K_CODES):
        self.alpha = float(alpha)
        self.k = k_codes
        # exactly the 8 optimized parameters (audit 6/13)
        self.theta = torch.full((k_codes,), float(THETA_INIT),
                                dtype=torch.float64, requires_grad=True)
        self.cache = {}

    # ---------------- cache preparation (one-off) ---------------------- #
    def prepare(self, G_tr, Hk_tr, y_tr, G_va, Hk_va, y_va):
        """Precompute centering stats, target encoding, and per-regime
        Gram/Gram-vector caches.

        The Ridge solve uses TRAIN rows ONLY; validation rows appear
        solely in the prediction/outer-loss path (audit 10)."""
        G_tr = np.ascontiguousarray(G_tr, dtype=np.float64)
        G_va = np.ascontiguousarray(G_va, dtype=np.float64)
        Hk_tr = np.ascontiguousarray(Hk_tr, dtype=np.float32)
        Hk_va = np.ascontiguousarray(Hk_va, dtype=np.float32)
        self.C = int(y_tr.max()) + 1
        Y_tr = np.eye(self.C)[np.asarray(y_tr, dtype=int)]
        Y_va = np.eye(self.C)[np.asarray(y_va, dtype=int)]
        # train-mean centering (sklearn fit_intercept=True convention)
        self.g_mean = G_tr.mean(axis=0)                      # (p1,)
        self.mu_k = Hk_tr.astype(np.float64).mean(axis=0)    # (F, K)
        self.Y_mean = Y_tr.mean(axis=0)                      # (C,)

        Gc_tr = torch.from_numpy(G_tr - self.g_mean)
        Gc_va = torch.from_numpy(G_va - self.g_mean)
        Hkc_tr = torch.from_numpy(
            Hk_tr.astype(np.float64) - self.mu_k[None])      # (n, F, K)
        Hkc_va = torch.from_numpy(
            Hk_va.astype(np.float64) - self.mu_k[None])
        Yc_tr = torch.from_numpy(Y_tr - self.Y_mean)         # (n, C)

        n = Gc_tr.shape[0]
        # --- constant caches ------------------------------------------ #
        # Kg = Gc Gc^T is constant; all v-dependent Gram blocks are
        # rebuilt per step from Hg = einsum(Hkc, v) -- O(n^2 F) each,
        # which is cheap even at ECG5000_BAL scale (n=5226, F=4998).
        Kg = Gc_tr @ Gc_tr.t()                               # (n, n)
        B_G = Gc_tr.t() @ Yc_tr                              # (p1, C)
        # per-regime target moments: Hkc_k^T Yc  (K, F, C) -- tiny
        M_H = torch.einsum("nfk,nc->kfc", Hkc_tr, Yc_tr)
        self.cache = {
            "Gc_tr": Gc_tr, "Gc_va": Gc_va,
            "Hkc_tr": Hkc_tr, "Hkc_va": Hkc_va,
            "Y_tr": torch.from_numpy(Y_tr.astype(np.float64)),
            "Y_va": torch.from_numpy(Y_va.astype(np.float64)),
            "Yc_tr": Yc_tr, "Kg": Kg, "B_G": B_G, "M_H": M_H,
            "n_train": n, "n_val": Gc_va.shape[0],
        }

    # ---------------- gated representation ----------------------------- #
    def v(self):
        return torch.sigmoid(self.theta)

    def gated_H(self, Hk):
        """H_gated(v) = einsum(Hk, v) for arbitrary rows (no centering).
        Used by the identity audits (v=1 -> H_R2, v=0 -> 0)."""
        return torch.einsum("nfk,k->nf", Hk, self.v())

    def train_gram(self):
        """K(v) = Xc_tr(v) Xc_tr(v)^T + alpha*I (train rows only).

        Built EXPLICITLY from the same centered matrix used for the
        beta reconstruction, so the solve and the prediction path share
        one identical matrix (no cached-vs-fresh Gram discrepancy).
        Differentiable in v via Hg."""
        c = self.cache
        Hg = torch.einsum("nfk,k->nf", c["Hkc_tr"], self.v())
        self._Xc_tr = torch.cat([c["Gc_tr"], Hg], dim=1)   # (n, p)
        K = self._Xc_tr @ self._Xc_tr.t()
        return K + self.alpha * torch.eye(c["n_train"],
                                          dtype=torch.float64)

    def rhs(self):
        """B(v) = B_G + sum_k v_k M_H[k]   (p1+F, C) -- primal RHS,
        retained for the independent cross-check in the runner."""
        c = self.cache
        B_H = torch.einsum("kfc,k->fc", c["M_H"], self.v())
        return torch.cat([c["B_G"], B_H], dim=0)

    def solve_beta(self):
        """Dual Ridge solve (n x n, n << p):

            alpha_coef = (Xc Xc^T + aI)^-1 Yc        (n, C)
            beta       = Xc_tr(v)^T alpha_coef        (p, C)

        mathematically identical to the primal (X^T X + aI)^-1 X^T Y
        (push-through identity).  beta is differentiable in v through
        both K(v) and Xc_tr(v).  Always builds a fresh graph (the
        returned tensors are part of the caller's autograd graph)."""
        c = self.cache
        K = self.train_gram()
        Xc_tr = self._Xc_tr
        L = torch.linalg.cholesky(K)
        alpha_coef = torch.cholesky_solve(c["Yc_tr"], L)      # (n, C)
        beta = Xc_tr.t() @ alpha_coef                          # (p, C)
        return beta, K, alpha_coef, L

    def decision_val(self, beta):
        """Validation decision function at the current v.

        Xc_va is already centered by TRAIN statistics, so the sklearn
        intercept semantics reduce EXACTLY to f = Xc_va @ beta + Y_mean
        (the X_mean@beta term of the raw-input formula is already inside
        Xc_va @ beta)."""
        c = self.cache
        Hg_va = torch.einsum("nfk,k->nf", c["Hkc_va"], self.v())
        Xc_va = torch.cat([c["Gc_va"], Hg_va], dim=1)
        return Xc_va @ beta + torch.from_numpy(self.Y_mean)

    def outer_loss(self):
        """L = MSE(decision_val, Y_val one-hot) + lambda_v sum theta^2.

        Validation labels enter ONLY here -- never the Ridge solve."""
        c = self.cache
        beta, K, alpha_coef, L = self.solve_beta()
        f = self.decision_val(beta)
        mse = ((f - c["Y_va"]) ** 2).mean()
        return mse + LAMBDA_V * (self.theta ** 2).sum(), beta, K, \
            alpha_coef, L

    # ---------------- final refit (train+val, R2 convention) ----------- #
    def refit_trainval(self, G_trva, Hk_trva, y_trva, v_np):
        """Canonical R2 final fit: closed-form Ridge on train+val at the
        frozen selected v (dual solve, alpha frozen; one-hot targets,
        centered intercept)."""
        G = np.ascontiguousarray(G_trva, dtype=np.float64)
        Hg = np.einsum("nfk,k->nf",
                       np.ascontiguousarray(Hk_trva, dtype=np.float32)
                       .astype(np.float64), v_np)
        C = int(np.max(y_trva)) + 1
        Y = np.eye(C)[np.asarray(y_trva, dtype=int)]
        X = np.hstack([G, Hg])
        X_mean = X.mean(axis=0)
        Xc = torch.from_numpy(X - X_mean)
        Y_mean = Y.mean(axis=0)
        Yc = torch.from_numpy(Y - Y_mean)
        n = X.shape[0]
        A = Xc @ Xc.t() + self.alpha * torch.eye(n, dtype=torch.float64)
        Lf = torch.linalg.cholesky(A)
        a_c = torch.cholesky_solve(Yc, Lf)
        beta = (Xc.t() @ a_c)
        b0 = torch.from_numpy(Y_mean) - torch.from_numpy(X_mean) @ beta
        resid = float((A @ a_c - Yc).abs().max())
        return beta.numpy(), b0.numpy(), resid

    def decision_matrix(self, G, Hk, beta, b0, v_np):
        """Decision function for arbitrary rows at frozen v."""
        Hg = np.einsum("nfk,k->nf",
                       np.ascontiguousarray(Hk, dtype=np.float32)
                       .astype(np.float64), v_np)
        X = np.hstack([np.asarray(G, dtype=np.float64), Hg])
        return X @ beta + b0


def run_outer_loop(model, y_va_np, max_steps=MAX_STEPS, patience=PATIENCE,
                   lr=LR, seed=42, cond_every=25):
    """Outer Adam on theta ONLY (audits 5/6).

    The Ridge is re-solved in closed form at every step; there are NO
    classifier epochs, NO classifier optimizer, NO CE loss.  Gradient
    flows: val MSE -> beta -> K(v)/B(v) -> v -> theta (audit 11).

    The optimizer's parameter group contains exactly [model.theta]
    (asserted here -- audit 6).  Selection is validation-only: best val
    Macro-F1, ties broken by lower val MSE.  The expensive condition
    number is computed only every `cond_every` steps (ECG5000_BAL scale:
    n=5226 SVD)."""
    from experiments.rcmkn_haptics_seed42.runner import macro_f1
    torch.manual_seed(seed)
    opt = torch.optim.Adam([model.theta], lr=lr)
    assert opt.param_groups[0]["params"] == [model.theta], \
        "AUDIT 6: optimizer must contain ONLY theta"
    c = model.cache
    traj = []
    best = {"val_mf1": -1.0, "val_mse": float("inf"), "step": -1,
            "theta": None, "v": None}
    max_cond = 0.0
    for step in range(max_steps):
        opt.zero_grad()
        loss, beta, K, a_c, L = model.outer_loss()
        with torch.no_grad():
            f = model.decision_val(beta)
            pred = f.argmax(dim=1).numpy()
            mf1 = float(macro_f1(y_va_np, pred))
            mse = float(((f - c["Y_va"]) ** 2).mean())
            # solver residual ||K alpha_coef - Yc_tr||_max (dual system)
            resid = float((K @ a_c - c["Yc_tr"]).abs().max())
            if step % cond_every == 0:
                max_cond = max(max_cond, float(torch.linalg.cond(K)))
            th = model.theta.detach().numpy().tolist()
            vv = model.v().detach().numpy().tolist()
        loss.backward()
        grad_ok = (model.theta.grad is not None
                   and bool(torch.isfinite(model.theta.grad).all())
                   and float(model.theta.grad.abs().max()) > 0.0)
        opt.step()
        rec = {"step": step, "outer_loss": float(loss), "val_mse": mse,
               "val_macro_f1": mf1, "solver_residual": resid,
               "grad_finite_nonzero": grad_ok, "theta": th, "v": vv}
        if step % cond_every == 0:
            rec["cond_number"] = max_cond
        traj.append(rec)
        if (mf1 > best["val_mf1"]
                or (mf1 == best["val_mf1"] and mse < best["val_mse"])):
            best = {"val_mf1": mf1, "val_mse": mse, "step": step,
                    "theta": th, "v": vv}
        if step - best["step"] >= patience:
            break
    final = {"theta": traj[-1]["theta"], "v": traj[-1]["v"],
             "val_mf1": traj[-1]["val_macro_f1"]}
    return best, final, traj
