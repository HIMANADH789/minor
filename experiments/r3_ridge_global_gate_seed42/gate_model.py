"""R3 gate model: ONE global scalar gate + closed-form dual Ridge.

Defining mechanism (spec sections 6-14):

    theta (scalar) -> w = sigmoid(theta)
    H_gate(w) = w * H                       (one scalar for ALL 4998)
    X(w) = [G || H_gate(w)]
    dual Ridge (TRAIN rows only):
        K(w) = Xc_tr(w) Xc_tr(w)^T + alpha*I
             = Kg + w^2 Kh + alpha*I           [exact block decomposition]
        a(w) = K(w)^-1 Yc_tr
        beta(w) = Xc_tr(w)^T a(w)
    f_val(w) = Xc_val(w) beta(w) + Y_mean
    L = MSE(f_val, Y_val one-hot) + lambda_w theta^2
    grad -> theta (the ONLY optimized parameter)

float64 throughout; sklearn-Ridge semantics mirrored exactly (one-hot
{0,1} targets, fit_intercept=True with TRAIN centering).  NO softmax
head, NO classifier optimizer, NO classifier epochs (audits: the R3
Ridge-update spec section 8).
"""

import numpy as np
import torch

THETA_INIT = -2.0                    # w_init = sigmoid(-2) ~ 0.1192 (<0.5)
LAMBDA_W = 1e-3                      # gate penalty lambda_w * theta^2
LR = 1e-2                            # outer Adam lr (spec section 14)
MAX_STEPS = 500
PATIENCE = 50


class GlobalGateRidge:
    """One global sigmoid gate + closed-form dual Ridge; float64."""

    def __init__(self, alpha):
        self.alpha = float(alpha)
        # exactly ONE learned scalar (audit 13)
        self.theta = torch.tensor(float(THETA_INIT), dtype=torch.float64,
                                  requires_grad=True)
        self.cache = {}

    # ---------------- cache preparation (one-off) ---------------------- #
    def prepare(self, G_tr, H_tr, y_tr, G_va, H_va, y_va):
        """Centering stats, one-hot targets, constant Gram blocks.

        Ridge is fit on TRAIN rows ONLY; validation appears solely in
        the prediction/outer-loss path (audit 19)."""
        G_tr = np.ascontiguousarray(G_tr, dtype=np.float64)
        H_tr = np.ascontiguousarray(H_tr, dtype=np.float64)
        G_va = np.ascontiguousarray(G_va, dtype=np.float64)
        H_va = np.ascontiguousarray(H_va, dtype=np.float64)
        self.C = int(y_tr.max()) + 1
        Y_tr = np.eye(self.C)[np.asarray(y_tr, dtype=int)]
        Y_va = np.eye(self.C)[np.asarray(y_va, dtype=int)]
        self.g_mean = G_tr.mean(axis=0)                      # (p1,)
        self.h_mean = H_tr.mean(axis=0)                      # (p2,)
        self.Y_mean = Y_tr.mean(axis=0)                      # (C,)

        self.Gc_tr = torch.from_numpy(G_tr - self.g_mean)
        self.Gc_va = torch.from_numpy(G_va - self.g_mean)
        self.Hc_tr = torch.from_numpy(H_tr - self.h_mean)
        self.Hc_va = torch.from_numpy(H_va - self.h_mean)
        self.Yc_tr = torch.from_numpy(Y_tr - self.Y_mean)
        self.Y_va = torch.from_numpy(Y_va.astype(np.float64))
        self.n_train = self.Gc_tr.shape[0]
        # constant Gram blocks (w-dependent part is w^2 * Kh, exact
        # because centering commutes with scalar gating)
        self.Kg = self.Gc_tr @ self.Gc_tr.t()                # (n, n)
        self.Kh = self.Hc_tr @ self.Hc_tr.t()                # (n, n)

    # ---------------- gated representation ----------------------------- #
    def w(self):
        return torch.sigmoid(self.theta)

    def gram(self):
        """K(w) = Kg + w^2 Kh + alpha I  (train rows only).

        Exact dual Gram of Xc_tr(w) = [Gc_tr, w Hc_tr] without
        materializing the (n, 9996) matrix; differentiable in w."""
        w = self.w()
        return (self.Kg + (w * w) * self.Kh
                + self.alpha * torch.eye(self.n_train,
                                         dtype=torch.float64))

    def xc_val(self):
        """Xc_val(w) = [Gc_va, w Hc_va]."""
        return torch.cat([self.Gc_va, self.w() * self.Hc_va], dim=1)

    def solve_beta(self):
        """Dual Ridge solve (n x n, n << p):

            a    = (Xc Xc^T + aI)^-1 Yc     (n, C)  via Cholesky
            beta = Xc_tr(w)^T a             (p, C)

        identical to the primal (X^T X + aI)^-1 X^T Y (push-through).
        beta differentiable in w through K(w) and Xc_tr(w)."""
        w = self.w()
        K = self.gram()
        Hg_tr = w * self.Hc_tr
        Xc_tr = torch.cat([self.Gc_tr, Hg_tr], dim=1)
        L = torch.linalg.cholesky(K)
        a_coef = torch.cholesky_solve(self.Yc_tr, L)         # (n, C)
        beta = Xc_tr.t() @ a_coef                            # (p, C)
        return beta, K, a_coef, L, Xc_tr

    def decision_val(self, beta):
        """f_val = Xc_val(w) beta + Y_mean (Xc_va already centered by
        TRAIN stats => sklearn intercept semantics exactly)."""
        return self.xc_val() @ beta + torch.from_numpy(self.Y_mean)

    def outer_loss(self):
        """L = MSE(f_val, Y_val one-hot) + lambda_w theta^2."""
        beta, K, a_c, L, Xc_tr = self.solve_beta()
        f = self.decision_val(beta)
        mse = ((f - self.Y_va) ** 2).mean()
        return mse + LAMBDA_W * self.theta ** 2, beta, K, a_c, L

    # ---------------- final refit (train+val, R2 convention) ----------- #
    def refit_trainval(self, G_trva, H_trva, y_trva, w_np):
        """Canonical R2 final fit: closed-form Ridge on train+val at the
        frozen selected w (dual solve, alpha frozen, one-hot targets,
        centered intercept)."""
        G = np.ascontiguousarray(G_trva, dtype=np.float64)
        H = np.ascontiguousarray(H_trva, dtype=np.float64)
        C = int(np.max(y_trva)) + 1
        Y = np.eye(C)[np.asarray(y_trva, dtype=int)]
        X = np.hstack([G, w_np * H])
        X_mean = X.mean(axis=0)
        Xc = torch.from_numpy(X - X_mean)
        Y_mean = Y.mean(axis=0)
        Yc = torch.from_numpy(Y - Y_mean)
        n = X.shape[0]
        A = Xc @ Xc.t() + self.alpha * torch.eye(n, dtype=torch.float64)
        sym = float((A - A.t()).abs().max())
        Lf = torch.linalg.cholesky(A)
        a_c = torch.cholesky_solve(Yc, Lf)
        beta = Xc.t() @ a_c
        b0 = torch.from_numpy(Y_mean) - torch.from_numpy(X_mean) @ beta
        resid = float((A @ a_c - Yc).abs().max())
        return beta.numpy(), b0.numpy(), resid, sym

    def decision_matrix(self, G, H, beta, b0, w_np):
        """Decision function for arbitrary rows at frozen w."""
        X = np.hstack([np.asarray(G, dtype=np.float64),
                       w_np * np.asarray(H, dtype=np.float64)])
        return X @ beta + b0


def run_outer_loop(model, y_va_np, max_steps=MAX_STEPS, patience=PATIENCE,
                   lr=LR, seed=42, cond_every=25):
    """Outer Adam on theta ONLY (audit 16).

    The Ridge is re-solved in closed form at every step; NO classifier
    epochs, NO classifier optimizer, NO CE loss.  Gradient flows
    val MSE -> beta -> K(w)/Xc_tr -> w -> theta (audit 20).  Selection:
    best val Macro-F1 with ties broken by lower val MSE; the expensive
    condition number is computed only every `cond_every` steps."""
    from experiments.rcmkn_haptics_seed42.runner import macro_f1
    torch.manual_seed(seed)
    opt = torch.optim.Adam([model.theta], lr=lr)
    assert opt.param_groups[0]["params"] == [model.theta], \
        "AUDIT 16: optimizer must contain ONLY theta"
    traj = []
    best = {"val_mf1": -1.0, "val_mse": float("inf"), "step": -1,
            "theta": None, "w": None}
    max_cond = 0.0
    for step in range(max_steps):
        opt.zero_grad()
        loss, beta, K, a_c, _ = model.outer_loss()
        with torch.no_grad():
            f = model.decision_val(beta)
            pred = f.argmax(dim=1).numpy()
            mf1 = float(macro_f1(y_va_np, pred))
            mse = float(((f - model.Y_va) ** 2).mean())
            # dual residual ||K a - Yc||_max and symmetry
            resid = float((K @ a_c - model.Yc_tr).abs().max())
            sym = float((K - K.t()).abs().max())
            if step % cond_every == 0:
                max_cond = max(max_cond,
                               float(torch.linalg.cond(K)))
            th = float(model.theta.detach())
            wv = float(model.w().detach())
        loss.backward()
        grad_ok = (model.theta.grad is not None
                   and bool(torch.isfinite(model.theta.grad).all())
                   and float(model.theta.grad.abs().max()) > 0.0)
        opt.step()
        rec = {"step": step, "outer_loss": float(loss), "val_mse": mse,
               "val_macro_f1": mf1, "solver_residual": resid,
               "gram_symmetry": sym, "grad_finite_nonzero": grad_ok,
               "theta": th, "w": wv}
        if step % cond_every == 0:
            rec["cond_number"] = max_cond
        traj.append(rec)
        if (mf1 > best["val_mf1"]
                or (mf1 == best["val_mf1"] and mse < best["val_mse"])):
            best = {"val_mf1": mf1, "val_mse": mse, "step": step,
                    "theta": th, "w": wv}
        if step - best["step"] >= patience:
            break
    final = {"theta": traj[-1]["theta"], "w": traj[-1]["w"],
             "val_mf1": traj[-1]["val_macro_f1"]}
    return best, final, traj
