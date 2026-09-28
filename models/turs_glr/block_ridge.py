"""TURS-GLR block-regularized ridge readout.

Solves the single closed-form readout with TWO regularization coefficients:

    min_{beta_g, beta_l} ||Y - Z_g beta_g - Z_l beta_l||^2
                         + lambda_g ||beta_g||^2 + lambda_l ||beta_l||^2

With Z = [Z_g || Z_l] and Lambda = diag(lambda_g I_g, lambda_l I_l):

    beta* = (Z^T Z + Lambda)^{-1} Z^T Y

Solved via torch.linalg.solve (Cholesky-backed on PSD systems; never an
explicit inverse). torch.linalg.solve is differentiable, so gradients flow
through the solution by implicit differentiation when the router is trained.

lambda_g / lambda_l are selected on VALIDATION ONLY over a compact log grid
(pairwise but small: 7x7 = 49 solves, each < 1 s at these dimensions).
Selection metric: validation Macro-F1; tie-break: validation NLL.
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F


def _mf1(y, pred, n_cls):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", zero_division=0,
                          labels=list(range(n_cls))))


class BlockRidge:
    """Closed-form block-regularized ridge classifier (one-vs-rest logits)."""

    def __init__(self, n_classes, lam_g=1.0, lam_l=1.0, eps=1e-6):
        self.C = n_classes
        self.lam_g = float(lam_g)
        self.lam_l = float(lam_l)
        self.eps = float(eps)
        self.W_star = None
        self.coef_norms = None
        # cached Gram pieces (set by precompute_gram / fit)
        self._cache = None

    def precompute_gram(self, Zg, Zl, y):
        """Compute and cache Z^T Z and Z^T Y once for a whole lambda sweep.

        The Gram matrix does not depend on the regularization, so one
        precompute serves all (lam_g, lam_l) pairs in the CV grid.
        """
        Zg_t = torch.as_tensor(np.asarray(Zg), dtype=torch.float64)
        Zl_t = torch.as_tensor(np.asarray(Zl), dtype=torch.float64)
        y_t = torch.as_tensor(np.asarray(y))
        Z = torch.cat([Zg_t, Zl_t], dim=1)
        Y = F.one_hot(y_t.long(), self.C).to(Z.dtype)
        self._cache = dict(Dg=Zg_t.shape[1], G=Z.T @ Z, ZtY=Z.T @ Y)
        return self._cache

    def fit(self, Zg, Zl, y, use_cache=False):
        """Zg: [N, Dg], Zl: [N, Dl] (numpy, already standardized); y: [N]."""
        if use_cache and self._cache is not None:
            Dg = self._cache["Dg"]
            A = self._cache["G"].clone()
            ZtY = self._cache["ZtY"]
        else:
            Zg_t = torch.from_numpy(np.asarray(Zg, np.float64))
            Zl_t = torch.from_numpy(np.asarray(Zl, np.float64))
            y_t = torch.as_tensor(np.asarray(y))
            N, Dg = Zg_t.shape
            Dl = Zl_t.shape[1]
            C = self.C
            Y = F.one_hot(y_t.long(), C).to(Zg_t.dtype)
            Z = torch.cat([Zg_t, Zl_t], dim=1)
            A = Z.T @ Z
            ZtY = Z.T @ Y
        Dl = A.shape[0] - Dg
        reg = torch.cat([torch.full((Dg,), self.lam_g),
                         torch.full((Dl,), self.lam_l)]).to(A.dtype)
        A = A + torch.diag(reg + self.eps)
        self.W_star = torch.linalg.solve(A, ZtY)              # [Dg+Dl, C]
        Wg = self.W_star[:Dg]
        Wl = self.W_star[Dg:]
        self.coef_norms = dict(
            global_norm=float(torch.linalg.norm(Wg)),
            local_norm=float(torch.linalg.norm(Wl)))
        return self

    def decision_function(self, Zg, Zl):
        Zg_t = torch.from_numpy(np.asarray(Zg, np.float64))
        Zl_t = torch.from_numpy(np.asarray(Zl, np.float64))
        return torch.cat([Zg_t, Zl_t], dim=1) @ self.W_star   # [N, C]

    def predict_proba(self, Zg, Zl):
        logits = self.decision_function(Zg, Zl)
        probs = F.softmax(logits.float(), dim=1).numpy()
        return probs

    def predict(self, Zg, Zl):
        return self.predict_proba(Zg, Zl).argmax(1)

    def block_logits(self, Zg, Zl):
        """Separate per-block linear contributions f_g(x), f_l(x)."""
        Zg_t = torch.from_numpy(np.asarray(Zg, np.float64))
        Zl_t = torch.from_numpy(np.asarray(Zl, np.float64))
        Dg = Zg_t.shape[1]
        return (Zg_t @ self.W_star[:Dg]).numpy(), (Zl_t @ self.W_star[Dg:]).numpy()


DEFAULT_LAM_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def _nll(probs, y):
    y = np.asarray(y)
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def fit_block_ridge_cv(Zg_tr, Zl_tr, y_tr, Zg_va, Zl_va, y_va, n_cls,
                       lam_grid=None, verbose=False, log=print):
    """Validation-only (lam_g, lam_l) selection.

    Primary metric: validation Macro-F1. Tie-break: validation NLL.
    The train Gram matrix is computed ONCE and reused for all pairs; only
    the diagonal regularization changes between solves.
    Returns (BlockRidge refit info, selection record).
    """
    if lam_grid is None:
        lam_grid = DEFAULT_LAM_GRID
    tmpl = BlockRidge(n_cls)
    tmpl.precompute_gram(Zg_tr, Zl_tr, y_tr)
    best = None
    records = []
    for lg, ll in itertools.product(lam_grid, lam_grid):
        m = BlockRidge(n_cls, lam_g=lg, lam_l=ll)
        m._cache = tmpl._cache
        m.fit(None, None, None, use_cache=True)
        probs_va = m.predict_proba(Zg_va, Zl_va)
        mf1 = _mf1(y_va, probs_va.argmax(1), n_cls)
        nll = _nll(probs_va, y_va)
        records.append(dict(lam_g=lg, lam_l=ll, val_mf1=mf1, val_nll=nll))
        if best is None or (mf1, -nll) > (best["val_mf1"], -best["val_nll"]):
            best = records[-1]
        if verbose:
            log(f"      lam_g={lg:g} lam_l={ll:g} val_mf1={mf1:.4f} val_nll={nll:.4f}")
    # refit the winner on train (train-only refit; val used only for selection)
    final = BlockRidge(n_cls, lam_g=best["lam_g"], lam_l=best["lam_l"])
    final._cache = tmpl._cache
    final.fit(None, None, None, use_cache=True)
    return final, dict(selected=best, grid=records)
