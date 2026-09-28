"""TURS-GLR feature blocks.

Assembles the two STRICTLY SEPARATE design matrices:
    Z_global: [N, 3*M_global]  global PPV/max/mean per kernel (unrouted)
    Z_local : [N, D_local]     routed transport stats || local pattern stats

Z_local composition (documented, deterministic):
    A. routed transport statistics     J * 15 = 60 dims
    B. local PPV statistics            per kernel: mean, std, max -> 3*M_local
    C. strength statistics             per kernel: mean, std, max -> 3*M_local
    D. routing statistics (flavor-agnostic): [mean entropy, persist, switch,
       top1-flavor frequencies (J)] -> 3 + J
    E. top-k route frequencies         J dims (top-1 already in D; top-2 kept)

No raw temporal tensors are included. Block identities are preserved: the
scaler is fit on TRAIN only, per block, and the block column counts are
recorded for the block-ridge solve.
"""

import numpy as np
import torch

from models.turs_glr.routed_transport import routed_statistics


def local_pattern_stats(A, S_norm, keep_torch=False):
    """A, S_norm: [B, M, T'] -> [B, 3M] each (mean/std/max over time).

    keep_torch=True keeps the results as torch tensors (grad-safe) for the
    differentiable router-training path; default returns float32 numpy.
    """
    B, M, T = A.shape
    def sts(x):
        mu = x.mean(-1)
        sd = x.std(-1) if T > 1 else torch.zeros_like(mu)
        mx = x.max(-1).values
        return torch.cat([mu, sd, mx], dim=1)
    a, s = sts(A), sts(S_norm)
    if keep_torch:
        return a, s
    return a.cpu().numpy().astype(np.float32), \
           s.cpu().numpy().astype(np.float32)


def routing_stats(w, keep_torch=False):
    """w: [B, J, T'] -> [B, 3 + J + J] aggregate routing features.

    mean entropy, persistence, switch rate, top-1 freq per flavor,
    top-2 freq per flavor. keep_torch=True keeps the torch tensor (grad-safe).
    """
    B, J, T = w.shape
    ent = -(w * torch.log(w + 1e-12)).sum(1).mean(1)          # [B]
    top_seq = w.argmax(1)                                     # [B, T]
    if T > 1:
        persist = (top_seq[:, 1:] == top_seq[:, :-1]).float().mean(1)
    else:
        persist = torch.ones(B, device=w.device)
    switch = 1.0 - persist
    # top-1 frequency per flavor (per sample)
    top1 = torch.zeros(B, J, device=w.device)
    top1.scatter_(1, top_seq, 1.0)
    top1 = top1 / T
    # top-2 frequency per flavor
    top2_idx = w.topk(min(2, J), dim=1).indices               # [B, 2, T]
    top2 = torch.zeros(B, J, device=w.device)
    top2.scatter_(1, top2_idx.reshape(B, -1), 1.0)
    top2 = top2 / (2 * T)
    Z = torch.cat([ent.unsqueeze(1), persist.unsqueeze(1),
                   switch.unsqueeze(1), top1, top2], dim=1)
    if keep_torch:
        return Z
    return Z.cpu().numpy().astype(np.float32)


def routed_statistics_torch(U, w):
    """Grad-safe twin of routed_statistics (returns torch [B, J*15]).

    Quantiles are replaced by grad-friendly max/mean proxies:
        q90 -> (max + mean) / 2     q10 -> (min + mean) / 2
    The numpy (inference) path keeps the exact quantiles; the router-training
    path uses this twin so the closed-form ridge solve stays differentiable.
    """
    B, J, T = U.shape
    top_seq = w.argmax(1)
    persist = ((top_seq[:, 1:] == top_seq[:, :-1]).float().mean(1)
               if T > 1 else torch.ones(B, device=U.device))
    switch = 1.0 - persist
    feats = []
    for j in range(J):
        u = U[:, j, :]
        wj = w[:, j, :]
        mu = u.mean(1)
        sd = u.std(1) if T > 1 else torch.zeros_like(mu)
        mx = u.max(1).values
        mn = u.min(1).values
        q90 = (mx + mu) / 2
        q10 = (mn + mu) / 2
        rng = q90 - q10
        ppv = (u > 0).float().mean(1)
        energy = (u ** 2).mean(1)
        wm = wj.mean(1)
        ws = wj.std(1) if T > 1 else torch.zeros_like(wm)
        wmx = wj.max(1).values
        wmin = wj.min(1).values
        ent = -(wj * torch.log(wj + 1e-12)).sum(1) / float(J)
        feats.append(torch.stack([mu, sd, mx, q90, q10, rng, ppv, energy,
                                  wm, ws, wmx, wmin, ent, persist, switch],
                                 dim=1))
    return torch.cat(feats, dim=1)


def build_local_block(U, w, A, S_norm, keep_torch=False):
    """Assemble Z_local from routed field, weights, and pattern activity.

    All inputs are torch tensors (any device). Default returns float32 numpy
    (inference/extraction path); keep_torch=True returns a torch tensor and
    keeps the autograd graph (differentiable router-training path, where
    quantiles are replaced by grad-friendly proxies).
    Order: [routed (J*15) || PPV stats (3M) || strength stats (3M) ||
            routing stats (3+J+J)]
    """
    if keep_torch:
        Z_routed = routed_statistics_torch(U, w)
        Z_ppv, Z_str = local_pattern_stats(A, S_norm, keep_torch=True)
        Z_rout = routing_stats(w, keep_torch=True)
        return torch.cat([Z_routed, Z_ppv, Z_str, Z_rout], dim=1)
    Z_routed = routed_statistics(U, w)
    Z_ppv, Z_str = local_pattern_stats(A, S_norm)
    Z_rout = routing_stats(w)
    return np.concatenate([Z_routed, Z_ppv, Z_str, Z_rout], axis=1)


def local_block_dims(M_local, J):
    """Documented Z_local dimensionality."""
    return dict(routed=J * 15, ppv=3 * M_local, strength=3 * M_local,
                routing=3 + 2 * J, total=J * 15 + 6 * M_local + 3 + 2 * J)


class BlockStandardizer:
    """Per-block standardization fit on TRAIN only. Keeps block identities."""

    def __init__(self, d_global, d_local):
        self.d_global = int(d_global)
        self.d_local = int(d_local)

    def fit(self, Zg, Zl):
        self.g_mu = Zg.mean(0)
        self.g_sd = np.clip(Zg.std(0), 1e-6, None)
        self.l_mu = Zl.mean(0)
        self.l_sd = np.clip(Zl.std(0), 1e-6, None)
        return self

    def transform(self, Zg, Zl):
        # tolerate 0-column blocks (A0/A1 ablations and global-only closures
        # pass empty matrices; subtracting a fitted mu would not broadcast)
        Zgs = (((Zg - self.g_mu) / self.g_sd).astype(np.float32)
               if Zg.shape[1] == self.d_global else
               np.zeros((len(Zg), 0), np.float32))
        Zls = (((Zl - self.l_mu) / self.l_sd).astype(np.float32)
               if Zl.shape[1] == self.d_local else
               np.zeros((len(Zl), 0), np.float32))
        return Zgs, Zls

    def state(self):
        return dict(g_mu=self.g_mu, g_sd=self.g_sd, l_mu=self.l_mu,
                    l_sd=self.l_sd, d_global=self.d_global,
                    d_local=self.d_local)

    @classmethod
    def from_state(cls, st):
        obj = cls(st["d_global"], st["d_local"])
        obj.g_mu, obj.g_sd = st["g_mu"], st["g_sd"]
        obj.l_mu, obj.l_sd = st["l_mu"], st["l_sd"]
        return obj
