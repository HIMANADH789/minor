"""TURS-GLR: Gated Global-Local Regime-Routed Ridge.

Top-level module wiring the two streams and the block-ridge readout.

  X -> GLOBAL STREAM  (fixed bank, M_g kernels, unrouted PPV/max/mean)
    -> Z_global [N, 3 M_g]
  X -> LOCAL STREAM   (fixed bank M_l -> local PPV A(t) + strength S~(t)
      -> P(t) -> SOFT ROUTER -> w(t) -> 4 transport flavors -> U_j = w_j T_j
      -> rich per-flavor statistics + pattern/routing stats)
    -> Z_local [N, D_local]

  prediction: BlockRidge(Z_g, Z_l) with (lambda_g, lambda_l) selected on
  validation. NO learned gate module, NO MLP head, NO fusion network.

The differentiable variant (A5) attaches the block-ridge solve inside the
training loop: router gradients flow through torch.linalg.solve by implicit
differentiation (the same machinery as models/turs_rrmt_v2/readout.py).
"""

import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.turs_glr.global_pattern_bank import GlobalPatternBank
from models.turs_glr.local_pattern_bank import LocalStream
from models.turs_glr.router import SoftRouter
from models.turs_glr.transport_bank import TransportBank
from models.turs_glr.routed_transport import routed_statistics
from models.turs_glr.feature_blocks import (build_local_block,
                                            local_block_dims)
from models.turs_glr.block_ridge import BlockRidge
from models.turs_rrmt.model import count_params  # (trainable, fixed) helper


class TURSGLR(nn.Module):
    """Unified global-local regime-routed model (feature extractor + router).

    The ridge readout lives outside this module (features are cached; the
    closed-form solve is batch-level, not per-sample). This module owns:
      - global bank (fixed), local bank (fixed), soft router (trainable)
      - transport flavors (fixed geometry, train-fitted references)
    """

    FLAVOR_NAMES = TransportBank.FLAVOR_NAMES

    def __init__(self, seq_len, M_global=2048, M_local=128, J=4,
                 tau=1.0, seed=42, lengths=(7, 11, 15, 23, 31)):
        super().__init__()
        self.seq_len = int(seq_len)
        self.M_global = int(M_global)
        self.M_local = int(M_local)
        self.J = int(J)
        self.tau = float(tau)
        self.seed = int(seed)

        self.global_bank = GlobalPatternBank(M_global=M_global, lengths=lengths,
                                             seed=seed)
        self.local_stream = LocalStream(M_local=M_local, lengths=lengths,
                                        seed=seed + 1, seq_len=seq_len)
        self.flavors = TransportBank(lag_set=(1, 2, 4, 8))
        self.router = SoftRouter(M_local=M_local, J=J, hidden=32, tau=tau)

        ppv_window = max(3, round(0.05 * seq_len))
        self.ppv_window = ppv_window
        dims = local_block_dims(M_local, J)
        self.local_dim = dims["total"]

    # ------------------------------------------------------------ fitting
    @torch.no_grad()
    def fit_train_only(self, X_train):
        """Everything data-dependent is fit on TRAIN only:
        transport reference templates + local strength medians."""
        self.flavors.fit_reference(torch.from_numpy(X_train).float())
        self.local_stream.set_train_medians(X_train)

    # -------------------------------------------------------- extraction
    def _device(self):
        """Device of the module's parameters (CPU if parameterless)."""
        return next(self.parameters()).device if any(True for _ in self.parameters()) \
            else torch.device("cpu")

    @torch.no_grad()
    def extract_global(self, X, batch=256):
        """X: [N, 1, T] numpy -> Z_global [N, 3*M_global]."""
        dev = self._device()
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(dev)
            outs.append(self.global_bank(xb).cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)

    def _local_forward(self, xb):
        """xb: [B, 1, T] tensor -> (U [B,J,T'], w [B,J,T'], A, S_norm, P)."""
        ls = self.local_stream(xb)
        R, A, S_norm, P = ls["R"], ls["A"], ls["S_norm"], ls["P"]
        Tv = self.flavors(xb, window=self.ppv_window)         # [B, J, T']
        Tp = min(P.shape[-1], Tv.shape[-1])
        P, Tv = P[..., :Tp], Tv[..., :Tp]
        A = A[..., :Tp]
        S_norm = S_norm[..., :Tp]
        w = self.router(P.permute(0, 2, 1)).permute(0, 2, 1)  # [B, J, T']
        U = w * Tv                                            # routed field
        return U, w, Tv, A, S_norm, P

    @torch.no_grad()
    def extract_local(self, X, batch=128, return_diag=False):
        """X: [N, 1, T] numpy -> Z_local [N, D_local] (optionally diag)."""
        dev = self._device()
        Zs, diags = [], []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(dev)
            U, w, Tv, A, S_norm, P = self._local_forward(xb)
            Zs.append(torch.from_numpy(build_local_block(U, w, A, S_norm)))
            if return_diag:
                diags.append(dict(U=U, w=w, Tv=Tv, A=A, S_norm=S_norm, P=P,
                                  x=xb))
        Z = torch.cat(Zs, 0).numpy().astype(np.float32)
        if return_diag:
            merged = {}
            for k in diags[0]:
                merged[k] = torch.cat([d[k] for d in diags], 0)
            return Z, merged
        return Z

    def forward_diag(self, xb):
        """Full temporals for diagnostics (xb: [B, 1, T] tensor)."""
        U, w, Tv, A, S_norm, P = self._local_forward(xb)
        return dict(U=U, w=w, Tv=Tv, A=A, S_norm=S_norm, P=P, x=xb)

    # ------------------------------------------------- differentiable path
    def differentiable_local_block(self, xb):
        """Differentiable version of the routed statistics block.

        Uses only the differentiable-safe subset of the local features
        (means / max / PPV / energy of U_j and w-statistics; quantiles are
        replaced by max/mean proxies to keep the graph). Returns
        Z_local_diff [B, D_diff] and (U, w) for diagnostics.
        """
        U, w, Tv, A, S_norm, P = self._local_forward(xb)
        B, J, T = U.shape
        feats = []
        for j in range(J):
            u = U[:, j, :]
            wj = w[:, j, :]
            top_seq = w.argmax(1)
            persist = ((top_seq[:, 1:] == top_seq[:, :-1]).float().mean(1)
                       if T > 1 else torch.ones(B, device=xb.device))
            feats.append(torch.stack([
                u.mean(1), u.max(1).values, (u > 0).float().mean(1),
                (u ** 2).mean(1), wj.mean(1), wj.max(1).values, persist],
                dim=1))
        Z_routed = torch.cat(feats, dim=1)                    # [B, J*7]
        Z_pat = torch.cat([A.mean(-1), A.max(-1).values,
                           S_norm.mean(-1)], dim=1)           # [B, 3M]
        Z = torch.cat([Z_routed, Z_pat], dim=1)
        return Z, dict(U=U, w=w, Tv=Tv, A=A)

    def spec_hash(self):
        h = hashlib.sha256()
        h.update(self.global_bank.spec_hash.encode())
        h.update(self.local_stream.bank.spec_hash.encode())
        h.update(f"{self.M_global}|{self.M_local}|{self.J}|{self.tau}|{self.seed}|"
                 f"{self.seq_len}|{self.ppv_window}".encode())
        return h.hexdigest()[:16]


def count_params_glr(model):
    """(trainable, fixed) parameter counts; fixed kernels are NOT trainable."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, fixed
