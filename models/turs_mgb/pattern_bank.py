"""TURS-MGB fixed temporal kernel bank.

Reuses the project's MiniROCKET-inspired deterministic kernel recipe from
models/turs_rrmt/model.py::build_pattern_bank (audit-verified: deterministic
seed, zero trainable parameters, multi-length, random 3-tap weights).

This module wraps it for batched extraction over arbitrary [B, 1, T] inputs
with explicit chunking (memory safety, Phase 31) and records the bank spec
(seed, lengths, kernel hash) for the audit trail (Phase 3).

The SAME bank instance must be applied to every transport geometry view
(Phase 4); this module makes sharing trivial by being device-agnostic and
stateless beyond the frozen weights.
"""

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from models.turs_rrmt.model import build_pattern_bank  # reuse, don't reinvent


class SharedKernelBank:
    """Fixed MiniROCKET-style kernel bank shared across geometry views.

    Parameters
    ----------
    M : number of kernels (e.g. 2048 / 4096 / 8192)
    lengths : kernel tap lengths (deterministic assignment across M)
    seed : RNG seed for the bank (fixed for reproducibility)
    device : torch device for convolutions

    Notes
    -----
    - zero trainable parameters
    - kernel_hash identifies the exact weight tensor (audit trail)
    - responses() returns per-length groups [B, n_L, T-L+1]; because kernels
      are right-aligned inside the L_max window, every group keeps FULL
      temporal resolution T (the receptive field is right-aligned) — this is
      the same behavior as the RRMT bank.
    """

    def __init__(self, M=4096, lengths=(7, 11, 15, 23, 31), seed=42,
                 device=None):
        self.M = int(M)
        self.lengths = tuple(lengths)
        self.seed = int(seed)
        W, spec = build_pattern_bank(self.M, self.lengths, in_channels=1,
                                     seed=self.seed)
        self.spec = spec
        self.distinct_lengths = sorted(set(s["length"] for s in spec))
        # one Conv1d per distinct length, weights frozen, moved once
        self.convs = []
        self._offsets = []
        for L in self.distinct_lengths:
            idx = [i for i, s in enumerate(spec) if s["length"] == L]
            w = W[idx]                                     # [n, 1, L]
            conv = torch.nn.Conv1d(1, len(idx), L, padding=0, bias=False)
            conv.weight.data = w
            conv.weight.requires_grad_(False)
            conv.eval()
            self.convs.append(conv)
            self._offsets.append((L, idx))
        self._spec_hash = hashlib.sha256(W.numpy().tobytes()).hexdigest()[:16]
        self.to(device or torch.device("cpu"))

    @property
    def kernel_hash(self):
        return self._spec_hash

    def to(self, device):
        self.convs = [c.to(device) for c in self.convs]
        return self

    @torch.no_grad()
    def responses(self, x, chunk=None):
        """x: [B, 1, T] torch tensor -> list over length-groups of
        [B, n_L, T] response tensors (right-aligned receptive fields)."""
        outs = []
        for conv in self.convs:
            r = conv(x)                                    # [B, n, T-L+1]
            pad = conv.kernel_size[0] - 1                  # right-align to T
            r = F.pad(r, (pad, 0))                         # -> [B, n, T]
            outs.append(r)
        return outs

    @torch.no_grad()
    def features(self, x, chunk=256, stats=("ppv", "max")):
        """x: [B, 1, T] numpy -> Z [B, M * len(stats)] float32 numpy.

        PPV  = mean_t I[r>0]        (MiniROCKET-primary)
        max  = max_t r              (bounded secondary)
        mean = mean_t r             (optional robust extra)

        Chunked along the batch dimension; intermediates freed per chunk.
        """
        n_stats = len(stats)
        Z = None
        for i in range(0, len(x), chunk):
            xb = torch.from_numpy(x[i:i + chunk]).float()
            dev = self.convs[0].weight.device
            xb = xb.to(dev)
            feats = []
            for conv in self.convs:
                r = conv(xb)                           # [B, n, T-L+1]
                group = []
                if "ppv" in stats:
                    group.append((r > 0).float().mean(dim=-1))
                if "max" in stats:
                    group.append(r.max(dim=-1).values)
                if "mean" in stats:
                    group.append(r.mean(dim=-1))
                feats.append(torch.cat(group, dim=1))
                del r
            zb = torch.cat(feats, dim=1).cpu().numpy().astype(np.float32)
            del feats
            if Z is None:
                Z = np.zeros((len(x), zb.shape[1]), np.float32)
            Z[i:i + chunk] = zb
        return Z
