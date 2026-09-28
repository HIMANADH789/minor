"""TURS-GLR global stream.

A LARGE fixed deterministic temporal kernel bank (MiniROCKET-inspired
construction reused from models/turs_rrmt/model.py so the kernel recipe is
carried over rather than reinvented) with GLOBAL (unrouted) pooling:
per-kernel PPV, max response, and response mean. Zero trainable parameters.

The bank is deliberately unrouted: this is the "big global bank" reference
stream. For the A8 ablation an optional wrapper exposes aeon MiniRocket as
an alternative global feature extractor.
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

from models.turs_rrmt.model import build_pattern_bank  # reuse, don't reinvent


class GlobalPatternBank(nn.Module):
    """Large fixed kernel bank -> global PPV / max / mean features.

    forward returns Z_global [B, 3*M_g]. Kernel construction is the exact
    RRMT build_pattern_bank (random-sign multi-length, deterministic seed).
    """

    def __init__(self, M_global=2048, lengths=(7, 11, 15, 23, 31), seed=42):
        super().__init__()
        self.M = M_global
        self.seed = seed
        # build_pattern_bank pads each kernel right-aligned into L_max
        W_full, spec = build_pattern_bank(M_global, lengths, in_channels=1, seed=seed)
        self.spec = spec
        self.lengths = sorted(set(s["length"] for s in spec))
        self.convs = nn.ModuleList()
        self._offsets = []
        for L in self.lengths:
            idx = [i for i, s in enumerate(spec) if s["length"] == L]
            conv = nn.Conv1d(1, len(idx), L, padding=0, bias=False)
            conv.weight.data = W_full[idx]          # [n, 1, L]
            conv.weight.requires_grad_(False)
            self.convs.append(conv)
            self._offsets.append((L, idx))
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self._spec_hash = hashlib.sha256(W_full.numpy().tobytes()).hexdigest()[:16]

    @property
    def spec_hash(self):
        return self._spec_hash

    def responses(self, x):
        """x: [B, 1, T] -> list of per-length response groups [B, n_L, T-L+1]."""
        outs = []
        for conv, (L, idx) in zip(self.convs, self._offsets):
            outs.append(conv(x))
        return outs

    def forward(self, x):
        """x: [B, 1, T] -> Z_global [B, 3*M_global] (PPV, max, mean per kernel)."""
        feats = []
        for conv, (L, idx) in zip(self.convs, self._offsets):
            r = conv(x)                              # [B, n, T-L+1]
            ppv = (r > 0).float().mean(dim=-1)
            mx = r.max(dim=-1).values
            mu = r.mean(dim=-1)
            feats.append(torch.cat([ppv, mx, mu], dim=1))
        return torch.cat(feats, dim=1)


class MiniRocketGlobal(nn.Module):
    """aeon MiniRocket wrapper (A8 ablation): the canonical 84-bias
    two-value-weight kernel construction, PPV-only features.

    fit() on TRAIN only (MiniRocket fits biases/dilations from data).
    """

    def __init__(self, n_kernels=2016, seed=42):
        super().__init__()
        self.n_kernels = n_kernels
        self.seed = seed
        self._mr = None

    def fit(self, X_train):
        """X_train: [N, 1, T] float32 numpy. Fits on TRAIN only."""
        from aeon.transformations.collection.convolution_based import MiniRocket
        X = np.ascontiguousarray(X_train[:, 0, :], dtype=np.float32)
        self._mr = MiniRocket(n_kernels=self.n_kernels, random_state=self.seed)
        self._mr.fit(X)
        return self

    def transform(self, X):
        X = np.ascontiguousarray(X[:, 0, :], dtype=np.float32)
        return self._mr.transform(X).astype(np.float32)

    @property
    def out_dim(self):
        return int(self._mr.transform(np.zeros((1, 1, 140), dtype=np.float32)).shape[1])
