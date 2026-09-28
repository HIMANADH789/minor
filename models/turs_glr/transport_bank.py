"""TURS-GLR transport flavor bank.

Reuses the canonical RRMT TransportFlavors (quantile-geometry transport over
the shared TransportBuilder views) with TRAIN-fitted reference templates:
  T1 standard    : W1 vs reference, coarse grid (10 quantiles)
  T2 tail        : tail-weighted quantile transport
  T3 fine        : W1 vs reference, fine grid (25 quantiles)
  T4 multilag    : multi-lag sorted-drift displacement vs reference

This module only adds a batched multi-split extraction helper; the flavor
mathematics are unchanged from models/turs_rrmt/model.py.
"""

import torch

from models.turs_rrmt.model import TransportFlavors


class TransportBank(TransportFlavors):
    """Thin subclass adding batched extraction for full splits."""

    FLAVOR_NAMES = ["standard", "tail", "fine", "multilag"]

    @torch.no_grad()
    def extract_batched(self, X, window, batch=256):
        """X: [N, 1, T] numpy -> Tv [N, J, T'] numpy (J=4 flavors)."""
        import numpy as np
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float()
            outs.append(self(xb, window=window).cpu().numpy())
        return np.concatenate(outs, axis=0)
