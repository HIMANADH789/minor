"""TURS-MGB transport geometry views (Phase 2).

Four FIXED feature-generation geometries, all computed independently for
every sample. These are NOT experts and are NOT routed; all are retained.

    G1 standard  : canonical W1 transport vs TRAIN reference, coarse grid
    G2 tail      : tail-weighted quantile transport (predefined weights)
    G3 fine      : W1 vs TRAIN reference on a dense (25-point) quantile grid
    G4 multilag  : multi-lag sorted-drift displacement vs TRAIN reference

Reuse contract: the per-window flavor mathematics come verbatim from
models/turs_rrmt/model.py::TransportFlavors (T1..T4), whose reference
templates are fit on TRAIN only. This module adds:
  - batched full-split extraction with memory-safe chunking
  - a persistent [N, J, T] cache per split
  - a spec hash covering geometry definitions (grids, tail weights, lags)

The geometry definitions are fixed project-wide (grid sizes, tail weights,
lag set are constants here, NOT dataset-tuned) — Phase 2's geometry rule.
"""

import hashlib

import numpy as np
import torch

from models.turs_rrmt.model import (
    TransportFlavors, QUANT_GRID_COARSE, QUANT_GRID_FINE, TAIL_WEIGHTS)

GEOMETRY_NAMES = ["G1_standard", "G2_tail", "G3_fine", "G4_multilag"]
LAG_SET = (1, 2, 4, 8)


class TransportGeometryBank:
    """Batched extractor for the four fixed transport geometries."""

    def __init__(self, lag_set=LAG_SET, ref_grid=10, fine_grid=25, seed=42):
        self.flavors = TransportFlavors(lag_set=lag_set, ref_grid=ref_grid)
        self.lag_set = tuple(lag_set)
        self.ref_grid = ref_grid
        self.fine_grid = fine_grid
        self.seed = seed
        self.fitted = False
        h = hashlib.sha256()
        h.update(QUANT_GRID_COARSE.numpy().tobytes())
        h.update(QUANT_GRID_FINE.numpy().tobytes())
        h.update(TAIL_WEIGHTS.numpy().tobytes())
        h.update(str(tuple(lag_set)).encode())
        self._spec_hash = h.hexdigest()[:16]

    @property
    def spec_hash(self):
        return self._spec_hash

    @torch.no_grad()
    def fit(self, X_train):
        """Fit reference templates on TRAIN only."""
        self.flavors.fit_reference(torch.from_numpy(X_train).float())
        self.fitted = True
        return self

    @torch.no_grad()
    def extract(self, X, window, batch=128, device=None):
        """X: [N, 1, T] numpy -> T [N, 4, T] float32 numpy (chunked).

        window: local quantile window (max(3, round(0.05*T)) by convention).
        """
        assert self.fitted, "fit() on TRAIN before extract()"
        device = device or torch.device("cpu")
        outs = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(device)
            outs.append(self.flavors(xb, window=window).cpu().numpy())
            del xb
        return np.concatenate(outs, 0).astype(np.float32)

    def view(self, T, j):
        """Slice one geometry view: T [N, 4, T'] -> [N, T'] (view j)."""
        assert 0 <= j < 4
        return np.ascontiguousarray(T[:, j, :])
