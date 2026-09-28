"""TURS-MGB feature extractor (Phases 4-5, 30-31).

The pipeline for ONE geometry view j:

    view_j [N, 1, T]  (transport geometry reshaped as a 1-channel signal)
        -> SHARED fixed kernel bank K (same instance for j=1..4)
        -> per-kernel PPV / max (+ optional mean)
        -> Z_j [N, M * n_stats]

Geometry-as-channel is the core trick: the transport geometry produces a
per-timestep scalar field T_j(t) [N, T']; we treat it exactly like a
1-channel time series and convolve it with the SAME bank used on the raw
signal — making transport geometry the independent feature axis.

Caching (Phase 30): views, features, and standardized blocks are cached to
disk per (dataset, split); the kernel bank itself is deterministic and
reused across all geometries without regeneration.
"""

import os

import numpy as np

from models.turs_mgb.pattern_bank import SharedKernelBank
from models.turs_mgb.transport_views import TransportGeometryBank


class MGBFeatureExtractor:
    """Shared-bank multi-geometry feature extractor."""

    def __init__(self, M=4096, seed=42, stats=("ppv", "max"),
                 lag_set=(1, 2, 4, 8), device=None, geometry_names=None):
        self.M = int(M)
        self.seed = int(seed)
        self.stats = tuple(stats)
        self.geometry_names = geometry_names or ["G1_standard", "G2_tail",
                                                 "G3_fine", "G4_multilag"]
        self.device = device or torch.device("cpu")
        # ONE bank shared by every geometry (Phase 4)
        self.bank = SharedKernelBank(M=M, seed=seed, device=self.device)
        self.views = TransportGeometryBank(lag_set=lag_set, seed=seed)
        self._fitted = False

    # ------------------------------------------------------------ refs
    def fit_transport_references(self, X_train):
        """Fit transport reference templates on TRAIN only."""
        self.views.fit(X_train)
        self._fitted = True

    @property
    def ppv_window(self):
        # local quantile window follows the canonical convention
        return None  # resolved per split length in extract_views

    # ----------------------------------------------------------- views
    def extract_views(self, X, window=None, batch=128):
        """X: [N, 1, T] -> dict name -> T_j [N, T] float32 (4 views)."""
        assert self._fitted, "call fit_transport_references on TRAIN first"
        T_len = X.shape[-1]
        window = window or max(3, round(0.05 * T_len))
        T = self.views.extract(X, window=window, batch=batch,
                               device=self.device)          # [N, 4, T]
        out = {name: self.views.view(T, j)
               for j, name in enumerate(self.geometry_names)}
        return out

    # -------------------------------------------------------- features
    def extract_features(self, X, window=None, chunk=256, batch=128):
        """X: [N, 1, T] raw signal -> dict:
            raw      : [N, M*n_stats]  (baseline A0 features)
            G1..G4   : per-geometry feature blocks
            block_dims, offsets, feature_dim
        """
        feats = {"raw": self.bank.features(X, chunk=chunk, stats=self.stats)}
        views = self.extract_views(X, window=window, batch=batch)
        for name, V in views.items():
            # geometry field as a 1-channel signal through the SAME bank
            feats[name] = self.bank.features(V[:, None, :], chunk=chunk,
                                             stats=self.stats)
        n_stats = len(self.stats)
        self.block_dims = [self.M * n_stats] * 4
        self.raw_dim = self.M * n_stats
        self.offsets = np.concatenate(
            [[0], np.cumsum([self.raw_dim] + self.block_dims)])
        self.feature_dim = int(self.offsets[-1])
        feats["block_dims"] = [self.raw_dim] + self.block_dims
        feats["feature_dim"] = int(self.offsets[-1])
        feats["raw_dim"] = int(self.raw_dim)
        return feats

    def concat_with_raw(self, feats, keys):
        """Deterministically concatenate selected feature blocks."""
        return np.concatenate([feats[k] for k in keys], axis=1).astype(np.float32)

    def spec(self):
        """Audit/spec record (Phases 3-5)."""
        return dict(
            M=self.M, stats=list(self.stats), seed=self.seed,
            kernel_hash=self.bank.kernel_hash,
            kernel_lengths=list(self.bank.distinct_lengths),
            geometry_names=list(self.geometry_names),
            geometry_spec_hash=self.views.spec_hash,
            lag_set=list(self.views.lag_set),
            raw_dim=int(self.raw_dim),
            block_dims=list(self.block_dims),
            feature_dim=int(self.feature_dim),
            offsets=[int(o) for o in self.offsets])


# torch import kept local to avoid a hard dependency in numpy-only contexts
import torch  # noqa: E402
