"""TURS-MGB group-wise standardization (Phase 6).

One scaler per geometry group, fit on TRAIN only, applied unchanged to
val/test. The geometry groups remain logically identifiable: the scaler
records block boundaries so Z can always be re-split into
[Z_1 || Z_2 || Z_3 || Z_4].

Mandatory per the protocol: prevents any single geometry from dominating
through raw feature scale alone.
"""

import json

import numpy as np


class GroupStandardizer:
    """Per-geometry-block standardization with TRAIN-fitted stats."""

    def __init__(self, block_dims):
        """block_dims: list of per-geometry feature dimensionalities."""
        self.block_dims = list(block_dims)
        self.offsets = np.concatenate([[0], np.cumsum(self.block_dims)])
        self.n_blocks = len(self.block_dims)
        self.mu = [None] * self.n_blocks
        self.sd = [None] * self.n_blocks
        self.fitted = False

    def fit(self, Z_blocks):
        """Z_blocks: list of [N, d_j] TRAIN blocks."""
        assert len(Z_blocks) == self.n_blocks
        for j, Z in enumerate(Z_blocks):
            self.mu[j] = Z.mean(0).astype(np.float64)
            self.sd[j] = np.clip(Z.std(0), 1e-6, None).astype(np.float64)
        self.fitted = True
        return self

    def transform_block(self, j, Z):
        assert self.fitted
        return ((Z - self.mu[j]) / self.sd[j]).astype(np.float32)

    def transform(self, Z_blocks):
        """Standardize every block; returns list of transformed blocks."""
        return [self.transform_block(j, Z) for j, Z in enumerate(Z_blocks)]

    def transform_concat(self, Z):
        """Standardize an already-concatenated matrix [N, sum(d_j)]."""
        assert self.fitted and Z.shape[1] == int(self.offsets[-1])
        out = np.empty_like(Z, dtype=np.float32)
        for j in range(self.n_blocks):
            lo, hi = int(self.offsets[j]), int(self.offsets[j + 1])
            out[:, lo:hi] = ((Z[:, lo:hi] - self.mu[j]) / self.sd[j]).astype(np.float32)
        return out

    def state(self):
        return dict(
            block_dims=self.block_dims,
            offsets=[int(o) for o in self.offsets],
            mu=[m.tolist() for m in self.mu],
            sd=[s.tolist() for s in self.sd])

    @classmethod
    def from_state(cls, st):
        obj = cls(st["block_dims"])
        obj.offsets = np.array(st["offsets"])
        obj.mu = [np.array(m) for m in st["mu"]]
        obj.sd = [np.array(s) for s in st["sd"]]
        obj.fitted = True
        return obj

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.state(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_state(json.load(f))
