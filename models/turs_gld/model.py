"""TURS-GLD core feature extractor.

Three streams, all from validated existing components:
  A. GLOBAL  — GlobalPatternBank (GLR A8), M_global kernels, PPV/max/mean
  B. LOCAL   — FixedPatternBank + LocalActivity (RRMT), M_local kernels,
               local PPV A(t) + normalized strength S~(t)
  C. LOCAL G4 — multi-lag drift, computed per local window via
                TransportBuilder, with lag-displacement statistics

All three streams are concatenated (after per-block TRAIN-fitted
standardization) into one feature matrix and fed to a single dual Ridge.

Ablation: the constructor accepts `blocks` — a tuple of stream names to
include.  This lets the runner script construct A0-A7 without duplicating
the extraction logic.
"""

import hashlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── reuse validated components ──────────────────────────────────────
from models.turs_rrmt.model import (
    build_pattern_bank,       # kernel recipe
    FixedPatternBank,         # local kernel bank
    LocalActivity,            # local PPV + strength
    TransportFlavors,         # transport reference fitting
)
from models.tursnet import TransportBuilder   # raw / sorted / drift views
from models.turs_glr.global_pattern_bank import GlobalPatternBank, MiniRocketGlobal

# ── defaults (one architecture for ALL datasets) ────────────────────
DEFAULT = dict(
    M_global=2048,
    M_local=128,
    seed=42,
    lengths=(7, 11, 15, 23, 31),
    lag_set=(1, 2, 4, 8),
    local_window_frac=0.05,   # window = max(3, round(frac * T))
)

# ── helpers ─────────────────────────────────────────────────────────
def _stats_np(x, name=""):
    """[B, D, T] or [B, T] -> dict of summary statistics (numpy)."""
    if x.ndim == 3:
        mu = x.mean(-1)
        sd = x.std(-1) if x.shape[-1] > 1 else np.zeros_like(mu)
        mx = x.max(-1)
        mn = x.min(-1)
        return np.stack([mu, sd, mx, mn], -1).astype(np.float32).reshape(x.shape[0], -1)
    elif x.ndim == 2:
        mu = x.mean(-1)
        sd = x.std(-1) if x.shape[-1] > 1 else np.zeros_like(mu)
        mx = x.max(-1)
        mn = x.min(-1)
        return np.stack([mu, sd, mx, mn], -1).astype(np.float32)
    raise ValueError(f"unexpected ndim={x.ndim}")


def _stats_torch(x):
    """[B, D, T] or [B, T] -> [B, D*4] or [B, 4] torch."""
    if x.ndim == 3:
        mu = x.mean(-1)
        sd = x.std(-1) if x.shape[-1] > 1 else torch.zeros_like(mu)
        mx = x.max(-1).values
        mn = x.min(-1).values
        return torch.stack([mu, sd, mx, mn], -1).reshape(x.shape[0], -1)
    elif x.ndim == 2:
        mu = x.mean(-1)
        sd = x.std(-1) if x.shape[-1] > 1 else torch.zeros_like(mu)
        mx = x.max(-1).values
        mn = x.min(-1).values
        return torch.stack([mu, sd, mx, mn], -1)
    raise ValueError


# ====================================================================
# GLD Feature Extractor
# ====================================================================
class GLDFeatureExtractor:
    """Fixed-feature multi-stream extractor (zero trainable parameters).

    Parameters
    ----------
    blocks : tuple of str
        Which streams to include.  Any subset of:
        ``('global', 'local', 'local_g4')``.
        e.g. ``('global',)`` for A0, ``('global', 'local')`` for A3,
        ``('global', 'local', 'local_g4')`` for A7.
    """

    ALL_BLOCKS = ('global', 'local', 'local_g4')

    def __init__(self, blocks=ALL_BLOCKS, **kwargs):
        self.blocks = tuple(blocks)
        self.cfg = {**DEFAULT, **kwargs}
        self.M_g = self.cfg['M_global']
        self.M_l = self.cfg['M_local']
        self.seed = self.cfg['seed']
        self.lengths = self.cfg['lengths']
        self.lag_set = self.cfg['lag_set']
        self.window_frac = self.cfg['local_window_frac']

        # ── global bank (aeon MiniRocket, A8-validated) ───────────
        self.global_bank = None
        if 'global' in self.blocks:
            self.global_bank = MiniRocketGlobal(
                n_kernels=2016, seed=self.seed)

        # ── local bank + activity ──────────────────────────────────
        self.local_bank = None
        self.activity = None
        self.s_med = None
        if 'local' in self.blocks:
            self.local_bank = FixedPatternBank(
                M=self.M_l, lengths=self.lengths, seed=self.seed + 1)
            ppv_w = None  # set after seeing seq_len
            self.activity = None  # deferred
            self._ppv_w = None
            self.s_med = torch.ones(self.M_l)

        # ── transport / G4 ─────────────────────────────────────────
        self.builder = TransportBuilder()
        self.transport_flavors = None  # deferred fit
        self._ref_lag = None

        # ── caching ────────────────────────────────────────────────
        self._fitted = False

    # ================================================================ fit
    @torch.no_grad()
    def fit_train(self, X_train, seq_len=None):
        """Fit TRAIN-only data-dependent quantities:
        - aeon MiniRocket biases/dilations (global stream)
        - transport reference templates
        - local strength medians
        """
        dev = torch.device('cpu')
        X_t = torch.from_numpy(X_train).float()

        # fit aeon MiniRocket on TRAIN
        if 'global' in self.blocks and self.global_bank is not None:
            self.global_bank.fit(X_train)
            # compute actual output dim from a small batch
            Z_sample = self.global_bank.transform(X_train[:2])
            self._global_dim = Z_sample.shape[1]

        # transport reference
        if 'local_g4' in self.blocks or 'local' in self.blocks:
            self.transport_flavors = TransportFlavors(
                lag_set=self.lag_set, ref_grid=10)
            self.transport_flavors.fit_reference(X_t)

        # local strength medians
        if 'local' in self.blocks and self.local_bank is not None:
            ppv_w = max(3, round(self.window_frac * (seq_len or X_train.shape[-1])))
            self._ppv_w = ppv_w
            self.activity = LocalActivity(ppv_w)
            self.local_bank = self.local_bank.to(dev)
            meds = []
            for i in range(0, len(X_train), 256):
                xb = X_t[i:i+256].to(dev)
                R = self.local_bank(xb)
                _, S = self.activity(R)
                meds.append(S.flatten(0, 2))
            all_s = torch.cat(meds)
            self.s_med = all_s.median().cpu()

        self._fitted = True
        self._seq_len = seq_len or X_train.shape[-1]
        return self

    # ================================================================ extract
    @torch.no_grad()
    def extract(self, X, batch=256):
        """X: [N, 1, T] numpy -> dict of block_name -> [N, D_block] numpy.

        Also returns spec dict with dimensions.
        """
        assert self._fitted, "Call fit_train() first"
        dev = torch.device('cpu')
        out = {b: [] for b in self.blocks}
        t0 = time.time()

        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).float().to(dev)
            B = xb.shape[0]

            # ── A. global stream (aeon MiniRocket) ───────────────
            if 'global' in self.blocks and self.global_bank is not None:
                # MiniRocketGlobal.transform expects [N, 1, T] numpy
                X_np = xb.cpu().numpy().astype(np.float32)
                Zg = self.global_bank.transform(X_np)  # [B, 2016*84]
                out['global'].append(Zg)

            # ── B. local stream ───────────────────────────────────
            if 'local' in self.blocks and self.local_bank is not None:
                R = self.local_bank(xb)  # [B, M_l, T']
                A, S = self.activity(R)
                s_med = self.s_med.to(dev).view(1, -1, 1)
                S_norm = S / (s_med + 1e-8)
                # PPV stats + strength stats = 3*M_l + 3*M_l + routing extras
                Z_ppv = _stats_torch(A)   # [B, 4*M_l]
                Z_str = _stats_torch(S_norm)  # [B, 4*M_l]
                Z_local = torch.cat([Z_ppv, Z_str], dim=1)
                out['local'].append(Z_local.cpu().numpy())

            # ── C. local G4 stream ────────────────────────────────
            if 'local_g4' in self.blocks and self.transport_flavors is not None:
                Z_g4 = self._extract_local_g4(xb)
                out['local_g4'].append(Z_g4.cpu().numpy())

        # concatenate chunks
        result = {}
        for b in self.blocks:
            result[b] = np.concatenate(out[b], 0).astype(np.float32)

        elapsed = time.time() - t0
        spec = {b: int(result[b].shape[1]) for b in self.blocks}
        spec['total'] = sum(spec.values())
        spec['elapsed_s'] = round(elapsed, 2)
        return result, spec

    @torch.no_grad()
    def _extract_local_g4(self, xb):
        """Compute LOCAL G4 multi-lag drift features.

        For each local window:
          1. Extract drift view from TransportBuilder
          2. Compute multi-lag |drift(t) - drift(t-lag)| per lag
          3. Normalize by train reference
          4. Aggregate per-window statistics
        """
        dev = xb.device
        B, _, T = xb.shape

        # get drift view (full temporal resolution)
        views = self.builder(xb)  # [B, 3, T]
        drift = views[:, 2, :]   # [B, T]

        # local window size
        k = max(3, round(self.window_frac * T))

        # compute multi-lag drift displacement for each window
        # use sliding window with stride=1 for maximum temporal resolution
        pad = k // 2
        drift_padded = F.pad(drift, (pad, pad), mode='reflect')  # [B, T+2*pad]

        # unfold into windows: [B, T, k]
        win = drift_padded.unfold(1, k, 1)  # [B, T, k]

        # for each window, compute multi-lag drift features
        # ref_lag normalization
        ref_lag = self.transport_flavors.ref_lag.to(dev)
        ref_norm = ref_lag.abs().mean() + 1e-8

        lag_feats = []
        for lag in self.lag_set:
            if lag < k:
                # within-window lag displacement
                shifted = torch.roll(win, shifts=lag, dims=-1)
                lag_disp = (win - shifted).abs()  # [B, T, k]
                # aggregate: mean over window, then stats over time
                lag_mean = lag_disp.mean(-1)  # [B, T]
                lag_feats.append(lag_mean / ref_norm)

        if lag_feats:
            lag_stack = torch.stack(lag_feats, dim=1)  # [B, n_lags, T]
            # aggregate statistics over time dimension
            Z_g4 = _stats_torch(lag_stack)  # [B, n_lags * 4]
        else:
            Z_g4 = torch.zeros(B, 4 * len(self.lag_set), device=dev)

        return Z_g4

    # ================================================================ helpers
    def block_dims(self):
        """Return dict of block_name -> feature dimensionality."""
        if not hasattr(self, '_cached_dims'):
            raise RuntimeError("Call extract() first to populate dims")
        return self._cached_dims

    def spec_hash(self):
        h = hashlib.sha256()
        if self.global_bank is not None:
            h.update(f"minirocket_{self.global_bank.n_kernels}_{self.global_bank.seed}".encode())
        if self.local_bank is not None:
            h.update(self.local_bank.spec_hash.encode())
        h.update(str(self.blocks).encode())
        h.update(str(self.cfg).encode())
        return h.hexdigest()[:16]
