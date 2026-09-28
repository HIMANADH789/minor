"""Stage 3: validated HardVQ reuse + EMA/revival audit helpers.

The HardVQ module from models/drtn/model.py is used AS-IS (EMA updates,
dead-code revival, straight-through assignment). This module only adds:
    - a thin wrapper exposing quantize/ema_step with the RCMKN config
    - one-step EMA correctness checks used in the unit tests
    - occupancy diagnostics
"""
import numpy as np
import torch

from models.drtn.model import HardVQ
from experiments.rcmkn_haptics_seed42.config import VQ


def build_vq(d_model=32):
    return HardVQ(d_model=d_model, n_codes=VQ["K"], ema_decay=VQ["ema_decay"],
                  beta=VQ["beta_vq"], dead_threshold=VQ["dead_threshold"],
                  revival_patience=VQ["revival_patience"])


@torch.no_grad()
def hard_assign(vq, z):
    """z: (B, T, D) -> (B, T) int64 hard code assignments (argmin d2)."""
    B, T, D = z.shape
    z_flat = z.reshape(-1, D)
    d2 = (z_flat.pow(2).sum(1, keepdim=True)
          - 2.0 * (z_flat @ vq.codes.T)
          + vq.codes.pow(2).sum(1).unsqueeze(0))
    return d2.argmin(1).reshape(B, T)


@torch.no_grad()
def ema_update(vq, z_flat, assign, eps=1e-5):
    """Reference EMA update (verified equivalent to HardVQ.ema_step).

    cluster_size <- decay * cluster_size + (1-decay) * counts
    embed_sum    <- decay * embed_sum    + (1-decay) * sums
    codes        <- embed_sum / cluster_size (eps-smoothed)
    """
    K, D = vq.codes.shape
    onehot = torch.zeros(len(assign), K, device=z_flat.device)
    onehot[torch.arange(len(assign)), assign] = 1.0
    counts = onehot.sum(0)
    sums = onehot.T @ z_flat
    vq.ema_count = vq.ema_decay * vq.ema_count + (1 - vq.ema_decay) * counts
    vq.ema_sum = vq.ema_decay * vq.ema_sum + (1 - vq.ema_decay) * sums
    n = vq.ema_count.sum()
    cluster = (vq.ema_count + eps) / (n + K * eps) * n
    vq.codes.copy_(vq.ema_sum / cluster.unsqueeze(1))


def occupancy_stats(regimes, K=8):
    """Diagnostics over a (N, T) regime array."""
    flat = regimes.reshape(-1)
    counts = np.bincount(flat, minlength=K).astype(np.float64)
    q = counts / max(counts.sum(), 1)
    nz = q[q > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "active_codes": int((counts > 0).sum()),
        "normalized_entropy": float(entropy / np.log(K)),
        "perplexity": float(np.exp(entropy)),
        "dominant_fraction": float(q.max()),
        "usage": q.tolist(),
        "counts": counts.tolist(),
    }
