"""Per-dataset SSL context model + regime extraction for the transfer screen.

Reuses, without modification, the validated Haptics R2 pieces:
    * SSLTemporalEncoder / MaskDecoder / ssl_masked_recon_loss /
      make_span_mask:  experiments/rcmkn_haptics_seed42/regime_encoder.py
    * RCMKNContextModel (encoder + HardVQ + decoder + aux head):
      experiments/rcmkn_haptics_seed42/model.py
    * HardVQ infrastructure: experiments/rcmkn_haptics_seed42/vq.py
    * joint training schedule: experiments/rcmkn_haptics_seed42/runner.py
      (train_context_model — identical loss weights and protocol)

This module only adapts *data plumbing* (batching over per-dataset sizes);
no architecture or loss changes are permitted or made.
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rcmkn_haptics_seed42.config import (  # noqa: E402
    SEED, ENCODER, VQ, JOINT, K_CODES)
from experiments.rcmkn_haptics_seed42.model import (  # noqa: E402
    RCMKNContextModel, parameter_report)
from experiments.rcmkn_haptics_seed42.vq import (  # noqa: E402
    hard_assign, occupancy_stats as _occupancy_stats)
from experiments.rcmkn_haptics_seed42.runner import (  # noqa: E402
    train_context_model, set_seed, extract_context_regimes)


def build_context_model(n_classes, device):
    """Fresh Haptics-R2 context model with the exact validated config."""
    model = RCMKNContextModel(n_classes=n_classes).to(device)
    return model


def train_ssl_context(model, Xtr_z, ytr, Xva_z, yva, device, smoke=False):
    """Train (SSL + VQ + aux) with the exact Haptics R2 schedule.

    Returns (train_info, params) from the validated runner code.
    """
    params = parameter_report(model)
    train_info = train_context_model(model, Xtr_z, ytr, Xva_z, yva,
                                     device, smoke=smoke)
    return train_info, params


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def context_regimes(model, X_z, device, batch=32):
    """Hard VQ regimes via the VALIDATED runner helper (no reimplementation).
    Returns (N, T) int array."""
    return extract_context_regimes(model, X_z, device, batch=batch)


def occupancy(regimes, K=K_CODES):
    """Aggregate code-usage diagnostics — validated vq.occupancy_stats."""
    return _occupancy_stats(regimes, K=K)


def make_mask_debug(T, mask_ratio=0.10, span_len=16, seed=SEED):
    """Deterministic mask for unit tests of the masked-span machinery."""
    from experiments.rcmkn_haptics_seed42.regime_encoder import make_span_mask
    g = torch.Generator().manual_seed(seed)
    return make_span_mask(4, T, mask_ratio, span_len, generator=g)
