"""
DRTN controlled-experiment models
=================================

Two capacity-matched controls that isolate WHERE the R5 gain comes from:

    DRTN_CTC  "Continuous Trajectory Control"
        encoder -> Z (continuous, D=64) -> SAME causal TrajectoryTransformer
                -> SAME attention pool -> SAME linear classifier -> CE only.
        No codebook, no VQ, no commitment, no diversity, no aux losses.

    DRTN_DTC  "Discrete Trajectory Control"
        encoder -> hard VQ (K=8, straight-through, EMA, revival, commitment)
                -> SAME causal TrajectoryTransformer -> SAME head -> CE + commit.
        NO population-diversity term (that is the R4->R5 ladder element).

Both reuse the EXACT classes from models.drtn.model — no duplicated
architecture. Parameter audit (same encoder/transformer/head as R5):

    encoder 3,632 + pool 4,224 + classifier 325 + trajectory 591,232
      = 599,413 total trainable for BOTH controls and R5.

R5's codebook (K*64 = 512 params) is an EMA buffer, not trainable, so
removing VQ does not change the trainable budget — matching is EXACT with
zero projection hacks. DTC additionally differs from R5 only by the absence
of the diversity loss (identical graph otherwise).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drtn.model import (
    DRTNBase, DRTN_R3, DRTN_R5, HardVQ, TemporalAttentionPool,
    TrajectoryTransformer, usage_stats,
)


class DRTN_CTC(DRTNBase):
    """Experiment 1: parameter-matched CONTINUOUS trajectory control.

    Identical encoder + causal TrajectoryTransformer + attention pool +
    classifier as R5. The transformer consumes the CONTINUOUS encoder
    sequence Z directly. Loss = CE only.
    """

    is_vq = False

    def __init__(self, c_in=1, n_classes=5, d_model=64,
                 traj_layers=2, traj_heads=4, traj_ffn=128, traj_dropout=0.1):
        super().__init__(c_in, n_classes, d_model)
        self.trajectory = TrajectoryTransformer(
            d_model, traj_layers, traj_heads, traj_ffn, traj_dropout)
        self.pool = TemporalAttentionPool(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.lam_div = 0.0

    def head_forward(self, z):
        H = self.trajectory(z)                       # (B, T, D), causal
        return self.pool(H)                          # attention over H

    def forward_with_assign(self, x):
        z = self.encoder(x)
        h, att = self.head_forward(z)
        return self.classifier(h), {"attn": att, "z_seq": z}

    def collect_batch_diag(self, x):
        z = self.encoder(x)
        h, att = self.head_forward(z)
        logits = self.classifier(h)
        d = {"attn": att, "logits": logits}
        return logits, d


class DRTN_DTC(DRTN_R5):
    """Experiment 2: parameter-matched DISCRETE trajectory control.

    Exactly R5's graph (encoder -> hard VQ -> trajectory -> head) with the
    population-diversity term REMOVED: L = CE + beta * L_commit.

    Implementation discipline: subclass R5 and force lam_div = 0 while also
    bypassing `loss_terms` so no diversity tensor is ever computed. This
    keeps encoder/VQ/transformer/head definitions bit-identical to R5 by
    construction (no copied code).
    """

    def __init__(self, c_in=1, n_classes=5, d_model=64, n_codes=8,
                 ema_decay=0.99, beta=0.25, dead_threshold=1e-3,
                 revival_patience=100,
                 traj_layers=2, traj_heads=4, traj_ffn=128, traj_dropout=0.1):
        super().__init__(c_in, n_classes, d_model, n_codes, ema_decay, beta,
                         lam_div=0.0, dead_threshold=dead_threshold,
                         revival_patience=revival_patience,
                         traj_layers=traj_layers, traj_heads=traj_heads,
                         traj_ffn=traj_ffn, traj_dropout=traj_dropout)
        self.lam_div = 0.0
        self.has_diversity = False

    def loss_terms(self, logits, y, assign, commit):
        """L = CE + beta * L_commit ONLY — no diversity term exists here."""
        ce = F.cross_entropy(logits, y)
        total = ce + self.vq.beta * commit
        return total, {"ce": ce, "commit": commit,
                       "commit_w": self.vq.beta * commit}


# ---------------------------------------------------------------------------
# Parameter audit by module group
# ---------------------------------------------------------------------------
def param_audit(model):
    """Exact per-module parameter counts + trainable/buffer breakdown.

    Codebook (codes/ema_count/ema_sum) are registered buffers in HardVQ ->
    they are NOT trainable parameters and are reported separately.
    """
    groups = {"encoder": 0, "vq": 0, "codebook_buffers": 0,
              "trajectory": 0, "pool": 0, "classifier": 0, "other": 0}
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            groups["encoder"] += p.numel()
        elif name.startswith("vq."):
            groups["vq"] += p.numel()
        elif name.startswith("trajectory."):
            groups["trajectory"] += p.numel()
        elif name.startswith("pool."):
            groups["pool"] += p.numel()
        elif name.startswith("classifier."):
            groups["classifier"] += p.numel()
        else:
            groups["other"] += p.numel()
    for name, buf in model.named_buffers():
        if name.startswith("vq."):
            groups["codebook_buffers"] += buf.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters()
                        if not p.requires_grad)
    groups["trainable"] = int(trainable)
    groups["non_trainable"] = int(non_trainable)
    groups["total"] = int(trainable + non_trainable)
    return groups
