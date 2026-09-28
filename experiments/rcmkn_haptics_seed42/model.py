"""RCMKN context model: Stage 2 encoder + Stage 3 VQ + SSL decoder.

The aux classification head exists ONLY to provide L_aux (lambda_cls) during
joint fine-tuning of the context model on TRAIN; it is discarded before
feature extraction and never enters the Ridge classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.rcmkn_haptics_seed42.config import ENCODER, VQ
from experiments.rcmkn_haptics_seed42.regime_encoder import (
    SSLTemporalEncoder, MaskDecoder, make_span_mask,
)
from experiments.rcmkn_haptics_seed42.vq import build_vq


class RCMKNContextModel(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.encoder = SSLTemporalEncoder(
            channels=ENCODER["channels"], kernel_sizes=ENCODER["kernel_sizes"],
            dilations=ENCODER["dilations"], d_model=ENCODER["d_model"])
        self.vq = build_vq(d_model=ENCODER["d_model"])
        self.decoder = MaskDecoder(d_model=ENCODER["d_model"])
        self.aux_head = nn.Linear(ENCODER["d_model"], n_classes)
        self.tau = 0.5    # stored for soft-assignment parity checks only

    def forward(self, x):
        """x: (B, 1, T) -> (logits_aux, z, q_st, assign, commit)."""
        z = self.encoder(x)
        q_st, assign, commit = self.vq.quantize(z)
        pooled = q_st.mean(dim=1)                 # temporal mean of VQ outputs
        logits = self.aux_head(pooled)
        return logits, z, q_st, assign, commit

    def ssl_loss(self, x_raw, mask_ratio=None, span_len=None):
        """Masked-span reconstruction loss (no labels). The mask RNG is a
        CPU generator (torch requirement); the mask is moved to the input's
        device afterwards."""
        mask_ratio = mask_ratio or ENCODER["mask_ratio"]
        span_len = span_len or ENCODER["span_len"]
        B, T = x_raw.shape
        gen = torch.Generator().manual_seed(
            int(torch.randint(0, 2**31 - 1, (1,)).item()))
        mask = make_span_mask(B, T, mask_ratio, span_len,
                              generator=gen).to(x_raw.device)
        x_in = x_raw.clone()
        x_in = x_in.masked_fill(mask, 0.0)
        z = self.encoder(x_in[:, None, :])
        pred = self.decoder(z)
        return F.mse_loss(pred[mask], x_raw[mask], reduction="mean")

    def diversity_loss(self, assign):
        """Population-level code-usage diversity: -H(batch histogram)."""
        q = self.vq.usage_from_assign(assign)
        return -(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum()

    def ema_step(self, z, assign, step):
        self.vq.ema_step(z.detach().reshape(-1, z.shape[-1]),
                         assign.reshape(-1), step=step)


def parameter_report(model):
    enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    aux = sum(p.numel() for p in model.aux_head.parameters() if p.requires_grad)
    vq_train = 0       # HardVQ: codebook is an EMA buffer, not trainable
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == enc + dec + aux + vq_train
    return {"encoder": enc, "decoder": dec, "aux_head": aux,
            "vq_trainable": vq_train, "total_trainable": total}
