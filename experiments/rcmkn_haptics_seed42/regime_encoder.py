"""Stage 2: small causal multi-scale encoder + masked-span SSL.

Design (predeclared in config.ENCODER):
    4 causal dilated Conv1d blocks (no downsampling, no stride>1)
    -> per-position LayerNorm over channels (no future mixing)
    -> GELU
    -> pointwise projection to D=32 -> Z in R^(B, T, 32)

SSL: masked-span reconstruction. Contiguous spans (~10% of positions) are
zero-masked on the INPUT; a small decoder predicts the original values ONLY
at masked positions. Labels never enter this loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.rcmkn_haptics_seed42.config import ENCODER


class CausalConv1d(nn.Conv1d):
    """Left-padded causal convolution (output t depends only on inputs <= t)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__(
            in_ch, out_ch, kernel_size, dilation=dilation,
            padding=(kernel_size - 1) * dilation, padding_mode="zeros")
        self._causal_trim = (kernel_size - 1) * dilation

    def forward(self, x):
        return super().forward(x)[..., :x.shape[-1]]


class _ChannelNorm(nn.Module):
    """Per-position LayerNorm over channels: (B, C, T) in -> (B, C, T) out.
    Normalizes each timestep independently over its channel vector (the
    DRTN CausalBlock convention) -- never mixes future positions."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SSLTemporalEncoder(nn.Module):
    """Causal dilated encoder with per-position channel normalization."""

    def __init__(self, channels=(32, 32, 64, 64), kernel_sizes=(3, 5, 7, 9),
                 dilations=(1, 2, 4, 8), d_model=32):
        super().__init__()
        blocks, in_ch = [], 1
        for ch, ks, dil in zip(channels, kernel_sizes, dilations):
            blocks.append(nn.Sequential(
                CausalConv1d(in_ch, ch, ks, dilation=dil),
                _ChannelNorm(ch),          # per-position channel norm
                nn.GELU(),
            ))
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.proj = CausalConv1d(in_ch, d_model, 1)

    def forward(self, x):
        """x: (B, 1, T) -> Z: (B, T, D)"""
        for b in self.blocks:
            x = b(x)
        return self.proj(x).transpose(1, 2)


class MaskDecoder(nn.Module):
    """Small reconstruction head; used ONLY during SSL training."""

    def __init__(self, d_model=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, z):
        """z: (B, T, D) -> (B, T) predicted raw values at every position."""
        return self.net(z).squeeze(-1)


def make_span_mask(B, T, mask_ratio=0.10, span_len=16, generator=None):
    """Boolean mask (B, T): True = masked. Contiguous spans of ~span_len.

    Spans start at multiples of span_len blocks; a block is masked with
    probability mask_ratio, keeping the realized ratio ~ mask_ratio.

    Bug fix (2026-09, ECG5000_BAL NaN investigation): for short series the
    block count is small (T=140, span_len=16 -> 9 blocks), so a sample had
    probability (1-mask_ratio)**n_blocks of drawing ZERO masked blocks
    (~39% at T=140 vs ~0.1% at T=1024). Starved samples silently
    contributed nothing to the SSL loss, and an all-starved batch made
    F.mse_loss return NaN. Each sample is now guaranteed at least one
    masked block whenever mask_ratio > 0. Rows that already contain a
    masked block are unchanged: the guarantee only draws additional RNG
    values AFTER the main draw, so non-starved masks are bit-identical to
    the previous implementation given the same generator state.

    The generator must be a CPU generator; the returned mask lives on CPU
    and is moved by the caller when needed.
    """
    n_blocks = int(np.ceil(T / span_len))
    block_mask = (torch.rand(B, n_blocks, generator=generator) < mask_ratio)
    if mask_ratio > 0:
        empty_rows = (~block_mask.any(dim=1)).nonzero(as_tuple=True)[0]
        if empty_rows.numel() > 0:
            forced = torch.randint(0, n_blocks, (int(empty_rows.numel()),),
                                   generator=generator)
            block_mask[empty_rows, forced] = True
    mask = block_mask.repeat_interleave(span_len, dim=1)[:, :T]
    return mask


def ssl_masked_recon_loss(encoder, decoder, x_raw, mask_ratio, span_len):
    """One SSL step. x_raw: (B, T) normalized signal.

    Returns (loss, diag). Loss = MSE over MASKED positions only.
    """
    B, T = x_raw.shape
    gen = torch.Generator(device=x_raw.device).manual_seed(
        int(torch.randint(0, 2**31 - 1, (1,)).item()))
    mask = make_span_mask(B, T, mask_ratio, span_len, generator=gen).to(x_raw.device)
    if not bool(mask.any()):
        raise RuntimeError(
            "empty SSL mask: make_span_mask per-sample guarantee violated")
    x_in = x_raw.clone()
    x_in[mask] = 0.0                       # zero-mask contiguous spans
    z = encoder(x_in[:, None, :])          # (B, T, D)
    pred = decoder(z)                      # (B, T)
    loss = F.mse_loss(pred[mask], x_raw[mask])
    return loss, {"mask_frac": float(mask.float().mean().item())}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
