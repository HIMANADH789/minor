"""TURS-RRMT — Regime-Routed Multi-Transport TURS (next-generation architecture).

Central principle (deliberately different from TURS-Lite/Stack):
    Do NOT compress local temporal evidence into a small regime latent before
    transport. Expose many local patterns first; use their local activation to
    condition WHICH transport geometry applies; compress only after the rich,
    regime-conditioned transport representation has been formed.

Pipeline:
  X -> fixed MiniROCKET-inspired pattern bank (0 trainable params)
      -> local PPV A(t) + response strength S(t)  = P(t) [B, T, 2M]
      -> small regime router -> routing weights w(t) in Delta^(J-1)
      -> J transport flavors T^(j)(t) (quantile geometry from TransportBuilder)
      -> regime-conditioned combination + top-K response preservation
      -> pattern residual path F_P(t)
      -> late mean/max/std pooling -> h
      -> small linear classifier

Variants (ablation ladder) share EVERYTHING except the stated difference:
  A1 pattern_only        no transport at all
  A2 single_transport    one flavor (standard), no routing
  A3 uniform             all J flavors, uniform weights (CRITICAL CONTROL)
  A4 routed              learned regime-conditioned routing + top-K (THE MODEL)
  A5 no_topk             learned routing but collapse sum_j w_j T_j immediately
  A6/A7 are inference-time interventions on the frozen A4 (shuffled/fixed
  routing) — implemented in the evaluation code, not separate models.

Diagnostic semantics (documented mapping, not inherited blindly):
  regime state  z_t := local pattern activity A(t) [B, T, M] (control metadata)
  reliance      replaces scalar alpha with the routing vector w(t) [B, T, J]
  uncertainty   standard predictive confidence/entropy/margin (no learned gate)

Non-negotiables:
  - fixed pattern bank: deterministic seed, no gradients, saved hash
  - no GRU/LSTM/attention/temporal downsampling
  - transport references (quantile template) fit on TRAIN only
  - no learned low-dimensional bottleneck before pattern detection/transport
"""

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.tursnet import TransportBuilder  # canonical [raw, sorted, drift] views


# ============================================================
# Stage B: fixed MiniROCKET-inspired local pattern bank
# ============================================================
def build_pattern_bank(M=128, lengths=(7, 11, 15, 23, 31), in_channels=1, seed=42):
    """Deterministic fixed kernel bank. Returns weight tensor [M, C, L_max].

    MiniROCKET-inspired (not claimed equivalent): random sign/bias from a fixed
    seed, biased toward length-7/9 taps, multi-scale lengths and dilations.
    Zero trainable parameters. Dilation/length assignment is recorded in the
    bank spec so the receptive-field diversity is auditable.
    """
    g = torch.Generator().manual_seed(seed)
    per = M // len(lengths)
    kernels, spec = [], []
    for li, L in enumerate(lengths):
        n = per if li < len(lengths) - 1 else M - per * (len(lengths) - 1)
        for _ in range(n):
            k = torch.zeros(1, L)
            # MiniROCKET-style weight distribution
            n_sel = min(3, L)
            idx = torch.randperm(L, generator=g)[:n_sel]
            k[0, idx] = torch.randn(n_sel, generator=g) * (2.0 / math.sqrt(n_sel))
            kernels.append(k)
            spec.append(dict(length=int(L)))
    W = torch.cat([k.permute(1, 0) for k in kernels], dim=0)      # [M, L_var]
    W_full = torch.zeros(M, in_channels, max(lengths))
    for i, s in enumerate(spec):
        L = s["length"]
        W_full[i, 0, -L:] = W[i, -L:]                              # right-align
    return W_full, spec


class FixedPatternBank(nn.Module):
    """Fixed (non-trainable) multi-length conv bank -> full-resolution responses."""

    def __init__(self, M=128, lengths=(7, 11, 15, 23, 31), in_channels=1, seed=42):
        super().__init__()
        self.M = M
        W, spec = build_pattern_bank(M, lengths, in_channels, seed)
        # one depthwise conv per distinct length (grouped), weights frozen
        self.lengths = sorted(set(s["length"] for s in spec))
        self.spec = spec
        self.convs = nn.ModuleList()
        self._offsets = []
        for L in self.lengths:
            idx = [i for i, s in enumerate(spec) if s["length"] == L]
            w = W[idx]                                            # [n, 1, L]
            conv = nn.Conv1d(in_channels, len(idx), L, padding=0, bias=False,
                             groups=in_channels if in_channels == 1 else 1)
            conv.weight.data = w
            conv.weight.requires_grad_(False)
            self.convs.append(conv)
            self._offsets.append((L, idx))
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self._spec_hash = hashlib.sha256(W.numpy().tobytes()).hexdigest()[:16]

    @property
    def spec_hash(self):
        return self._spec_hash

    def forward(self, x):
        """x: [B, C, T] -> R [B, M, T'] (full temporal resolution preserved)."""
        outs = []
        T = x.shape[-1]
        for conv, (L, idx) in zip(self.convs, self._offsets):
            r = conv(x)                                           # [B, n, T-L+1]
            # left-pad so all lengths share the same T'
            pad = T - L + 1 - r.shape[-1]
            if pad > 0:
                r = F.pad(r, (0, 0))
            outs.append((idx, r))
        Tp = min(o[1].shape[-1] for o in outs)
        R = torch.zeros(x.shape[0], self.M, Tp, device=x.device, dtype=x.dtype)
        for idx, r in outs:
            R[:, idx, :] = r[:, :, -Tp:]
        return R


# ============================================================
# Stage C/D: local PPV + response strength -> P(t)
# ============================================================
class LocalActivity(nn.Module):
    """Local PPV window W_ppv and |response| strength, both [B, T', M].

    Uses average pooling over a sliding window (local statistic, not global).
    """

    def __init__(self, ppv_window):
        super().__init__()
        self.ppv_window = max(1, int(ppv_window))

    def forward(self, R):
        # R: [B, M, T']
        pos = (R > 0).float()
        k = self.ppv_window
        A = F.avg_pool1d(pos, kernel_size=k, stride=1, count_include_pad=True,
                         padding=k // 2)                        # local PPV
        S = F.avg_pool1d(R.abs(), kernel_size=k, stride=1,
                         count_include_pad=True, padding=k // 2)
        return A, S


# ============================================================
# Stage F: four transport flavors (quantile geometry on TransportBuilder views)
# ============================================================
QUANT_GRID_COARSE = torch.linspace(0.05, 0.95, 10)
QUANT_GRID_FINE = torch.linspace(0.02, 0.98, 25)
TAIL_WEIGHTS = torch.tensor([2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0])
TAIL_WEIGHTS = TAIL_WEIGHTS / TAIL_WEIGHTS.sum() * 10.0  # normalized to grid size


class TransportFlavors(nn.Module):
    """J=4 fixed-geometry transport flavors over local windows.

    Base views from models.tursnet.TransportBuilder: raw / sorted / drift.
    For each flavor and window we compare the window's local quantile profile
    against a TRAIN-fitted reference template (closed-form 1-D transport cost).

      T1 standard    : W1 vs reference, coarse grid (10 quantiles), raw view
      T2 tail        : tail-weighted quantile transport, coarse grid, raw view
      T3 fine        : W1 vs reference, fine grid (25 quantiles), raw view
      T4 multilag    : |sorted - lagged sorted| drift displacement vs reference
    """

    FLAVORS = ["standard", "tail", "fine", "multilag"]

    def __init__(self, lag_set=(1, 2, 4, 8), ref_grid=10):
        super().__init__()
        self.builder = TransportBuilder()
        self.lag_set = list(lag_set)
        self.ref_grid = ref_grid
        # TRAIN-fitted reference templates (registered as buffers after fit)
        self.register_buffer("ref_q", torch.zeros(ref_grid))
        self.register_buffer("ref_q_fine", torch.zeros(25))
        self.register_buffer("ref_lag", torch.zeros(len(lag_set)))
        self.fitted = False

    @torch.no_grad()
    def fit_reference(self, X_train):
        """Fit quantile/lag reference templates on TRAIN only."""
        dev = X_train.device
        with torch.no_grad():
            views = self.builder(X_train.to(dev))              # [B, 3, T]
            raw = views[:, 0, :]
            n = min(400, raw.shape[0])
            sub = raw[torch.randperm(raw.shape[0])[:n]]
            qs_c = torch.quantile(sub, QUANT_GRID_COARSE.to(dev), dim=1)  # [10, B]
            qs_f = torch.quantile(sub, QUANT_GRID_FINE.to(dev), dim=1)    # [25, B]
            self.ref_q.copy_(qs_c.mean(1))
            self.ref_q_fine.copy_(qs_f.mean(1))
            sx, _ = torch.sort(raw, dim=1)
            lags = []
            for k in self.lag_set:
                if k < sx.shape[1]:
                    lags.append((sx[:, k:] - sx[:, :-k]).abs().mean())
                else:
                    lags.append(torch.zeros((), device=dev))
            self.ref_lag.copy_(torch.stack(lags))
            self.fitted = True

    def forward(self, x, window):
        """x: [B, 1, T] -> T^(j): [B, J, T'] local transport features."""
        B, _, T = x.shape
        views = self.builder(x)                                # [B, 3, T]
        raw, sorted_x, drift = views[:, 0:1, :], views[:, 1, :], views[:, 2, :]
        k = max(1, int(window))
        # local quantile profile via patch unfolding: windows of length k
        Tp = T                                                 # keep resolution
        pad = k // 2
        raw_p = F.pad(raw.squeeze(1), (pad, pad), mode="reflect")
        # unfold -> [B, T, k] windows
        win = raw_p.unfold(1, k, 1)                            # [B, T, k]
        q_coarse = torch.quantile(win, QUANT_GRID_COARSE.to(x.device), dim=2)
        q_coarse = q_coarse.permute(1, 0, 2)                   # [10, B, T] -> [B, 10, T]
        q_fine = torch.quantile(win, QUANT_GRID_FINE.to(x.device), dim=2)
        q_fine = q_fine.permute(1, 0, 2)                       # [B, 25, T]

        ref_q = self.ref_q.to(x.device).view(1, -1, 1)
        T1 = (q_coarse - ref_q).abs().mean(1)                  # [B, T] W1 coarse
        w_tail = TAIL_WEIGHTS.to(x.device).view(1, -1, 1)
        T2 = ((q_coarse - ref_q).abs() * w_tail).sum(1)        # [B, T] tail-weighted
        ref_fine = self.ref_q_fine.to(x.device).view(1, -1, 1)
        T3 = (q_fine - ref_fine).abs().mean(1)                 # [B, T] fine-grid W1
        # T4: multi-lag drift displacement, aggregated over lag set
        d = drift                                              # [B, T]
        T4 = torch.zeros(B, T, device=x.device)
        for i, lag in enumerate(self.lag_set):
            if lag < T:
                shifted = torch.roll(d, shifts=lag, dims=1)
                T4 += (d - shifted).abs()
        T4 /= len(self.lag_set)
        T4 = T4 / (self.ref_lag.to(x.device).abs().mean() + 1e-8)

        return torch.stack([T1, T2, T3, T4], dim=1)            # [B, J=4, T]


# ============================================================
# Full TURS-RRMT model
# ============================================================
class TURSRRMT(nn.Module):
    """Regime-Routed Multi-Transport TURS.

    variant:
      A1 pattern_only | A2 single_transport | A3 uniform | A4 routed | A5 no_topk
    """

    FLAVORS = ["standard", "tail", "fine", "multilag"]

    def __init__(self, num_classes, seq_len, M=128, J=4, topk=2,
                 ppv_window=None, router_hidden=32, pattern_hidden=64,
                 transport_dim=32, variant="A4", seed=42, dropout=0.1,
                 head_width=192):
        super().__init__()
        assert variant in ("A1", "A2", "A3", "A4", "A5")
        self.variant = variant
        self.J, self.M, self.topk = J, M, topk
        self.seq_len = seq_len
        ppv_window = ppv_window or max(3, round(0.05 * seq_len))
        self.ppv_window = ppv_window

        self.bank = FixedPatternBank(M=M, seed=seed)
        self.activity = LocalActivity(ppv_window)
        self.flavors = TransportFlavors(lag_set=(1, 2, 4, 8))

        P_dim = 2 * M
        self.router = nn.Sequential(
            nn.Linear(P_dim, router_hidden), nn.ReLU(inplace=True),
            nn.Linear(router_hidden, J))
        self.pattern_path = nn.Sequential(
            nn.Linear(P_dim, pattern_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(pattern_hidden, pattern_hidden // 2))
        # per-variant transport projection input channels (A2: 1 flavor;
        # A3/A5: J flavors; A4: J kept + 1 rest = J+1; A1: unused)
        t_in = 1 if variant == "A2" else (J + 1 if variant == "A4" else J)
        self.transport_proj = nn.Conv1d(t_in, transport_dim, 1)
        d_total = transport_dim + pattern_hidden // 2
        self.head = nn.Sequential(
            nn.Linear(d_total * 3, head_width), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(head_width, head_width // 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(head_width // 2, num_classes))

    # ---------------- core forward ----------------
    def _features(self, x, diag=False):
        """Returns dict with h and all intermediates (diag mode keeps temporals)."""
        B, _, T = x.shape
        R = self.bank(x)                                       # [B, M, T']
        A, S = self.activity(R)                                # [B, M, T']
        # per-window normalization of S by train-median is baked into flavor fit;
        # use robust scale via log1p to avoid train/test stats leakage
        P = torch.cat([A, torch.log1p(S)], dim=1)              # [B, 2M, T']

        Tv = self.flavors(x, window=self.ppv_window)           # [B, J, T']
        Tp = min(P.shape[-1], Tv.shape[-1])
        P, Tv = P[..., :Tp], Tv[..., :Tp]

        if self.variant == "A1":
            # no transport: zero-pad the F_T slot so ALL variants share the
            # exact same classifier (fair-capacity comparison)
            F_T = torch.zeros(B, self.transport_proj.out_channels, Tp,
                              device=x.device)
            w, top2, rest = None, None, None
        elif self.variant == "A2":
            F_T = self.transport_proj(Tv[:, :1, :])            # flavor 1 only
            w, top2, rest = None, None, None
        elif self.variant == "A3":
            w = torch.full((B, self.J, Tp), 1.0 / self.J, device=x.device)
            F_T = self.transport_proj(w * Tv)
            top2, rest = None, None
        elif self.variant == "A5":
            w = torch.softmax(self.router(P.permute(0, 2, 1)), -1).permute(0, 2, 1)
            F_T = self.transport_proj(w * Tv)                  # collapse all flavors
            top2, rest = None, None
        else:  # A4: learned routing + top-K preservation
            w = torch.softmax(self.router(P.permute(0, 2, 1)), dim=-1).permute(0, 2, 1)
            top2 = w.topk(min(self.topk, self.J), dim=1).indices        # [B, K, T]
            mask = torch.zeros_like(w).scatter_(1, top2, 1.0)
            kept = (w * mask * Tv)                                   # [B, J, T]
            rest = w.sum(1, keepdim=True) - (w * mask).sum(1, keepdim=True)
            F_T = self.transport_proj(torch.cat([kept, rest], dim=1))  # [J+1 -> d]

        if F_T is not None:
            F_P = self.pattern_path(P.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 32, T]
            F = torch.cat([F_T, F_P], dim=1)
        else:
            F = self.pattern_path(P.permute(0, 2, 1)).permute(0, 2, 1)

        mu = F.mean(-1)
        mx = F.max(-1).values
        sd = F.std(-1) if F.shape[-1] > 1 else torch.zeros_like(mu)
        h = torch.cat([mu, mx, sd], dim=1)

        out = dict(h=h, w=w if self.variant in ("A4", "A5") else None,
                   F_T=F_T, F_P=None if F_T is None else F_P,
                   topk=top2 if self.variant == "A4" else None)
        if diag:
            out.update(R=R, A=A, S=S, P=P, Tv=Tv)
        return out

    def forward(self, x, return_aux=False):
        o = self._features(x, diag=False)
        logits = self.head(o["h"])
        if return_aux:
            return logits, o
        return logits

    def forward_diag(self, x):
        """Full diagnostic extraction (temporals preserved)."""
        o = self._features(x, diag=True)
        logits = self.head(o["h"])
        return logits, o


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, fixed
