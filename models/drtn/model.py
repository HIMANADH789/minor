"""
Discrete Regime Trajectory Network (DRTN)
==========================================

First implementation probe. Models R0-R5 (spec sec. 4-15):

    R0: encoder -> GAP -> linear                (neural floor)
    R1: encoder -> temporal attention -> linear (temporal preserved)
    R2: encoder -> SOFT codebook -> attention   (continuous regimes, collapse control)
    R3: encoder -> HARD VQ (commitment only)    (discreteness alone)
    R4: R3 + population-diversity regularizer   (the proposed mechanism)
    R5: R4 + causal trajectory transformer      (full DRTN)

Design constraints honored (spec sec. 13, 31, 32):
  - No downsampling anywhere: encoder preserves T_out == T.
  - Causal convolutions + causal attention only (no future leakage).
  - Instance level: each timestep picks exactly ONE code (hard VQ) —
    no soft mixtures in the forward representation.
  - Population level: diversity regularizes the BATCH usage histogram only,
    never per-timestep entropy. It never touches labels (sec. 23).
  - EMA codebook updates + dead-code revival with logged revivals (sec. 10-11).
  - No velocity/phase/uncertainty/quantile/drift hand-crafted features.
  - No reconstruction/contrastive/auxiliary losses: only CE + commitment + diversity.

Every model exposes `extract_diagnostics(dl, device)` so the runner records the
required per-epoch codebook statistics (usage histogram, entropies, perplexity)
and per-sample regime trajectories without reaching into internals ad hoc.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal building blocks
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """kernel-size-k, dilation-d causal conv: left-pads by (k-1)*d only."""

    def __init__(self, c_in, c_out, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalBlock(nn.Module):
    """CausalConv -> LayerNorm -> GELU, with an optional residual.

    Normalization is applied PER TIMESTEP over channels (LayerNorm on the
    (B, T, C) view). GroupNorm(1, C) on (B, C, T) would normalize over the
    whole time axis and leak future statistics into earlier positions —
    caught by the causality unit test (sec. 26 test 1).
    """

    def __init__(self, c_in, c_out, kernel_size=3, dilation=1, residual=True):
        super().__init__()
        self.conv = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.norm = nn.LayerNorm(c_out)              # per-position, causal
        self.act = nn.GELU()
        self.residual = residual and c_in == c_out

    def forward(self, x):
        y = self.conv(x)                             # (B, C, T)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        y = self.act(y)
        return y + x if self.residual else y


class CausalEncoder(nn.Module):
    """Multi-scale dilated causal conv stem (sec. 5).

    Input  X in R^(B, 1, T)
    Output Z in R^(B, T, D)   -- T preserved exactly, no downsampling.
    Dilations 1, 2, 4, 8; kernel 3; light channel count; residual where shapes
    allow. Dilation 1 block changes channels so no residual there.
    """

    def __init__(self, c_in=1, d_model=64, channels=(16, 16, 16, 16),
                 dilations=(1, 2, 4, 8), kernel_size=3):
        super().__init__()
        assert len(channels) == len(dilations)
        blocks, prev = [], c_in
        for ch, dil in zip(channels, dilations):
            blocks.append(CausalBlock(prev, ch, kernel_size, dil,
                                      residual=(prev == ch)))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.proj = CausalConv1d(prev, d_model, 1)   # pointwise, causal-safe

    def forward(self, x):                            # (B, C_in, T)
        for b in self.blocks:
            x = b(x)
        return self.proj(x).transpose(1, 2)          # (B, T, D)


class TemporalAttentionPool(nn.Module):
    """Learned scalar attention pooling (sec. 7/15): e_t = w^T tanh(W z_t)."""

    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1, bias=False)

    def forward(self, h):                            # (B, T, D)
        e = self.w(torch.tanh(self.W(h))).squeeze(-1)  # (B, T)
        a = torch.softmax(e, dim=-1)
        return (a.unsqueeze(-1) * h).sum(dim=1), a


# ---------------------------------------------------------------------------
# Codebooks
# ---------------------------------------------------------------------------
class SoftCodebook(nn.Module):
    """R2: continuous soft assignment (the control condition).

    s_tk = -||z_t - c_k||^2 ; a_t = softmax(s_t / tau); r_t = sum_k a_tk c_k.
    tau is FIXED at 0.5 (configurable, never tuned on test).
    """

    def __init__(self, d_model, n_codes=8, tau=0.5):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(n_codes, d_model) * 0.1)
        self.tau = tau
        self.n_codes = n_codes

    def forward(self, z):                            # (B, T, D)
        d2 = torch.cdist(z, self.codes[None].expand(z.shape[0], -1, -1)) ** 2
        a = torch.softmax(-d2 / self.tau, dim=-1)    # (B, T, K)
        r = a @ self.codes                           # (B, T, D)
        return r, a

    def diag_batch(self, a):
        """Per-batch usage stats from soft assignment a (B, T, K)."""
        q = a.mean(dim=(0, 1)).clamp_min(0).detach().cpu()
        q = q / q.sum().clamp_min(1e-12)
        return usage_stats(q)


class HardVQ(nn.Module):
    """R3-R5: discrete VQ with straight-through estimator (sec. 9-11).

    - assignment k_t = argmin ||z_t - c_k||^2  (exactly one code per timestep)
    - straight-through: q_ST = z + sg(q - z)
    - codebook updated by EMA (decay configurable), NOT by gradients
    - dead codes revived from a random current-batch latent, every revival
      logged with step/code/previous usage/source (sec. 11: never silent)
    - commitment loss  beta * ||sg[z] - c_k||^2  with beta = 0.25
    """

    def __init__(self, d_model, n_codes=8, ema_decay=0.99, beta=0.25,
                 dead_threshold=1e-3, revival_patience=100, eps=1e-5):
        super().__init__()
        self.n_codes = n_codes
        self.ema_decay = ema_decay
        self.beta = beta
        self.dead_threshold = dead_threshold
        self.revival_patience = revival_patience
        self.eps = eps

        embed = torch.randn(n_codes, d_model) * 0.1
        self.register_buffer("codes", embed)             # (K, D)
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_sum", embed.clone())
        self.register_buffer("dead_steps", torch.zeros(n_codes, dtype=torch.long))

        self.revival_log = []        # dicts: step, code, prev_usage, source
        self.total_revivals = 0

    @torch.no_grad()
    def ema_step(self, z_flat, assign, step):
        """Canonical EMA update (sec. 10):
            n_k <- decay*n_k + (1-decay)*count_k
            m_k <- decay*m_k + (1-decay)*sum_{t: k_t=k} z_t
            c_k = m_k / (n_k + eps)
        followed by dead-code revival accounting (sec. 11).
        """
        K = self.n_codes
        decay = self.ema_decay
        onehot = F.one_hot(assign, K).float()
        counts = onehot.sum(0)
        sums = onehot.T @ z_flat
        self.ema_count.mul_(decay).add_(counts * (1 - decay))
        self.ema_sum.mul_(decay).add_(sums * (1 - decay))
        self.codes.copy_(self.ema_sum / self.ema_count.unsqueeze(1).clamp_min(self.eps))

        # dead-code accounting: EMA usage below threshold for N consecutive steps.
        # `prev_healthy` snapshots who was healthy BEFORE this update so the
        # reset does not wipe patience of codes revived in THIS call.
        prev_healthy = self.ema_count >= self.dead_threshold
        dead = ~prev_healthy
        self.dead_steps[prev_healthy] = 0      # reset patience for healthy codes
        self.dead_steps += dead.long()         # accumulate for dead codes
        revived = []
        impatient = self.dead_steps >= self.revival_patience
        if impatient.any():
            for k in impatient.nonzero(as_tuple=False).flatten().tolist():
                prev_usage = float(self.ema_count[k])
                src = z_flat[torch.randint(0, z_flat.shape[0], (1,),
                                           device=z_flat.device)[0]]
                self.codes[k].copy_(src)
                self.ema_sum[k].copy_(src)
                self.ema_count[k] = self.dead_threshold
                self.dead_steps[k] = 0
                self.total_revivals += 1
                revived.append(int(k))
                self.revival_log.append({
                    "step": int(step),
                    "code": int(k),
                    "prev_usage": prev_usage,
                    "source": "current_batch_latent",
                })
        return revived

    def quantize(self, z):
        """Returns (q_st, assign, commit_loss, a_hard_onehot).

        q_st   : straight-through embeddings, (B, T, D)
        assign : (B, T) int64 — exactly one code per timestep
        """
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)
        d2 = (z_flat.pow(2).sum(1, keepdim=True)
              - 2.0 * (z_flat @ self.codes.T)
              + self.codes.pow(2).sum(1).unsqueeze(0))
        assign = d2.argmin(1)
        q = self.codes[assign]
        # RAW commitment MSE; callers multiply by self.beta exactly once.
        commit = F.mse_loss(z_flat.detach(), q)
        q_st = z_flat + (q - z_flat).detach()
        q_st = q_st.reshape(B, T, D)
        assign = assign.reshape(B, T)
        return q_st, assign, commit

    @torch.no_grad()
    def usage_from_assign(self, assign):
        """Batch usage histogram q_k over ALL timesteps in the batch (sec. 12)."""
        K = self.n_codes
        counts = torch.bincount(assign.reshape(-1), minlength=K).float()
        q = counts / counts.sum().clamp_min(1e-12)
        return q

    def diag_batch(self, assign):
        return usage_stats(self.usage_from_assign(assign))


def usage_stats(q):
    """Descriptive codebook diagnostics from a usage vector q (sums to 1).

    Accepts numpy arrays or torch tensors (any device; moved to CPU).
    Defensively re-normalizes when the input is an unnormalized count vector.
    Returns entropy H(q), normalized entropy H/log(K), perplexity exp(H),
    dominant fraction, active fraction, and the histogram itself.
    """
    if torch.is_tensor(q):
        q = q.detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float64)
    s = q.sum()
    if s > 0 and abs(s - 1.0) > 1e-6:
        q = q / s                                    # defensive normalization
    q = np.clip(q, 0.0, None)
    K = len(q)
    nz = q[q > 0]
    H = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    return {
        "usage": q.tolist(),
        "entropy": H,
        "normalized_entropy": H / math.log(K) if K > 1 else 1.0,
        "perplexity": float(np.exp(min(H, 700.0))),
        "perplexity_over_K": float(np.exp(min(H, 700.0))) / K,
        "active_codes": int((q > 0).sum()),
        "active_fraction": float((q > 0).mean()),
        "dominant_fraction": float(q.max()),
        "min_nonzero_usage": float(nz.min()) if len(nz) else 0.0,
    }


# ---------------------------------------------------------------------------
# Trajectory model (R5)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", None)

    def _mask(self, T, device):
        if self.mask is None or self.mask.shape[-1] != T:
            m = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            self.mask = m[None, None]
        return self.mask

    def forward(self, x):                            # (B, T, D)
        B, T, D = x.shape
        m = self._mask(T, x.device)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.n_heads)
        att = att.masked_fill(m, float("-inf"))
        att = att.softmax(-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class TrajectoryTransformer(nn.Module):
    """Small causal transformer over the code-embedding sequence (sec. 14).

    d_model=64, layers=2, heads=4, ffn=128, dropout=0.1, causal mask,
    sinusoidal positional encoding (no future information).
    """

    def __init__(self, d_model=64, n_layers=2, n_heads=4, ffn=128, dropout=0.1,
                 max_len=8192):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn, dropout,
            batch_first=True, norm_first=True, activation="gelu")
        # causal mask passed each call; nested-tensor fast path is incompatible
        # with norm_first and only produces a warning -> disable it
        self.layers = nn.TransformerEncoder(layer, n_layers,
                                            enable_nested_tensor=False)

    def forward(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device)
        x = x + self.pos(pos)[None]
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        return self.layers(x, mask=causal)


# ---------------------------------------------------------------------------
# Full models R0-R5
# ---------------------------------------------------------------------------
class DRTNBase(nn.Module):
    """Shared encoder + head scaffolding and diagnostics plumbing."""

    def __init__(self, c_in=1, n_classes=5, d_model=64):
        super().__init__()
        self.encoder = CausalEncoder(c_in, d_model)
        self.n_classes = n_classes
        self.is_vq = False

    def _head_from_pooled(self, h):
        return self.classifier(h)

    def embed(self, x):
        return self.encoder(x)

    def forward_with_assign(self, x):
        raise NotImplementedError

    def forward(self, x):
        logits, _ = self.forward_with_assign(x)
        return logits

    # -- diagnostics helpers shared by all variants -------------------------
    @torch.no_grad()
    def collect_batch_diag(self, x):
        """Returns (logits, extra) where extra carries variant diagnostics."""
        raise NotImplementedError

    @torch.no_grad()
    def extract_diagnostics(self, dl, device, max_batches=None,
                            save_trajectories=0, save_input=False):
        """Iterate a loader and aggregate model diagnostics (sec. 19-21).

        Always returns mean batch usage stats + loss breakdown when applicable.
        `save_trajectories=k` stores code assignments (+ optionally the raw
        input) for the first k examples for later visualization.
        """
        self.eval()
        acc_usage = None
        out = {}
        traj, inputs, attn_all = [], [], []
        n_seen = 0
        for i, (xb, _) in enumerate(dl):
            if max_batches is not None and i >= max_batches:
                break
            xb = xb.to(device)
            logits, extra = self.collect_batch_diag(xb)
            for key, val in extra.items():
                if key in ("usage", "assign", "attn", "soft_assign",
                           "logits"):
                    continue
                if isinstance(val, torch.Tensor):
                    if val.numel() != 1:
                        continue
                    val = float(val.detach().cpu())
                out[key] = out.get(key, 0.0) + float(val)
            if "usage" in extra:
                u = np.asarray(extra["usage"], dtype=np.float64)
                acc_usage = u if acc_usage is None else acc_usage + u
            if save_trajectories and n_seen < save_trajectories:
                take = min(save_trajectories - n_seen, xb.shape[0])
                if "assign" in extra:
                    traj.append(np.asarray(extra["assign"].detach().cpu())[:take])
                if "attn" in extra:
                    attn_all.append(np.asarray(extra["attn"].detach().cpu())[:take])
                if save_input:
                    inputs.append(xb.detach().cpu().numpy()[:take])
            n_seen += xb.shape[0]
        nb = max(i + 1, 1)
        means = {k: v / nb for k, v in out.items() if k != "usage"}
        if acc_usage is not None:
            q = acc_usage / acc_usage.sum()
            means.update(usage_stats(q))
            means["usage"] = q.tolist()
        if traj:
            means["trajectories"] = np.concatenate(traj, 0)
        if attn_all:
            means["attn"] = np.concatenate(attn_all, 0)
        if inputs:
            means["inputs"] = np.concatenate(inputs, 0)
        return means


# ---------------------------------------------------------------------------
# R0 / R1
# ---------------------------------------------------------------------------
class DRTN_R0(DRTNBase):
    """Encoder -> GAP -> Linear (neural floor)."""

    def __init__(self, c_in=1, n_classes=5, d_model=64):
        super().__init__(c_in, n_classes, d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward_with_assign(self, x):
        z = self.encoder(x)
        h = z.mean(dim=1)
        return self._head_from_pooled(h), {}

    def collect_batch_diag(self, x):
        logits, extra = self.forward_with_assign(x)
        extra["logits"] = logits
        return logits, extra


class DRTN_R1(DRTNBase):
    """Encoder -> temporal attention pool -> Linear (sec. 7)."""

    def __init__(self, c_in=1, n_classes=5, d_model=64):
        super().__init__(c_in, n_classes, d_model)
        self.pool = TemporalAttentionPool(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward_with_assign(self, x):
        z = self.encoder(x)
        h, a = self.pool(z)
        return self._head_from_pooled(h), {"attn": a}

    def collect_batch_diag(self, x):
        logits, extra = self.forward_with_assign(x)
        extra["logits"] = logits
        return logits, extra


# ---------------------------------------------------------------------------
# R2: soft codebook control
# ---------------------------------------------------------------------------
class DRTN_R2(DRTNBase):
    """Encoder -> soft codebook (tau=0.5) -> attention pool -> Linear."""

    def __init__(self, c_in=1, n_classes=5, d_model=64, n_codes=8, tau=0.5):
        super().__init__(c_in, n_classes, d_model)
        self.codebook = SoftCodebook(d_model, n_codes, tau)
        self.pool = TemporalAttentionPool(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward_with_assign(self, x):
        z = self.encoder(x)
        r, a = self.codebook(z)
        h, att = self.pool(r)
        return self._head_from_pooled(h), {"attn": att, "soft_assign": a}

    def collect_batch_diag(self, x):
        z = self.encoder(x)
        r, a = self.codebook(z)
        h, att = self.pool(r)
        logits = self.classifier(h)
        d = self.codebook.diag_batch(a)
        # mean per-timestep assignment entropy + max prob (sec. 8)
        ent_t = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1)
        d["mean_assignment_entropy"] = float(ent_t.mean())
        d["mean_max_assignment_prob"] = float(a.max(-1).values.mean())
        d["attn"] = att
        d["soft_assign"] = a.detach().cpu()
        d["logits"] = logits
        return logits, d


# ---------------------------------------------------------------------------
# R3/R4/R5: hard VQ family
# ---------------------------------------------------------------------------
class DRTNVQMixin:
    """Adds hard-VQ forward path + population-diversity loss + EMA hook."""

    def set_diversity(self, lam_div):
        self.lam_div = float(lam_div)

    def vq_forward(self, x):
        z = self.encoder(x)                          # (B, T, D), causal
        q_st, assign, commit = self.vq.quantize(z)
        return q_st, assign, commit

    def diversity_loss_from_assign(self, assign):
        """L_div = -H(q) over the BATCH usage histogram (sec. 12).

        Label-independent by construction: assign is derived from encoder
        latents only. Aggregated over the whole minibatch before entropy.
        """
        q = self.vq.usage_from_assign(assign)
        H = -(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum()
        return -H

    def collect_batch_diag(self, x):
        z = self.encoder(x)
        q_st, assign, commit = self.vq.quantize(z)
        h, att = self.head_forward(q_st)
        logits = self.classifier(h)
        q = self.vq.usage_from_assign(assign)
        d = usage_stats(q.detach().cpu().numpy())
        d["commitment_loss"] = float(commit)
        d["assign"] = assign.detach().cpu()
        d["attn"] = att
        d["logits"] = logits
        return logits, d


class DRTN_R3(DRTNVQMixin, DRTNBase):
    """Hard VQ + commitment only (beta=0.25), NO diversity (sec. 9).

    Mixin listed FIRST so its collect_batch_diag overrides the base stub
    (Python MRO: leftmost base wins).
    """

    def __init__(self, c_in=1, n_classes=5, d_model=64, n_codes=8,
                 ema_decay=0.99, beta=0.25, dead_threshold=1e-3,
                 revival_patience=100):
        super().__init__(c_in, n_classes, d_model)
        self.vq = HardVQ(d_model, n_codes, ema_decay, beta,
                         dead_threshold, revival_patience)
        self.pool = TemporalAttentionPool(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.lam_div = 0.0

    def head_forward(self, q_st):
        """Shared head path -> (pooled, attention)."""
        return self.pool(q_st)

    def pool_and_head(self, q_st):
        h, _ = self.head_forward(q_st)
        return h

    def forward_with_assign(self, x):
        q_st, assign, commit = self.vq_forward(x)
        h, att = self.pool(q_st)
        return self.classifier(h), {"assign": assign, "commit": commit,
                                    "attn": att}


class DRTN_R4(DRTN_R3):
    """R3 + population-diversity regularizer (sec. 12) — the contribution."""

    def __init__(self, c_in=1, n_classes=5, d_model=64, n_codes=8,
                 ema_decay=0.99, beta=0.25, lam_div=0.01,
                 dead_threshold=1e-3, revival_patience=100):
        super().__init__(c_in, n_classes, d_model, n_codes, ema_decay, beta,
                         dead_threshold, revival_patience)
        self.lam_div = lam_div

    def loss_terms(self, logits, y, assign, commit):
        ce = F.cross_entropy(logits, y)
        div = self.diversity_loss_from_assign(assign)
        total = ce + self.vq.beta * commit + self.lam_div * div
        return total, {"ce": ce, "commit": commit, "div": div,
                       "commit_w": self.vq.beta * commit,
                       "div_w": self.lam_div * div}


class DRTN_R5(DRTN_R4):
    """Full DRTN: R4 + causal trajectory transformer + attention head (14-15)."""

    def __init__(self, c_in=1, n_classes=5, d_model=64, n_codes=8,
                 ema_decay=0.99, beta=0.25, lam_div=0.01,
                 dead_threshold=1e-3, revival_patience=100,
                 traj_layers=2, traj_heads=4, traj_ffn=128, traj_dropout=0.1):
        super().__init__(c_in, n_classes, d_model, n_codes, ema_decay, beta,
                         lam_div, dead_threshold, revival_patience)
        self.trajectory = TrajectoryTransformer(
            d_model, traj_layers, traj_heads, traj_ffn, traj_dropout)
        self.pool = TemporalAttentionPool(d_model)   # head over H, re-built

    def head_forward(self, q_st):
        H = self.trajectory(q_st)                    # (B, T, D), causal
        return self.pool(H)                          # attention over H

    def forward_with_assign(self, x):
        q_st, assign, commit = self.vq_forward(x)
        H = self.trajectory(q_st)
        h, att = self.pool(H)
        return self.classifier(h), {"assign": assign, "commit": commit,
                                    "attn": att}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(rung, c_in=1, n_classes=5, d_model=64, n_codes=8,
                tau=0.5, ema_decay=0.99, beta=0.25, lam_div=0.01,
                dead_threshold=1e-3, revival_patience=100,
                traj_layers=2, traj_heads=4, traj_ffn=128, traj_dropout=0.1):
    """Construct rung in {R0..R5} with the spec defaults."""
    rung = rung.upper()
    if rung == "R0":
        return DRTN_R0(c_in, n_classes, d_model)
    if rung == "R1":
        return DRTN_R1(c_in, n_classes, d_model)
    if rung == "R2":
        return DRTN_R2(c_in, n_classes, d_model, n_codes, tau)
    if rung == "R3":
        return DRTN_R3(c_in, n_classes, d_model, n_codes, ema_decay, beta,
                       dead_threshold, revival_patience)
    if rung == "R4":
        return DRTN_R4(c_in, n_classes, d_model, n_codes, ema_decay, beta,
                       lam_div, dead_threshold, revival_patience)
    if rung == "R5":
        return DRTN_R5(c_in, n_classes, d_model, n_codes, ema_decay, beta,
                       lam_div, dead_threshold, revival_patience,
                       traj_layers, traj_heads, traj_ffn, traj_dropout)
    raise ValueError(f"unknown rung {rung}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters()
                        if not p.requires_grad)
    return {"total": int(total), "trainable": int(trainable),
            "non_trainable": int(non_trainable)}
