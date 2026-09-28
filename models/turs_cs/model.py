"""
TURS-CS — Calibrate-then-Specialize Adaptive Multiscale Architecture
=====================================================================

Canonical implementation for biomedical benchmarks (ECG5000, CWRU).

Architecture:
  Transport Path: reuses TURS-Lite TransportBuilder + TransportEncoder
  Regime Path:
    - 3 depthwise-separable branches at scales σ, 3σ, 9σ
    - differentiable soft (interpolated) dilation
    - within-window EMA with dataset-level ρ₁, ρ₂, ρ₃
    - per-scale velocity
    - β scale-trust gate (softmax, conditioned on state + velocity)
    - fused regime z_t with local residual
    - fused velocity ṽ_t (same β)
    - bounded change c_t = tanh(P_c(ṽ_t))
    - uncertainty u_t = sigmoid(P_u([z_t ∥ ṽ_t]))
    - compact regime feature F_R with c_t gating
  Fusion:
    - transport-regime interaction I_TR (elementwise, no attention)
    - adaptive α fusion
  Classifier:
    - 65-D: GAP(F)→32, GAP(z_t)→16, GAP(ṽ_t)→16, GAP(u_t)→1
    - hidden → num_classes

Two-phase training:
  Phase 0: Calibration (σ/ρ at high LR, rest frozen/low LR)
  Phase 1: Specialization (σ/ρ frozen or very low LR)

Non-negotiable invariants:
  - No cross-window hidden state
  - No GRU/LSTM/ACT/FPN
  - No temporal downsampling
  - Batch-parallel execution
  - EMA within current window only
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.tursnet import TransportBuilder, TransportEncoder, InceptionBlock


# ============================================================
# Differentiable Soft Dilation Conv1d
# ============================================================
class SoftDilatedDepthwise(nn.Module):
    """Depthwise temporal convolution with differentiable (non-integer) dilation.

    For each kernel tap k ∈ {0, 1, ..., K-1}, the sampling position is:
        pos = t + (k - K//2) * dilation
    where dilation is a continuous value derived from sigma.

    Sampling is done via linear interpolation between floor/ceil positions.
    This is fully vectorized — no Python loops over batch/channel/time.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # Depthwise weights: [channels, 1, kernel_size]
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size) * 0.1)

    def forward(self, x, dilation):
        """
        x: [B, C, T]  (C == self.channels)
        dilation: scalar tensor (continuous, positive)
        Returns: [B, C, T]  (same temporal dimension)
        """
        B, C, T = x.shape
        K = self.kernel_size
        half_K = K // 2

        # Kernel tap offsets: [-half_K, ..., 0, ..., half_K]
        offsets = torch.arange(K, device=x.device, dtype=x.dtype) - half_K  # [K]
        # Scaled offsets: [K]
        scaled_offsets = offsets * dilation  # continuous

        # Time positions: [T]
        t_positions = torch.arange(T, device=x.device, dtype=x.dtype)  # [T]

        # Sample positions: [T, K]
        sample_pos = t_positions.unsqueeze(1) + scaled_offsets.unsqueeze(0)  # [T, K]

        # Clamp to valid range
        sample_pos = sample_pos.clamp(0, T - 1)

        # Floor and ceil indices
        pos_floor = sample_pos.floor().long()  # [T, K]
        pos_ceil = (pos_floor + 1).clamp(max=T - 1)  # [T, K]

        # Interpolation weight
        w = sample_pos - pos_floor.float()  # [T, K], fractional part

        # Gather values: x is [B, C, T]
        # We need x[:, :, pos_floor] and x[:, :, pos_ceil]
        # pos_floor/pos_ceil: [T, K] → flatten to [T*K]
        flat_floor = pos_floor.reshape(-1)  # [T*K]
        flat_ceil = pos_ceil.reshape(-1)    # [T*K]

        # Gather: [B, C, T*K]
        x_floor = x[:, :, flat_floor]  # [B, C, T*K]
        x_ceil = x[:, :, flat_ceil]    # [B, C, T*K]

        # Reshape to [B, C, T, K]
        x_floor = x_floor.reshape(B, C, T, K)
        x_ceil = x_ceil.reshape(B, C, T, K)

        # Interpolate: [B, C, T, K]
        w_4d = w.unsqueeze(0).unsqueeze(0)  # [1, 1, T, K]
        x_sampled = (1 - w_4d) * x_floor + w_4d * x_ceil  # [B, C, T, K]

        # Apply depthwise weights: self.weight is [C, 1, K]
        # x_sampled: [B, C, T, K], weight: [C, K] (squeeze dim 1)
        wt = self.weight.squeeze(1)  # [C, K]
        # Weighted sum over K: [B, C, T]
        out = (x_sampled * wt.unsqueeze(0).unsqueeze(2)).sum(dim=-1)

        return out


# ============================================================
# Depthwise-Separable Branch
# ============================================================
class DepthwiseSeparableBranch(nn.Module):
    """One multiscale branch: soft-dilated depthwise + pointwise projection.

    Produces z_i ∈ R^(B × T × branch_dim).
    """

    def __init__(self, in_channels, branch_dim=16, kernel_size=3):
        super().__init__()
        self.depthwise = SoftDilatedDepthwise(in_channels, kernel_size)
        self.bn_dw = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, branch_dim, 1, bias=False)
        self.bn_pw = nn.BatchNorm1d(branch_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, dilation):
        """
        x: [B, C, T]
        dilation: scalar tensor
        Returns: [B, branch_dim, T]
        """
        h = self.depthwise(x, dilation)
        h = self.act(self.bn_dw(h))
        h = self.pointwise(h)
        h = self.act(self.bn_pw(h))
        return h


# ============================================================
# Sigma / Rho Parameterization
# ============================================================
class ScaleParams(nn.Module):
    """Dataset-level learnable sigma and rho parameters.

    sigma: positive, via sigma_min + softplus(raw_sigma)
    rho_i: in (0, 1), via sigmoid(raw_rho_i)
    """

    def __init__(self, sigma_init=5.0, rho_init=(0.9, 0.95, 0.98),
                 sigma_min=1.0, sigma_max=50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Initialize raw_sigma so that softplus(raw_sigma) ≈ sigma_init - sigma_min
        init_val = max(sigma_init - sigma_min, 0.1)
        raw_init = math.log(math.exp(init_val) - 1)  # inverse softplus
        self.raw_sigma = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # Initialize raw_rho so that sigmoid(raw_rho) ≈ rho_init
        raw_rhos = []
        for r in rho_init:
            r_clamped = max(min(r, 0.999), 0.001)
            raw_rhos.append(math.log(r_clamped / (1 - r_clamped)))  # inverse sigmoid
        self.raw_rho = nn.Parameter(torch.tensor(raw_rhos, dtype=torch.float32))

    @property
    def sigma(self):
        s = self.sigma_min + F.softplus(self.raw_sigma)
        return s.clamp(max=self.sigma_max)

    @property
    def rho(self):
        return torch.sigmoid(self.raw_rho)  # [3]

    @property
    def dilations(self):
        """Returns [d1, d2, d3] = [sigma, 3*sigma, 9*sigma]."""
        s = self.sigma
        return [s, 3 * s, 9 * s]


# ============================================================
# Within-Window EMA (safe tensor construction)
# ============================================================
def ema_within_window(z, rho):
    """Compute EMA within each window (no cross-window state).

    z: [B, D, T]
    rho: scalar tensor in (0, 1)

    Returns:
        zbar: [B, D, T]  — EMA of z
        v: [B, D, T]     — velocity = zbar(t) - zbar(t-1), with v(:,:,0) = 0

    Uses an exact per-timestep causal recurrence. The recurrence is stored
    in a tensor list rather than assigning through an in-place slice so that
    autograd sees a clean graph for the first training step. This preserves
    the same within-window semantics as the canonical recurrence.
    """
    B, D, T = z.shape

    one_minus_rho = 1.0 - rho
    frames = []
    cur = z[:, :, 0:1]
    frames.append(cur)

    for t in range(1, T):
        cur = rho * cur + one_minus_rho * z[:, :, t:t+1]
        frames.append(cur)

    zbar = torch.cat(frames, dim=2)

    # Velocity: v(t) = zbar(t) - zbar(t-1), v(0) = 0
    v = torch.zeros_like(z)
    v[:, :, 1:] = zbar[:, :, 1:] - zbar[:, :, :-1]

    return zbar, v


# ============================================================
# Classical Warm Start — Autocorrelation Decay
# ============================================================
def compute_sigma0(X_train, max_lag=50):
    """Compute characteristic timescale σ₀ from training data autocorrelation.

    X_train: numpy array [N, T] or [N, 1, T]
    Returns: float σ₀ (the lag at which autocorrelation drops to 1/e)

    Deterministic, training-only, lightweight.
    """
    import numpy as np

    if X_train.ndim == 3:
        X_train = X_train.squeeze(1)

    N, T = X_train.shape
    max_lag = min(max_lag, T // 2)

    # Use mean signal for coarse estimate
    mean_sig = X_train.mean(axis=0)  # [T]
    mean_sig = mean_sig - mean_sig.mean()
    var = np.var(mean_sig)
    if var < 1e-10:
        # Fallback: use individual signals
        acfs = []
        for i in range(min(N, 100)):
            sig = X_train[i] - X_train[i].mean()
            v = np.var(sig)
            if v < 1e-10:
                continue
            acf = np.correlate(sig, sig, mode='full')
            acf = acf[len(acf) // 2:]
            acf = acf / (v * len(sig))
            acfs.append(acf[:max_lag])
        if not acfs:
            return 5.0  # safe fallback
        acf_mean = np.mean(acfs, axis=0)
    else:
        acf = np.correlate(mean_sig, mean_sig, mode='full')
        acf = acf[len(acf) // 2:]
        acf_mean = acf / (var * len(mean_sig))
        acf_mean = acf_mean[:max_lag]

    # Find lag where ACF drops below 1/e ≈ 0.368
    threshold = 1.0 / math.e
    sigma0 = float(max_lag)  # fallback
    for lag in range(1, len(acf_mean)):
        if acf_mean[lag] < threshold:
            # Linear interpolation for precision
            if lag > 0 and acf_mean[lag - 1] > threshold:
                frac = (acf_mean[lag - 1] - threshold) / (acf_mean[lag - 1] - acf_mean[lag] + 1e-10)
                sigma0 = (lag - 1) + frac
            else:
                sigma0 = float(lag)
            break

    # Clamp to reasonable range
    sigma0 = max(1.5, min(sigma0, T / 4.0))
    return float(sigma0)


# ============================================================
# TURS-CS Model
# ============================================================
class TURSCS(nn.Module):
    """Transport–Uncertainty Regime Synergy: Calibrate-then-Specialize.

    Args:
        in_channels: input channels (1 for ECG/CWRU)
        num_classes: number of output classes
        regime_dim: regime branch output dim (default 16)
        branch_dim: per-branch dim (default 16)
        sigma_init: initial sigma (from warm start)
        rho_init: initial rho tuple (ρ₁, ρ₂, ρ₃)
        dropout: classifier dropout
        lambda_I: interaction strength (default 0.1)
    """

    def __init__(self, in_channels=1, num_classes=5, regime_dim=16,
                 branch_dim=16, sigma_init=5.0,
                 rho_init=(0.9, 0.95, 0.98),
                 dropout=0.1, lambda_I=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.regime_dim = regime_dim
        self.branch_dim = branch_dim
        self.lambda_I_val = lambda_I

        base_ch = 32
        D = base_ch * 2  # 64

        # --- Transport Path (reused from TURS-Lite, identical) ---
        self.transport_builder = TransportBuilder()
        self.transport_encoder = TransportEncoder(out_ch=16, final_ch=D)
        self.C_T = D  # 64

        # --- Inception backbone (identical to TURS-Lite) ---
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True),
        )
        self.block1 = InceptionBlock(base_ch, base_ch)
        self.block2 = InceptionBlock(base_ch, base_ch)
        self.block3 = InceptionBlock(base_ch, D)
        self.block4 = InceptionBlock(D, D)

        # --- Scale Parameters ---
        self.scale_params = ScaleParams(
            sigma_init=sigma_init,
            rho_init=rho_init,
            sigma_min=1.0,
            sigma_max=50.0,
        )

        # --- Three Multiscale Regime Branches ---
        # Input to branches: H4 features [B, D, T]
        self.branch1 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)
        self.branch2 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)
        self.branch3 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)

        # --- Beta (scale-trust gate) ---
        # Input: [zbar1 || zbar2 || zbar3 || v1 || v2 || v3] = 6 * branch_dim = 96
        beta_in_dim = 6 * branch_dim  # 96
        self.P_beta = nn.Linear(beta_in_dim, 3)

        # --- Bounded change: c_t = tanh(P_c(vtilde_t)) ---
        self.P_c = nn.Linear(branch_dim, branch_dim)

        # --- Uncertainty: u_t = sigmoid(P_u([z_t || vtilde_t])) ---
        self.P_u = nn.Linear(2 * branch_dim, 1)

        # --- Regime Feature: F_R = P_R([z_t || vtilde_t || u_t]) ---
        # With c_t compact gating
        regime_in_dim = 2 * branch_dim + 1  # 33
        self.P_R = nn.Linear(regime_in_dim, regime_dim * 2)  # 32
        # c_t gating layer: modulates F_R
        self.c_gate = nn.Sequential(
            nn.Linear(branch_dim, regime_dim * 2),
            nn.Sigmoid(),
        )

        # F_R output dimension = regime_dim * 2 = 32
        F_R_dim = regime_dim * 2  # 32

        # --- Transport-Regime Interaction ---
        # I_TR = P_I(F_T ⊙ P_T(F_R))
        self.P_T_proj = nn.Linear(F_R_dim, self.C_T, bias=False)  # F_R_dim → C_T
        self.P_I = nn.Linear(self.C_T, F_R_dim, bias=False)  # C_T → F_R_dim

        # --- Adaptive Fusion ---
        # alpha = sigmoid(P_alpha(GAP([F_T || F_R])))
        self.P_alpha = nn.Sequential(
            nn.Linear(self.C_T + F_R_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        # --- Classifier ---
        # F (fused): F_R_dim = 32
        # GAP(z_t): branch_dim = 16
        # GAP(vtilde_t): branch_dim = 16
        # GAP(u_t): 1
        # Total: 32 + 16 + 16 + 1 = 65
        cls_in = F_R_dim + branch_dim + branch_dim + 1  # 65
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, x, return_aux=False):
        """
        x: [B, 1, T]
        Returns: logits [B, num_classes]
                 aux dict if return_aux=True
        """
        B, _, T = x.shape

        # === 1. Transport Path ===
        T_repr = self.transport_builder(x)   # [B, 3, T]
        T_e = self.transport_encoder(T_repr)  # [B, C_T, T]
        t_pooled = T_e.mean(dim=2)            # [B, C_T]

        # === 2. Inception Backbone ===
        h = self.proj(x)       # [B, 32, T]
        H1 = self.block1(h)    # [B, 32, T]
        H2 = self.block2(H1)   # [B, 32, T]
        H3 = self.block3(H2)   # [B, 64, T]
        H4 = self.block4(H3)   # [B, 64, T]

        # === 3. Scale Parameters ===
        sigma = self.scale_params.sigma
        rho = self.scale_params.rho  # [3]
        d1, d2, d3 = self.scale_params.dilations

        # === 4. Three Multiscale Branches ===
        z1 = self.branch1(H4, d1)  # [B, branch_dim, T]
        z2 = self.branch2(H4, d2)  # [B, branch_dim, T]
        z3 = self.branch3(H4, d3)  # [B, branch_dim, T]

        # === 5. Within-Window EMA ===
        zbar1, v1 = ema_within_window(z1, rho[0])
        zbar2, v2 = ema_within_window(z2, rho[1])
        zbar3, v3 = ema_within_window(z3, rho[2])

        # === 6. Scale-Trust Beta Gate ===
        # Q_t: [B, 96, T] → need per-timestep beta
        Q_t = torch.cat([zbar1, zbar2, zbar3, v1, v2, v3], dim=1)  # [B, 96, T]
        Q_t_perm = Q_t.permute(0, 2, 1)  # [B, T, 96]
        beta = F.softmax(self.P_beta(Q_t_perm), dim=-1)  # [B, T, 3]

        # === 7. Fused Regime z_t ===
        # z_t = Σ_i beta_i(t) * zbar_i(t) + zbar_1(t)  [local residual]
        beta1 = beta[:, :, 0:1]  # [B, T, 1]
        beta2 = beta[:, :, 1:2]
        beta3 = beta[:, :, 2:3]

        # zbar_i are [B, D, T] → permute to [B, T, D]
        zbar1_p = zbar1.permute(0, 2, 1)  # [B, T, branch_dim]
        zbar2_p = zbar2.permute(0, 2, 1)
        zbar3_p = zbar3.permute(0, 2, 1)

        z_t = (beta1 * zbar1_p + beta2 * zbar2_p + beta3 * zbar3_p
               + zbar1_p)  # [B, T, branch_dim]

        # === 8. Fused Velocity vtilde_t ===
        v1_p = v1.permute(0, 2, 1)  # [B, T, branch_dim]
        v2_p = v2.permute(0, 2, 1)
        v3_p = v3.permute(0, 2, 1)
        vtilde_t = beta1 * v1_p + beta2 * v2_p + beta3 * v3_p  # [B, T, branch_dim]

        # === 9. Bounded Change Signal ===
        c_t = torch.tanh(self.P_c(vtilde_t))  # [B, T, branch_dim]

        # === 10. Uncertainty ===
        u_t = torch.sigmoid(self.P_u(
            torch.cat([z_t, vtilde_t], dim=-1)
        ))  # [B, T, 1]

        # === 11. Regime Feature F_R ===
        # F_R = P_R([z_t || vtilde_t || u_t]) * (1 + c_gate(c_t))
        F_R_raw = self.P_R(torch.cat([z_t, vtilde_t, u_t], dim=-1))  # [B, T, 32]
        c_gate_val = self.c_gate(c_t)  # [B, T, 32]
        F_R = F_R_raw * (1.0 + c_gate_val)  # [B, T, 32]

        # Global average pool over time
        F_R_gap = F_R.mean(dim=1)  # [B, 32]

        # === 12. Transport-Regime Interaction ===
        # I_TR = P_I(F_T ⊙ P_T(F_R))
        F_T_gap = t_pooled  # [B, C_T]
        P_T_FR = self.P_T_proj(F_R_gap)  # [B, C_T]
        I_TR = self.P_I(F_T_gap * P_T_FR)  # [B, F_R_dim]

        # === 13. Adaptive Fusion ===
        alpha = torch.sigmoid(self.P_alpha(
            torch.cat([F_T_gap, F_R_gap], dim=1)
        ))  # [B, 1]

        # F_T needs projection to F_R_dim for fusion
        # Use P_I as implicit projection (F_T_gap is C_T=64, F_R_gap is 32)
        # Instead: project both to same dim
        # F_T contribution: we use the transport-projected regime
        # F = alpha * F_T_contribution + (1-alpha) * F_R + lambda * I_TR
        # For simplicity and to match spec: F_T projected to F_R_dim
        F_T_proj = self.P_I(F_T_gap)  # [B, F_R_dim]
        F_fused = (alpha * F_T_proj
                   + (1 - alpha) * F_R_gap
                   + self.lambda_I_val * I_TR)  # [B, 32]

        # === 14. Classifier ===
        # 65-D: GAP(F)→32, GAP(z_t)→16, GAP(vtilde_t)→16, GAP(u_t)→1
        z_t_gap = z_t.mean(dim=1)          # [B, branch_dim=16]
        vtilde_gap = vtilde_t.mean(dim=1)  # [B, branch_dim=16]
        u_t_gap = u_t.mean(dim=1)          # [B, 1]

        G = torch.cat([F_fused, z_t_gap, vtilde_gap, u_t_gap], dim=1)  # [B, 65]
        logits = self.classifier(G)

        if return_aux:
            aux = {
                "sigma": sigma.detach(),
                "rho": rho.detach(),
                "alpha": alpha.detach(),
                "beta": beta.detach(),
                "uncertainty": u_t.detach(),
                "z_t": z_t.detach(),
                "vtilde": vtilde_t.detach(),
                "c_t": c_t.detach(),
                "F_R": F_R_gap.detach(),
                "I_TR": I_TR.detach(),
                "F_fused": F_fused.detach(),
            }
            return logits, aux
        return logits

    def get_scale_params(self):
        """Return current scale parameters for logging."""
        return {
            "sigma": self.scale_params.sigma.item(),
            "rho1": self.scale_params.rho[0].item(),
            "rho2": self.scale_params.rho[1].item(),
            "rho3": self.scale_params.rho[2].item(),
        }

    def get_calibration_params(self):
        """Return parameters for Phase 0 calibration (high LR)."""
        return [self.scale_params.raw_sigma, self.scale_params.raw_rho]

    def get_specialization_params(self):
        """Return parameters for Phase 1 specialization (normal LR)."""
        calib_ids = {id(p) for p in self.get_calibration_params()}
        return [p for p in self.parameters() if id(p) not in calib_ids]


# ============================================================
# TURS-CS Loss
# ============================================================
class TURSCSLoss(nn.Module):
    """Loss for TURS-CS with gate variance regularization.

    L = L_task - lambda_beta * Var(beta) - lambda_alpha * Var(alpha)

    The negative variance terms ENCOURAGE gate variation when minimized.
    """

    def __init__(self, num_classes=5, lambda_beta=1e-3, lambda_alpha=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_beta = lambda_beta
        self.lambda_alpha = lambda_alpha

    def forward(self, logits, targets, aux=None):
        """Returns (total_loss, loss_dict)."""
        task_loss = F.cross_entropy(logits, targets)
        losses = {"task": task_loss.item()}
        total = task_loss

        if aux is not None:
            # Beta variance regularization (encourage scale diversity)
            beta = aux.get("beta")
            if beta is not None and beta.numel() > 0:
                # beta: [B, T, 3]
                beta_var = beta.var()
                beta_reg = -self.lambda_beta * beta_var
                losses["beta_reg"] = beta_reg.item()
                losses["beta_var"] = beta_var.item()
                total = total + beta_reg

            # Alpha variance regularization (encourage fusion diversity)
            alpha = aux.get("alpha")
            if alpha is not None and alpha.numel() > 0:
                # alpha: [B, 1]
                alpha_var = alpha.var()
                alpha_reg = -self.lambda_alpha * alpha_var
                losses["alpha_reg"] = alpha_reg.item()
                losses["alpha_var"] = alpha_var.item()
                total = total + alpha_reg

        losses["total"] = total.item()
        return total, losses


# ============================================================
# Module parameter counting
# ============================================================
def count_turs_cs_params(model):
    """Detailed parameter count by module."""
    counts = {}

    def _count(name, module):
        n = sum(p.numel() for p in module.parameters())
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts[name] = {"total": n, "trainable": n_train}

    _count("Transport (builder)", model.transport_builder)
    _count("Transport (encoder)", model.transport_encoder)
    _count("Backbone (proj)", model.proj)
    _count("Backbone (block1)", model.block1)
    _count("Backbone (block2)", model.block2)
    _count("Backbone (block3)", model.block3)
    _count("Backbone (block4)", model.block4)
    _count("Scale params (sigma/rho)", model.scale_params)
    _count("Branch 1", model.branch1)
    _count("Branch 2", model.branch2)
    _count("Branch 3", model.branch3)
    _count("P_beta", nn.ModuleList([model.P_beta]))
    _count("P_c", nn.ModuleList([model.P_c]))
    _count("P_u", nn.ModuleList([model.P_u]))
    _count("P_R", nn.ModuleList([model.P_R]))
    _count("c_gate", model.c_gate)
    _count("Interaction (P_T_proj)", nn.ModuleList([model.P_T_proj]))
    _count("Interaction (P_I)", nn.ModuleList([model.P_I]))
    _count("P_alpha", model.P_alpha)
    _count("Classifier", model.classifier)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts["TOTAL"] = {"total": total, "trainable": trainable}
    return counts


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TURS-CS Shape Verification")
    print("=" * 60)

    for T_len, nc in [(140, 5), (1024, 4)]:
        model = TURSCS(in_channels=1, num_classes=nc, sigma_init=5.0)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  T={T_len} classes={nc} | Trainable params: {n:,}")

        x = torch.randn(4, 1, T_len)
        logits, aux = model(x, return_aux=True)

        assert logits.shape == (4, nc), f"Expected ({4}, {nc}), got {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN in logits"
        print(f"    logits: {logits.shape}")

        for k, v in aux.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k:20s}: {list(v.shape)}")

        # Gradient check
        logits.sum().backward()
        no_grad = [name for name, p in model.named_parameters()
                   if p.grad is None and p.requires_grad]
        if no_grad:
            print(f"    WARNING: no gradient for {no_grad}")
        else:
            print(f"    Gradient check: OK")

        # Parameter breakdown
        counts = count_turs_cs_params(model)
        print(f"\n    Parameter breakdown:")
        for name, c in counts.items():
            if name != "TOTAL":
                print(f"      {name:35s}: {c['total']:>7,}")
        print(f"      {'─' * 45}")
        print(f"      {'TOTAL':35s}: {counts['TOTAL']['total']:>7,} "
              f"(trainable: {counts['TOTAL']['trainable']:>7,})")

    print("\nAll checks passed.")
