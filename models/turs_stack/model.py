"""
TURS-Stack — Shared-trunk, multi-regime-head stacking architecture
==================================================================

Canonical implementation for the biomedical benchmark (ECG5000, CWRU only).

Shared trunk (computed EXACTLY ONCE per forward):
  X_raw -> TransportBuilder -> TransportEncoder -> F_T   (TURS-Lite transport path)
  X_raw -> CompactBackbone (4 InceptionBlocks)   -> H    (TURS-Lite regime-path backbone)

Four structurally distinct regime heads consume the SAME (F_T, H):
  Branch 1 (Lite): regime z/v from H, uncertainty, adaptive T-R fusion
  Branch 2 (RV):   Lite regime + multi-scale DW response correction ADDED to velocity
  Branch 3 (CS):   calibrated soft-dilated multiscale EMA + beta scale-trust
  Branch 4 (CMR):  CS-style calibrated regime + independent bounded response
                   + response-aware scale trust + agreement gating

Combination is performed on class-probability vectors only:
  soft vote | hard vote | static learned weights | linear stacking |
  diagnostic-conditioned stacking (novelty scalar e_t appended)

Invariants (per spec):
  - no cross-window state (EMA recurrence is within-window only)
  - no diversity-forcing losses (branches differ structurally)
  - transport encoder / backbone are called exactly once per forward
  - per-branch fusion/classifier heads are independent
  - calibration (sigma, rho) is a learnable dataset-level scalar set with
    TRAINING data only (warm-started by the runner before training)

NOTE: F_T per timestep is 64-dim (C_T). Branch 1/2 use it at full temporal
resolution; Branches 3/4 consume its GAP summary (consistent with TURS-CS /
TURS-CRM which fuse on the pooled transport feature).
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.tursnet import TransportBuilder, TransportEncoder, InceptionBlock
from models.turs_cs.model import (
    SoftDilatedDepthwise,
    DepthwiseSeparableBranch,
    ScaleParams,
    ema_within_window,
    compute_sigma0,
)


def ema_multi(z, rho):
    """Grouped within-window causal EMA for the 3 scales of one branch.

    z: [B, 3*bd, T] (scale1 || scale2 || scale3)
    rho: [3] tensor of per-scale decay values
    Returns: zbar [B, 3*bd, T], v [B, 3*bd, T] with v(0) = 0.
    Equivalent to running ema_within_window 3 times, but with ONE Python
    recurrence loop over T instead of three (3x fewer kernel launches).
    """
    B, D3, T = z.shape
    bd = D3 // 3
    rho_full = rho.repeat_interleave(bd)          # [D3]
    one_minus = 1.0 - rho_full
    frames = []
    cur = z[:, :, 0:1]                            # zbar_i(0) = z_i(0)
    frames.append(cur)
    for t in range(1, T):
        cur = rho_full[None, :, None] * cur + one_minus[None, :, None] * z[:, :, t:t + 1]
        frames.append(cur)
    zbar = torch.cat(frames, dim=2)
    v = torch.zeros_like(z)
    v[:, :, 1:] = zbar[:, :, 1:] - zbar[:, :, :-1]
    return zbar, v


# ============================================================
# Shared compact temporal backbone (TURS-Lite regime path)
# ============================================================
class SharedCompactBackbone(nn.Module):
    """proj + 4 InceptionBlocks, exactly the TURS-Lite backbone."""

    def __init__(self, in_channels=1, base_ch=32, D=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.block1 = InceptionBlock(base_ch, base_ch)
        self.block2 = InceptionBlock(base_ch, base_ch)
        self.block3 = InceptionBlock(base_ch, D)
        self.block4 = InceptionBlock(D, D)
        self.out_ch = D

    def forward(self, x):
        h = self.proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        return h


# ============================================================
# Branch 1 — TURS-Lite style
# ============================================================
class LiteBranch(nn.Module):
    """Regime z/v from H, uncertainty, adaptive transport-regime fusion."""

    def __init__(self, C_H, C_T, num_classes, regime_dim=16, lambda_I=0.1, dropout=0.1):
        super().__init__()
        self.lambda_I = lambda_I
        rd = regime_dim

        # z_t = P_z(H_t) ; per-timestep regime state
        self.P_z = nn.Conv1d(C_H, rd, 1)
        self.vel_linear = nn.Linear(rd, rd)  # residual-style velocity embedding
        # u_t = sigmoid(P_u([z_t || v_t]))
        self.P_u = nn.Linear(2 * rd, 1)

        # F_R = P_R([z_t || v_t || u_t])
        self.P_R = nn.Linear(2 * rd + 1, 2 * rd)  # -> 32 channels

        F_R_dim = 2 * rd
        # I_TR = P_I(F_T ⊙ P_T(F_R))
        self.P_T = nn.Linear(F_R_dim, C_T, bias=False)
        self.P_I = nn.Linear(C_T, F_R_dim, bias=False)
        # alpha = sigmoid(P_alpha(GAP([F_T || F_R])))
        self.P_alpha = nn.Sequential(
            nn.Linear(C_T + F_R_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1)
        )

        cls_in = F_R_dim + rd + rd + 1  # GAP(F), GAP(z), GAP(v), GAP(u)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 48), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, F_T, H):
        B, C, T = H.shape
        F_T_t = F_T.permute(0, 2, 1)             # [B, T, C_T]
        F_T_gap = F_T_t.mean(dim=1)              # [B, C_T]

        # --- regime state / velocity (Lite formulation) ---
        z_t = self.P_z(H).permute(0, 2, 1)       # [B, T, rd]
        v_t = self.vel_linear(z_t)               # [B, T, rd] (Lite residual velocity)
        u_t = torch.sigmoid(self.P_u(torch.cat([z_t, v_t], dim=-1)))  # [B, T, 1]

        F_R = self.P_R(torch.cat([z_t, v_t, u_t], dim=-1))     # [B, T, 32]
        F_R_gap = F_R.mean(dim=1)                              # [B, 32]

        # --- transport-regime interaction ---
        I_TR = self.P_I(F_T_gap * self.P_T(F_R_gap))           # [B, 32]
        alpha = torch.sigmoid(self.P_alpha(torch.cat([F_T_gap, F_R_gap], dim=1)))
        F_T_proj = self.P_I(F_T_gap)
        F_fused = alpha * F_T_proj + (1 - alpha) * F_R_gap + self.lambda_I * I_TR

        feats = torch.cat([F_fused, z_t.mean(1), v_t.mean(1), u_t.mean(1)], dim=1)
        logits = self.classifier(feats)
        return logits


# ============================================================
# Branch 2 — TURS-RV style
# ============================================================
class RVBranch(nn.Module):
    """Lite regime + small DW multi-scale response correction ADDED to velocity."""

    def __init__(self, C_H, C_T, num_classes, regime_dim=16, lambda_I=0.1,
                 resp_ch=4, dropout=0.1):
        super().__init__()
        self.lambda_I = lambda_I
        rd = regime_dim
        self.resp_ch = resp_ch

        self.P_z = nn.Conv1d(C_H, rd, 1)
        self.vel_linear = nn.Linear(rd, rd)

        # Small response branch: MultiDWConv 7/15/31, resp_ch each -> 12 channels
        self.dw7 = nn.Conv1d(C_H, resp_ch, 7, padding=3, groups=1, bias=False)
        self.dw15 = nn.Conv1d(C_H, resp_ch, 15, padding=7, groups=1, bias=False)
        self.dw31 = nn.Conv1d(C_H, resp_ch, 31, padding=15, groups=1, bias=False)
        resp_total = 3 * resp_ch

        # g_t = sigmoid(P_g([z_t || v_t || R_t])) ; R_t pooled to vector
        self.P_g = nn.Linear(2 * rd + resp_total, 1)
        # correction = g ⊙ P_Delta(DeltaR_t)
        self.P_Delta = nn.Linear(resp_total, rd)

        # F_R = P_R([z_t || vtilde_t || u_t])
        self.P_u = nn.Linear(2 * rd, 1)
        self.P_R = nn.Linear(2 * rd + 1, 2 * rd)
        F_R_dim = 2 * rd
        self.P_T = nn.Linear(F_R_dim, C_T, bias=False)
        self.P_I = nn.Linear(C_T, F_R_dim, bias=False)
        self.P_alpha = nn.Sequential(
            nn.Linear(C_T + F_R_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1)
        )

        cls_in = F_R_dim + rd + rd + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 48), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, F_T, H):
        B, C, T = H.shape
        F_T_t = F_T.permute(0, 2, 1)
        F_T_gap = F_T_t.mean(dim=1)

        z_t = self.P_z(H).permute(0, 2, 1)
        v_t = self.vel_linear(z_t)

        # --- response branch (compact, 12 channels total) ---
        R7 = self.dw7(H).permute(0, 2, 1)    # [B, T, rc]
        R15 = self.dw15(H).permute(0, 2, 1)
        R31 = self.dw31(H).permute(0, 2, 1)
        R_t = torch.cat([R7, R15, R31], dim=-1)             # [B, T, 3rc]
        DeltaR = torch.zeros_like(R_t)
        DeltaR[:, 1:, :] = R_t[:, 1:, :] - R_t[:, :-1, :]   # DeltaR_0 = 0

        g_t = torch.sigmoid(self.P_g(torch.cat([z_t, v_t, R_t], dim=-1)))  # [B, T, 1]
        correction = g_t * self.P_Delta(DeltaR)             # [B, T, rd]
        vtilde = v_t + correction                           # RV-style: ADD to velocity

        u_t = torch.sigmoid(self.P_u(torch.cat([z_t, vtilde], dim=-1)))
        F_R = self.P_R(torch.cat([z_t, vtilde, u_t], dim=-1))
        F_R_gap = F_R.mean(dim=1)

        I_TR = self.P_I(F_T_gap * self.P_T(F_R_gap))
        alpha = torch.sigmoid(self.P_alpha(torch.cat([F_T_gap, F_R_gap], dim=1)))
        F_T_proj = self.P_I(F_T_gap)
        F_fused = alpha * F_T_proj + (1 - alpha) * F_R_gap + self.lambda_I * I_TR

        feats = torch.cat([F_fused, z_t.mean(1), vtilde.mean(1), u_t.mean(1)], dim=1)
        logits = self.classifier(feats)
        return logits


# ============================================================
# Branch 3 — TURS-CS style (calibrated multiscale, no response)
# ============================================================
class CSBranch(nn.Module):
    """Three soft-dilated scales (sigma, 3sigma, 9sigma) + within-window EMA
    + beta scale-trust gate + bounded change + uncertainty."""

    def __init__(self, C_H, C_T, num_classes, regime_dim=16, branch_dim=16,
                 sigma_init=5.0, rho_init=(0.9, 0.95, 0.98), lambda_I=0.1,
                 dropout=0.1):
        super().__init__()
        self.lambda_I = lambda_I
        self.branch_dim = branch_dim
        bd = branch_dim
        rd = regime_dim

        self.scale_params = ScaleParams(sigma_init=sigma_init, rho_init=rho_init)
        self.branch1 = DepthwiseSeparableBranch(C_H, bd, kernel_size=3)
        self.branch2 = DepthwiseSeparableBranch(C_H, bd, kernel_size=3)
        self.branch3 = DepthwiseSeparableBranch(C_H, bd, kernel_size=3)

        self.P_beta = nn.Linear(6 * bd, 3)
        self.P_c = nn.Linear(bd, bd)
        self.P_u = nn.Linear(2 * bd, 1)
        self.P_R = nn.Linear(2 * bd + 1, 2 * rd)   # F_R at 32 channels
        self.c_gate = nn.Sequential(nn.Linear(bd, 2 * rd), nn.Sigmoid())

        F_R_dim = 2 * rd
        self.P_T = nn.Linear(F_R_dim, C_T, bias=False)
        self.P_I = nn.Linear(C_T, F_R_dim, bias=False)
        self.P_alpha = nn.Sequential(
            nn.Linear(C_T + F_R_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1)
        )

        cls_in = F_R_dim + bd + bd + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 48), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, F_T, H):
        F_T_gap = F_T.mean(dim=2)                  # [B, C_T]
        sigma = self.scale_params.sigma
        rho = self.scale_params.rho                # [3]
        d1, d2, d3 = sigma, 3.0 * sigma, 9.0 * sigma

        z1 = self.branch1(H, d1)                   # [B, bd, T]
        z2 = self.branch2(H, d2)
        z3 = self.branch3(H, d3)

        zbar_g, v_g = ema_multi(torch.cat([z1, z2, z3], dim=1), rho)
        zbar1, zbar2, zbar3 = zbar_g.split(self.branch_dim, dim=1)
        v1, v2, v3 = v_g.split(self.branch_dim, dim=1)

        def p(x):
            return x.permute(0, 2, 1)              # [B, T, bd]

        Q_t = torch.cat([p(zbar1), p(zbar2), p(zbar3), p(v1), p(v2), p(v3)], dim=-1)
        beta = F.softmax(self.P_beta(Q_t), dim=-1)  # [B, T, 3]
        b1, b2, b3 = beta[..., 0:1], beta[..., 1:2], beta[..., 2:3]

        z_t = b1 * p(zbar1) + b2 * p(zbar2) + b3 * p(zbar3) + p(zbar1)
        vtilde = b1 * p(v1) + b2 * p(v2) + b3 * p(v3)

        c_t = torch.tanh(self.P_c(vtilde))
        u_t = torch.sigmoid(self.P_u(torch.cat([z_t, vtilde], dim=-1)))
        F_R = self.P_R(torch.cat([z_t, vtilde, u_t], dim=-1)) * (1.0 + self.c_gate(c_t))
        F_R_gap = F_R.mean(dim=1)

        I_TR = self.P_I(F_T_gap * self.P_T(F_R_gap))
        alpha = torch.sigmoid(self.P_alpha(torch.cat([F_T_gap, F_R_gap], dim=1)))
        F_T_proj = self.P_I(F_T_gap)
        F_fused = alpha * F_T_proj + (1 - alpha) * F_R_gap + self.lambda_I * I_TR

        feats = torch.cat([F_fused, z_t.mean(1), vtilde.mean(1), u_t.mean(1)], dim=1)
        logits = self.classifier(feats)

        # --- novelty diagnostic e_t (deterministic, no labels) ---
        # distance between current fused regime state and its persistent
        # (slow-EMA) expectation:  e_t = tanh(1 - cos(z_t, zbar_1))
        cos = F.cosine_similarity(z_t, p(zbar1), dim=-1)          # [B, T]
        e_t = torch.tanh(1.0 - cos).mean(dim=1)                   # [B]
        return logits, beta, e_t


# ============================================================
# Branch 4 — TURS-CMR style (calibrated regime + bounded response)
# ============================================================
class CMRBranch(nn.Module):
    """CS-style calibrated multiscale regime + compact independent response bank
    with BOUNDED (gamma-gated) response — response never added to velocity."""

    def __init__(self, C_H, C_T, num_classes, regime_dim=16, branch_dim=16,
                 resp_dim=8, sigma_init=5.0, rho_init=(0.9, 0.95, 0.98),
                 lambda_I=0.1, dropout=0.1):
        super().__init__()
        self.lambda_I = lambda_I
        self.branch_dim = branch_dim
        bd = branch_dim
        rd = regime_dim
        self.resp_dim = resp_dim

        self.scale_params = ScaleParams(sigma_init=sigma_init, rho_init=rho_init)
        self.branch1 = DepthwiseSeparableBranch(C_H, bd, kernel_size=3)
        self.branch2 = DepthwiseSeparableBranch(C_H, bd, kernel_size=3)
        self.branch3 = DepthwiseSeparableBranch(C_H, bd, kernel_size=3)

        # Small learned response bank: DWConv 5/9/17 at resp_dim each
        self.resp1 = nn.Conv1d(C_H, resp_dim, 5, padding=2, bias=False)
        self.resp2 = nn.Conv1d(C_H, resp_dim, 9, padding=4, bias=False)
        self.resp3 = nn.Conv1d(C_H, resp_dim, 17, padding=8, bias=False)
        # descriptors: [R1,R2,R3, A1..3, D1..3, E1..3] -> P_R to resp latent
        self.P_Rdesc = nn.Linear(12 * resp_dim, resp_dim)

        # bounded response strength + local gate
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))
        self.P_g = nn.Linear(2 * bd, 1)

        # response-aware scale trust: s_i = P_i([zbar_i || v_i || R_i])
        self.P_s1 = nn.Linear(2 * bd + resp_dim, 1)
        self.P_s2 = nn.Linear(2 * bd + resp_dim, 1)
        self.P_s3 = nn.Linear(2 * bd + resp_dim, 1)

        # agreement, change, uncertainty
        self.P_a = nn.Linear(2 * bd + resp_dim, 1)
        self.P_c = nn.Linear(bd, bd)
        self.P_u = nn.Linear(2 * bd + resp_dim + 1, 1)
        self.P_R2 = nn.Linear(2 * bd + resp_dim + 2, 32)

        F_R_dim = 32
        self.P_T = nn.Linear(F_R_dim, C_T, bias=False)
        self.P_I = nn.Linear(C_T, F_R_dim, bias=False)
        self.P_alpha = nn.Sequential(
            nn.Linear(C_T + F_R_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1)
        )

        cls_in = F_R_dim + bd + bd + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 48), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    @property
    def gamma(self):
        return torch.sigmoid(self.raw_gamma)

    def forward(self, F_T, H):
        F_T_gap = F_T.mean(dim=2)
        sigma = self.scale_params.sigma
        rho = self.scale_params.rho
        d1, d2, d3 = sigma, 3.0 * sigma, 9.0 * sigma

        z1 = self.branch1(H, d1)
        z2 = self.branch2(H, d2)
        z3 = self.branch3(H, d3)
        zbar_g, v_g = ema_multi(torch.cat([z1, z2, z3], dim=1), rho)
        zbar1, zbar2, zbar3 = zbar_g.split(self.branch_dim, dim=1)
        v1, v2, v3 = v_g.split(self.branch_dim, dim=1)

        def p(x):
            return x.permute(0, 2, 1)

        # --- response descriptors ---
        R1 = self.resp1(H).permute(0, 2, 1)
        R2 = self.resp2(H).permute(0, 2, 1)
        R3 = self.resp3(H).permute(0, 2, 1)
        D1 = torch.zeros_like(R1); D1[:, 1:, :] = R1[:, 1:, :] - R1[:, :-1, :]
        D2 = torch.zeros_like(R2); D2[:, 1:, :] = R2[:, 1:, :] - R2[:, :-1, :]
        D3 = torch.zeros_like(R3); D3[:, 1:, :] = R3[:, 1:, :] - R3[:, :-1, :]
        desc = torch.cat([R1, R2, R3,
                          R1.abs(), R2.abs(), R3.abs(),
                          D1, D2, D3,
                          R1 * R1, R2 * R2, R3 * R3], dim=-1)     # [B, T, 12rd]
        R_lat = self.P_Rdesc(desc)                                # [B, T, resp_dim]

        # --- bounded controlled response ---
        z_c = p(zbar1) + p(zbar2) + p(zbar3)     # provisional state for gating
        v_c = p(v1) + p(v2) + p(v3)              # provisional velocity for gating
        g_R = torch.sigmoid(self.P_g(torch.cat([z_c, v_c], dim=-1)))  # [B, T, 1]
        R_controlled = self.gamma * g_R * R_lat  # bounded; NEVER added to velocity

        # --- response-aware scale trust ---
        s1 = self.P_s1(torch.cat([p(zbar1), p(v1), R_lat], dim=-1))
        s2 = self.P_s2(torch.cat([p(zbar2), p(v2), R_lat], dim=-1))
        s3 = self.P_s3(torch.cat([p(zbar3), p(v3), R_lat], dim=-1))
        beta = F.softmax(torch.cat([s1, s2, s3], dim=-1), dim=-1)    # [B, T, 3]
        b1, b2, b3 = beta[..., 0:1], beta[..., 1:2], beta[..., 2:3]

        # --- fused state + velocity (SAME beta for both) ---
        z_t = b1 * p(zbar1) + b2 * p(zbar2) + b3 * p(zbar3) + p(zbar1)
        vtilde = b1 * p(v1) + b2 * p(v2) + b3 * p(v3)

        # --- agreement / change / uncertainty ---
        a_t = torch.sigmoid(self.P_a(torch.cat([z_t, vtilde, R_controlled], dim=-1)))
        c_t = torch.tanh(self.P_c(vtilde))
        u_t = torch.sigmoid(self.P_u(
            torch.cat([z_t, vtilde, R_controlled, a_t], dim=-1)))

        F_R = self.P_R2(torch.cat([z_t, vtilde, R_controlled, a_t, u_t], dim=-1))
        F_R_gap = F_R.mean(dim=1)

        I_TR = self.P_I(F_T_gap * self.P_T(F_R_gap))
        alpha = torch.sigmoid(self.P_alpha(torch.cat([F_T_gap, F_R_gap], dim=1)))
        F_T_proj = self.P_I(F_T_gap)
        F_fused = alpha * F_T_proj + (1 - alpha) * F_R_gap + self.lambda_I * I_TR

        feats = torch.cat([F_fused, z_t.mean(1), vtilde.mean(1), u_t.mean(1)], dim=1)
        logits = self.classifier(feats)

        self._last_diag = {
            "beta": beta.detach(),
            "gamma": self.gamma.detach(),
            "R_norm": R_controlled.detach().norm(dim=-1).mean(),
            "agreement": a_t.detach().mean(),
            "uncertainty": u_t.detach().mean(),
        }
        return logits, beta


# ============================================================
# TURS-Stack full model
# ============================================================
class TURSStack(nn.Module):
    """One canonical shared-trunk model with four structurally distinct heads."""

    def __init__(self, in_channels=1, num_classes=5, sequence_length=140,
                 regime_dim=16, branch_dim=16, sigma_init=5.0,
                 rho_init=(0.9, 0.95, 0.98), lambda_I=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_I = lambda_I

        base_ch = 32
        D = 64
        self.C_T = D
        self.C_H = D

        # ---- SHARED TRUNK (instantiated exactly once) ----
        self.transport_builder = TransportBuilder()
        self.transport_encoder = TransportEncoder(out_ch=16, final_ch=D)
        self.backbone = SharedCompactBackbone(in_channels=in_channels,
                                              base_ch=base_ch, D=D)

        # ---- four branch heads ----
        self.lite = LiteBranch(D, D, num_classes, regime_dim, lambda_I)
        self.rv = RVBranch(D, D, num_classes, regime_dim, lambda_I)
        self.cs = CSBranch(D, D, num_classes, regime_dim, branch_dim,
                           sigma_init, rho_init, lambda_I)
        self.cmr = CMRBranch(D, D, num_classes, regime_dim, branch_dim,
                             resp_dim=8, sigma_init=sigma_init,
                             rho_init=rho_init, lambda_I=lambda_I)

        self.branch_names = ["lite", "rv", "cs", "cmr"]
        self._transport_calls = 0
        self._backbone_calls = 0

    # public counters for the shared-computation test
    def reset_call_counters(self):
        self._transport_calls = 0
        self._backbone_calls = 0

    def forward(self, x, return_aux=False):
        self._transport_calls += 1
        self._backbone_calls += 1

        # ---- shared trunk, computed ONCE ----
        T_repr = self.transport_builder(x)          # [B, 3, T]
        F_T = self.transport_encoder(T_repr)        # [B, C_T, T]
        H = self.backbone(x)                        # [B, C_H, T]

        logits1 = self.lite(F_T, H)
        logits2 = self.rv(F_T, H)
        logits3, beta3, e_t = self.cs(F_T, H)
        logits4, beta4 = self.cmr(F_T, H)

        probs = [F.softmax(l, dim=-1) for l in (logits1, logits2, logits3, logits4)]

        out = {
            "branch_logits": [logits1, logits2, logits3, logits4],
            "probs": probs,
            "F_T": F_T,
            "H": H,
            "beta_cs": beta3,
            "beta_cmr": beta4,
            "novelty": e_t,          # scalar diagnostic e_t per sample
        }
        if return_aux:
            out["aux"] = {
                "gamma": self.cmr.gamma.detach(),
                "sigma_cs": self.cs.scale_params.sigma.detach(),
                "rho_cs": self.cs.scale_params.rho.detach(),
                "sigma_cmr": self.cmr.scale_params.sigma.detach(),
                "rho_cmr": self.cmr.scale_params.rho.detach(),
            }
        return out

    def get_calibration_params(self):
        """raw sigma/rho params for warm start / two-phase training."""
        return ([self.cs.scale_params.raw_sigma, self.cs.scale_params.raw_rho,
                 self.cmr.scale_params.raw_sigma, self.cmr.scale_params.raw_rho])


# ============================================================
# Combination methods (operate on frozen branch probability vectors)
# ============================================================
def soft_vote(probs):
    """probs: list of 4 [B, C] tensors -> [B, C]"""
    return torch.stack(probs, dim=0).mean(dim=0)


def hard_vote(probs):
    """Majority vote with confidence tie-break. No trainable params."""
    P = torch.stack(probs, dim=0)                     # [4, B, C]
    votes = P.argmax(dim=-1)                          # [4, B]
    n_cls = P.shape[-1]
    B = P.shape[1]
    onehot = F.one_hot(votes, n_cls).float()          # [4, B, C]
    vote_counts = onehot.sum(dim=0)                   # [B, C]
    max_votes = vote_counts.max(dim=-1, keepdim=True).values  # [B, 1]
    is_tied = (vote_counts == max_votes)              # [B, C] bool
    conf = P.sum(dim=0)                               # [B, C] summed confidence
    tie_break = torch.where(is_tied, conf, torch.full_like(conf, -1.0))
    return tie_break.argmax(dim=-1)


class StaticWeightCombiner(nn.Module):
    """theta in R^4 -> w = softmax(theta); p_final = sum w_k p_k. 4 params."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(4))

    def forward(self, probs):
        w = torch.softmax(self.theta, dim=0)
        return torch.einsum("k,kbc->bc", w, torch.stack(probs, dim=0)), w


class StackingCombiner(nn.Module):
    """Linear meta-classifier on [p1||p2||p3||p4] (4C -> C)."""

    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(4 * num_classes, num_classes)

    def forward(self, probs):
        P = torch.cat(probs, dim=-1)
        return self.linear(P)


class DiagnosticConditionedCombiner(nn.Module):
    """Meta-classifier on [p1||p2||p3||p4 || e_t] where e_t is a scalar
    novelty/deviation diagnostic derived deterministically from the frozen
    model (no labels). 4C + 1 -> C."""

    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(4 * num_classes + 1, num_classes)

    def forward(self, probs, e_t):
        P = torch.cat(probs + [e_t.unsqueeze(-1)], dim=-1)
        return self.linear(P)
