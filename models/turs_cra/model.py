"""
TURS-CRA — Calibrated Regime–Response Adaptation
=================================================
Canonical compact model scaffold for the biomedical benchmark.

This file implements a minimal, repository-consistent TURS-CRA architecture
that preserves the TURS-CS-style transport and regime core while introducing
one independent lightweight response path and an agreement/response-aware
fusion path.

It is intentionally compact and avoids recurrent state, FPN, attention,
large Rocket banks, or cross-window memory.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.tursnet import TransportBuilder, TransportEncoder, InceptionBlock


class SoftDilatedDepthwise(nn.Module):
    """Vectorized differentiable soft dilation depthwise temporal conv."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size) * 0.1)

    def forward(self, x, dilation):
        B, C, T = x.shape
        K = self.kernel_size
        half_K = K // 2
        offsets = torch.arange(K, device=x.device, dtype=x.dtype) - half_K
        scaled_offsets = offsets * dilation
        t_positions = torch.arange(T, device=x.device, dtype=x.dtype)
        sample_pos = t_positions.unsqueeze(1) + scaled_offsets.unsqueeze(0)
        sample_pos = sample_pos.clamp(0, T - 1)

        pos_floor = sample_pos.floor().long()
        pos_ceil = (pos_floor + 1).clamp(max=T - 1)
        w = sample_pos - pos_floor.float()

        flat_floor = pos_floor.reshape(-1)
        flat_ceil = pos_ceil.reshape(-1)
        x_floor = x[:, :, flat_floor].reshape(B, C, T, K)
        x_ceil = x[:, :, flat_ceil].reshape(B, C, T, K)
        w_4d = w.unsqueeze(0).unsqueeze(0)
        x_sampled = (1 - w_4d) * x_floor + w_4d * x_ceil
        wt = self.weight.squeeze(1)
        out = (x_sampled * wt.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        return out


class DepthwiseSeparableBranch(nn.Module):
    """Compact branch with depthwise soft dilation + pointwise projection."""

    def __init__(self, in_channels, branch_dim=16, kernel_size=3):
        super().__init__()
        self.depthwise = SoftDilatedDepthwise(in_channels, kernel_size)
        self.bn_dw = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, branch_dim, 1, bias=False)
        self.bn_pw = nn.BatchNorm1d(branch_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, dilation):
        h = self.depthwise(x, dilation)
        h = self.act(self.bn_dw(h))
        h = self.pointwise(h)
        h = self.act(self.bn_pw(h))
        return h


class ScaleParams(nn.Module):
    """Dataset-level sigma/rho parameterization with stable positivity."""

    def __init__(self, sigma_init=5.0, rho_init=(0.9, 0.95, 0.98), sigma_min=1.0, sigma_max=50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        init_val = max(sigma_init - sigma_min, 0.1)
        raw_init = math.log(math.exp(init_val) - 1)
        self.raw_sigma = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        raw_rhos = []
        for r in rho_init:
            r_clamped = max(min(r, 0.999), 0.001)
            raw_rhos.append(math.log(r_clamped / (1 - r_clamped)))
        self.raw_rho = nn.Parameter(torch.tensor(raw_rhos, dtype=torch.float32))

        # gamma uses a small raw parameter for bounded gamma
        self.raw_gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def sigma(self):
        s = self.sigma_min + F.softplus(self.raw_sigma)
        return s.clamp(max=self.sigma_max)

    @property
    def rho(self):
        return torch.sigmoid(self.raw_rho)

    @property
    def gamma(self):
        # bounded response influence, 0–1 by default
        return torch.sigmoid(self.raw_gamma)

    @property
    def dilations(self):
        s = self.sigma
        return [s, 3 * s, 9 * s]


def ema_within_window(z, rho):
    """Causal EMA recurrence within the current window only.
    z: [B,D,T]
    rho: scalar in (0,1)
    returns zbar and velocity v.
    """
    B, D, T = z.shape
    one_minus_rho = 1.0 - rho
    frames = []
    cur = z[:, :, 0:1]
    frames.append(cur)
    for t in range(1, T):
        cur = rho * cur + one_minus_rho * z[:, :, t:t + 1]
        frames.append(cur)
    zbar = torch.cat(frames, dim=2)
    v = torch.zeros_like(z)
    v[:, :, 1:] = zbar[:, :, 1:] - zbar[:, :, :-1]
    return zbar, v


class ResponseBranch(nn.Module):
    """Compact response branch. Produces a [B, 16, T] response feature map."""

    def __init__(self, in_channels, out_dim=16, kernel_size=5):
        super().__init__()
        self.depth = SoftDilatedDepthwise(in_channels, kernel_size)
        self.bn = nn.BatchNorm1d(in_channels)
        self.proj = nn.Conv1d(in_channels, out_dim, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, dilation):
        h = self.depth(x, dilation)
        h = self.act(self.bn(h))
        h = self.proj(h)
        return h


class TURSCRA(nn.Module):
    """TURS-CRA: Calibrated Regime–Response Adaptation.

    Compact architecture: transport evidence + shared temporal stem +
    three soft-dilated regime branches + lightweight response branches +
    agreement gate + adaptive fusion + classifier.
    """

    def __init__(self, in_channels=1, num_classes=5, regime_dim=16,
                 branch_dim=16, sigma_init=5.0,
                 rho_init=(0.9, 0.95, 0.98), dropout=0.1, lambda_I=0.1,
                 response_dim=8):
        super().__init__()
        self.num_classes = num_classes
        self.regime_dim = regime_dim
        self.branch_dim = branch_dim
        self.response_dim = response_dim
        self.lambda_I_val = lambda_I

        base_ch = 32
        D = base_ch * 2

        # Transport path: same as TURS-Lite
        self.transport_builder = TransportBuilder()
        self.transport_encoder = TransportEncoder(out_ch=16, final_ch=D)
        self.C_T = D

        # Shared stem, compact local/contextual temporal extractor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            InceptionBlock(base_ch, base_ch),
            InceptionBlock(base_ch, D),
        )

        # Calibration/init parameterization
        self.scale_params = ScaleParams(sigma_init=sigma_init, rho_init=rho_init)

        # Regime branches
        self.branch1 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)
        self.branch2 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)
        self.branch3 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)

        # Response branches
        # Lightweight response dictionary operating on stem representation
        self.response_branch1 = ResponseBranch(D, response_dim, kernel_size=5)
        self.response_branch2 = ResponseBranch(D, response_dim, kernel_size=9)
        self.response_branch3 = ResponseBranch(D, response_dim, kernel_size=17)

        # Response descriptor projection
        # Concatenate response features and descriptors [R1,R2,R3,A1,A2,A3,D1,D2,D3,E1,E2,E3]
        # Each branch contributes R_j, A_j, D_j, E_j → 4 descriptors × 3 branches = 12 maps
        response_in = response_dim * 3 * 4  # 12 maps, e.g. 8*12=96
        self.response_proj = nn.Sequential(
            nn.Conv1d(response_in, response_dim, 1, bias=False),
            nn.BatchNorm1d(response_dim),
            nn.ReLU(inplace=True),
        )

        # Beta gate: input is [zbar1,zbar2,zbar3,v1,v2,v3,R_repr] = 6*branch_dim + response_dim
        beta_in_dim = 6 * branch_dim + response_dim
        self.P_beta = nn.Linear(beta_in_dim, 3)

        # Agreement gate on [z_t || v_t || R_t]
        self.P_a = nn.Linear(2 * branch_dim + response_dim, 1)

        # c_t bounded change
        self.P_c = nn.Linear(branch_dim, branch_dim)

        # Uncertainty
        self.P_u = nn.Linear(2 * branch_dim + response_dim + 1, 1)

        # F_R projection and c-gate
        regime_in_dim = 2 * branch_dim + response_dim + 1 + 1
        self.P_R = nn.Linear(regime_in_dim, regime_dim * 2)
        self.c_gate = nn.Sequential(nn.Linear(branch_dim, regime_dim * 2), nn.Sigmoid())

        # Response gate and gamma control
        self.P_g = nn.Sequential(nn.Linear(2 * branch_dim, 1), nn.Sigmoid())

        # Transport-regime and response-regime interaction
        F_R_dim = regime_dim * 2
        self.P_T_proj = nn.Linear(F_R_dim, self.C_T, bias=False)
        self.P_I = nn.Linear(self.C_T, F_R_dim, bias=False)
        # I_R = P_RI(F_R ⊙ P_RR(R_controlled))
        self.P_RR = nn.Linear(response_dim, F_R_dim, bias=False)  # response→F_R space
        self.P_RI = nn.Linear(F_R_dim, F_R_dim, bias=False)  # interaction projection

        # Adaptive fusion
        self.P_alpha = nn.Sequential(
            nn.Linear(self.C_T + F_R_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        # Classifier consumes 32 + 16 + 16 + 8 + 1 = 73 dims
        cls_in = F_R_dim + branch_dim + branch_dim + response_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, x, return_aux=False):
        """Forward path of compact TURS-CRA.
        x: [B,1,T]
        """
        B, _, T = x.shape

        # Transport path
        T_repr = self.transport_builder(x)
        T_e = self.transport_encoder(T_repr)
        t_pooled = T_e.mean(dim=2)

        # Shared stem
        H = self.stem(x)

        # Scale params
        sigma = self.scale_params.sigma
        rho = self.scale_params.rho
        d1, d2, d3 = self.scale_params.dilations
        gamma = self.scale_params.gamma

        # Regime branches
        z1 = self.branch1(H, d1)
        z2 = self.branch2(H, d2)
        z3 = self.branch3(H, d3)

        zbar1, v1 = ema_within_window(z1, rho[0])
        zbar2, v2 = ema_within_window(z2, rho[1])
        zbar3, v3 = ema_within_window(z3, rho[2])

        # Response branches on same H
        R1 = self.response_branch1(H, d1)
        R2 = self.response_branch2(H, d2)
        R3 = self.response_branch3(H, d3)

        # Derive descriptors from response branches
        # R_j(t), A_j(t)=abs(R_j), D_j(t)=R_j(t)-R_j(t-1), E_j=R_j^2
        A1 = torch.abs(R1)
        A2 = torch.abs(R2)
        A3 = torch.abs(R3)
        D1 = R1 - torch.cat([R1[:, :, 0:1], R1[:, :, :-1]], dim=2)
        D2 = R2 - torch.cat([R2[:, :, 0:1], R2[:, :, :-1]], dim=2)
        D3 = R3 - torch.cat([R3[:, :, 0:1], R3[:, :, :-1]], dim=2)
        E1 = R1 * R1
        E2 = R2 * R2
        E3 = R3 * R3

        # Compact response descriptor stack: [R1,R2,R3,A1,A2,A3,D1,D2,D3,E1,E2,E3]
        # Equivalent response descriptor tensor [B, 33, T]
        resp_stack = torch.cat([R1, R2, R3, A1, A2, A3, D1, D2, D3, E1, E2, E3], dim=1)

        # Project into compact response representation [B, response_dim, T]
        R_repr = self.response_proj(resp_stack)

        # Response energy/shape descriptor for response-aware fusion
        R_mean = R_repr.mean(dim=2)

        # Beta gate conditioned on state+velocity+response descriptor
        Q_beta = torch.cat([zbar1, zbar2, zbar3, v1, v2, v3, R_repr], dim=1)
        Q_beta = Q_beta.permute(0, 2, 1)
        beta = F.softmax(self.P_beta(Q_beta), dim=-1)

        # Fused regime features
        beta1 = beta[:, :, 0:1]
        beta2 = beta[:, :, 1:2]
        beta3 = beta[:, :, 2:3]

        zbar1_p = zbar1.permute(0, 2, 1)
        zbar2_p = zbar2.permute(0, 2, 1)
        zbar3_p = zbar3.permute(0, 2, 1)

        v1_p = v1.permute(0, 2, 1)
        v2_p = v2.permute(0, 2, 1)
        v3_p = v3.permute(0, 2, 1)

        z_t = (beta1 * zbar1_p + beta2 * zbar2_p + beta3 * zbar3_p + zbar1_p)
        vtilde_t = beta1 * v1_p + beta2 * v2_p + beta3 * v3_p

        # agreement gate
        # a_t = sigmoid(P_a([z_t||vtilde_t||R_t]))
        R_repr_p = R_repr.permute(0, 2, 1)
        a_t = torch.sigmoid(self.P_a(torch.cat([z_t, vtilde_t, R_repr_p], dim=-1)))

        # bounded change signal
        c_t = torch.tanh(self.P_c(vtilde_t))

        # uncertainty uses regime, dynamics, response, agreement
        u_t = torch.sigmoid(self.P_u(torch.cat([z_t, vtilde_t, R_repr_p, a_t], dim=-1)))

        # response gate
        g_R = self.P_g(torch.cat([z_t, vtilde_t], dim=-1))
        R_controlled = gamma * g_R * R_repr_p

        # fused regime response memory
        F_R_raw = self.P_R(torch.cat([z_t, vtilde_t, R_repr_p, a_t, u_t], dim=-1))
        c_gate_val = self.c_gate(c_t)
        F_R = F_R_raw * (1.0 + c_gate_val)
        F_R_gap = F_R.mean(dim=1)

        # transport-regime interaction
        F_T_gap = t_pooled
        P_T_FR = self.P_T_proj(F_R_gap)
        I_TR = self.P_I(F_T_gap * P_T_FR)

        # response-regime interaction: I_R = P_RI(F_R ⊙ P_RR(R_controlled))
        R_ctrl_mean = R_controlled.mean(dim=1)  # [B, response_dim]
        R_ctrl_proj = self.P_RR(R_ctrl_mean)    # [B, F_R_dim]
        I_R = self.P_RI(F_R_gap * R_ctrl_proj)  # [B, F_R_dim]

        # adaptive fusion
        alpha = torch.sigmoid(self.P_alpha(torch.cat([F_T_gap, F_R_gap], dim=1)))
        F_T_proj = self.P_I(F_T_gap)
        F_fused = alpha * F_T_proj + (1 - alpha) * F_R_gap + self.lambda_I_val * I_TR + 0.05 * I_R

        # classifier
        z_t_gap = z_t.mean(dim=1)
        vtilde_gap = vtilde_t.mean(dim=1)
        R_gap = R_controlled.mean(dim=1)
        u_t_gap = u_t.mean(dim=1)

        # Final 73-dim vector
        G = torch.cat([F_fused, z_t_gap, vtilde_gap, R_gap, u_t_gap], dim=1)
        logits = self.classifier(G)

        if return_aux:
            aux = {
                'sigma': sigma.detach(),
                'rho': rho.detach(),
                'gamma': gamma.detach(),
                'alpha': alpha.detach(),
                'beta': beta.detach(),
                'agreement': a_t.detach(),
                'uncertainty': u_t.detach(),
                'z_t': z_t.detach(),
                'vtilde': vtilde_t.detach(),
                'c_t': c_t.detach(),
                'response': R_controlled.detach(),
                'F_R': F_R_gap.detach(),
                'I_TR': I_TR.detach(),
                'I_R': I_R.detach(),
            }
            return logits, aux

        return logits


# --- Lightweight utilities ---

def count_turs_cra_params(model):
    """Return total number of trainable and total params by named module."""
    out = {}

    def count_module(name, module):
        out[name] = {
            'params_total': sum(p.numel() for p in module.parameters()),
            'params_trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
        }

    for name, module in [
        ('transport_builder', model.transport_builder),
        ('transport_encoder', model.transport_encoder),
        ('stem', model.stem),
        ('regime_branch1', model.branch1),
        ('regime_branch2', model.branch2),
        ('regime_branch3', model.branch3),
        ('response_branch1', model.response_branch1),
        ('response_branch2', model.response_branch2),
        ('response_branch3', model.response_branch3),
        ('response_proj', model.response_proj),
        ('beta_gate', model.P_beta),
        ('agreement_gate', model.P_a),
        ('change_gate', model.P_c),
        ('uncertainty_gate', model.P_u),
        ('regime_fuser', model.P_R),
        ('response_gate', model.P_g),
        ('transport_regime_interaction', model.P_I),
        ('fusion_alpha', model.P_alpha),
        ('classifier', model.classifier),
    ]:
        count_module(name, module)

    return out


__all__ = ['TURSCRA', 'SoftDilatedDepthwise', 'DepthwiseSeparableBranch', 'ScaleParams', 'ema_within_window', 'count_turs_cra_params']
