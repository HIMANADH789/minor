"""
TURS-CRM — Calibrated Regime–Response Multiscale model
=======================================================
Compact canonical scaffold implementing the attached specification.
"""

import math
import os
import sys

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
    """Depthwise soft-dilated branch + pointwise projection."""

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
    """Positive dataset-level scalar calibration values sigma, rho and gamma."""

    def __init__(self, sigma_init=5.0, rho_init=(0.90, 0.95, 0.98), sigma_min=1.0, sigma_max=50.0):
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
        return torch.sigmoid(self.raw_gamma)

    @property
    def dilations(self):
        s = self.sigma
        return [s, 3 * s, 9 * s]


def ema_within_window(z, rho):
    """Causal EMA recurrence within current window only: zbar, velocity v."""
    B, D, T = z.shape
    frames = []
    cur = z[:, :, 0:1]
    frames.append(cur)
    for t in range(1, T):
        cur = rho * cur + (1.0 - rho) * z[:, :, t:t + 1]
        frames.append(cur)
    zbar = torch.cat(frames, dim=2)
    v = torch.zeros_like(z)
    v[:, :, 1:] = zbar[:, :, 1:] - zbar[:, :, :-1]
    return zbar, v


class ResponseBranch(nn.Module):
    """Compact response branch with a differentiable soft-dilated depthwise stem."""

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


class TURSCRM(nn.Module):
    """Compact TURS-CRM model implementing the attached spec in a repository-fitting way."""

    def __init__(self, in_channels=1, num_classes=5, regime_dim=16,
                 branch_dim=16, sigma_init=5.0,
                 rho_init=(0.90, 0.95, 0.98), dropout=0.1,
                 lambda_I=0.1, response_dim=8):
        super().__init__()
        self.num_classes = num_classes
        self.regime_dim = regime_dim
        self.branch_dim = branch_dim
        self.response_dim = response_dim
        self.lambda_I = lambda_I

        base_ch = 32
        D = base_ch * 2

        self.transport_builder = TransportBuilder()
        self.transport_encoder = TransportEncoder(out_ch=16, final_ch=D)
        self.C_T = D

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            InceptionBlock(base_ch, base_ch),
            InceptionBlock(base_ch, D),
        )

        self.scale_params = ScaleParams(sigma_init=sigma_init, rho_init=rho_init)

        self.branch1 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)
        self.branch2 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)
        self.branch3 = DepthwiseSeparableBranch(D, branch_dim, kernel_size=3)

        self.response_branch1 = ResponseBranch(D, response_dim, kernel_size=5)
        self.response_branch2 = ResponseBranch(D, response_dim, kernel_size=9)
        self.response_branch3 = ResponseBranch(D, response_dim, kernel_size=17)

        self.response_proj = nn.Sequential(
            nn.Conv1d(response_dim * 12, response_dim, 1, bias=False),
            nn.BatchNorm1d(response_dim),
            nn.ReLU(inplace=True),
        )

        self.P_beta = nn.Conv1d(6 * branch_dim + 3 * response_dim, 3, 1, bias=True)
        self.P_a = nn.Conv1d(2 * branch_dim + response_dim, 1, 1, bias=True)
        self.P_c = nn.Conv1d(branch_dim, branch_dim, 1, bias=True)
        self.P_u = nn.Conv1d(2 * branch_dim + response_dim + 1, 1, 1, bias=True)
        self.P_R = nn.Conv1d(2 * branch_dim + response_dim + 1 + 1, branch_dim * 2, 1, bias=True)
        self.P_g = nn.Conv1d(2 * branch_dim, 1, 1, bias=True)

        self.P_T = nn.Conv1d(branch_dim * 2, D, 1, bias=False)
        self.P_I = nn.Conv1d(D, branch_dim * 2, 1, bias=False)
        self.P_RR = nn.Conv1d(branch_dim * 2, response_dim, 1, bias=False)
        self.P_RI = nn.Conv1d(response_dim, branch_dim * 2, 1, bias=False)

        self.P_alpha = nn.Sequential(
            nn.Linear(D + branch_dim * 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        cls_in = D + branch_dim + branch_dim + response_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    def forward(self, x, return_aux=False):
        B, _, T = x.shape

        T_repr = self.transport_builder(x)
        F_T = self.transport_encoder(T_repr)
        F_T_gap = F_T.mean(dim=2)

        H = self.stem(x)

        sigma = self.scale_params.sigma
        rho = self.scale_params.rho
        d1, d2, d3 = self.scale_params.dilations
        gamma = self.scale_params.gamma

        z1 = self.branch1(H, d1)
        z2 = self.branch2(H, d2)
        z3 = self.branch3(H, d3)

        zbar1, v1 = ema_within_window(z1, rho[0])
        zbar2, v2 = ema_within_window(z2, rho[1])
        zbar3, v3 = ema_within_window(z3, rho[2])

        R1 = self.response_branch1(H, d1)
        R2 = self.response_branch2(H, d2)
        R3 = self.response_branch3(H, d3)

        A1 = torch.abs(R1)
        A2 = torch.abs(R2)
        A3 = torch.abs(R3)
        D1 = R1 - torch.cat([R1[:, :, 0:1], R1[:, :, :-1]], dim=2)
        D2 = R2 - torch.cat([R2[:, :, 0:1], R2[:, :, :-1]], dim=2)
        D3 = R3 - torch.cat([R3[:, :, 0:1], R3[:, :, :-1]], dim=2)
        E1 = R1 * R1
        E2 = R2 * R2
        E3 = R3 * R3

        resp_stack = torch.cat([R1, R2, R3, A1, A2, A3, D1, D2, D3, E1, E2, E3], dim=1)
        R_repr = self.response_proj(resp_stack)

        beta_in = torch.cat([zbar1, zbar2, zbar3, v1, v2, v3, R_repr], dim=1)
        beta_logits = self.P_beta(beta_in)
        beta = F.softmax(beta_logits, dim=1)

        beta1 = beta[:, 0:1, :]
        beta2 = beta[:, 1:2, :]
        beta3 = beta[:, 2:3, :]

        z_t = beta1 * zbar1 + beta2 * zbar2 + beta3 * zbar3 + zbar1
        v_tilde = beta1 * v1 + beta2 * v2 + beta3 * v3

        c_t = torch.tanh(self.P_c(v_tilde))
        a_t = torch.sigmoid(self.P_a(torch.cat([z_t, v_tilde, R_repr], dim=1)))
        u_t = torch.sigmoid(self.P_u(torch.cat([z_t, v_tilde, R_repr, a_t], dim=1)))

        g_R = torch.sigmoid(self.P_g(torch.cat([z_t, v_tilde], dim=1)))
        R_controlled = gamma * g_R * R_repr

        FR_in = torch.cat([z_t, v_tilde, R_controlled, a_t, u_t], dim=1)
        FR = self.P_R(FR_in)

        I_TR = F_T * self.P_T(FR)
        alpha_in = torch.cat([F_T_gap, FR.mean(dim=2)], dim=1)
        alpha = torch.sigmoid(self.P_alpha(alpha_in)).squeeze(-1)

        F = alpha[:, None, None] * F_T + (1.0 - alpha)[:, None, None] * FR + 0.1 * I_TR

        fF = F.mean(dim=2)
        fz = z_t.mean(dim=2)
        fv = v_tilde.mean(dim=2)
        fR = R_controlled.mean(dim=2)
        fu = u_t.mean(dim=2)

        G = torch.cat([fF, fz, fv, fR, fu], dim=1)
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
                'vtilde': v_tilde.detach(),
                'c_t': c_t.detach(),
                'R_controlled': R_controlled.detach(),
                'F_R': FR.detach(),
                'I_TR': I_TR.detach(),
            }
            return logits, aux

        return logits


def count_turs_crm_params(model):
    out = {}
    groups = [
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
        ('P_beta', model.P_beta),
        ('P_a', model.P_a),
        ('P_c', model.P_c),
        ('P_u', model.P_u),
        ('P_R', model.P_R),
        ('P_g', model.P_g),
        ('P_T', model.P_T),
        ('P_I', model.P_I),
        ('P_RR', model.P_RR),
        ('P_RI', model.P_RI),
        ('P_alpha', model.P_alpha),
        ('classifier', model.classifier),
    ]
    for name, module in groups:
        out[name] = {
            'params_total': sum(p.numel() for p in module.parameters()),
            'params_trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
        }
    return out


__all__ = [
    'TURSCRM',
    'SoftDilatedDepthwise',
    'DepthwiseSeparableBranch',
    'ScaleParams',
    'ema_within_window',
    'ResponseBranch',
    'count_turs_crm_params',
]
