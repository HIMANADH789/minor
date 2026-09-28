"""
HCRMN-Lite — Hierarchical Continuous Regime Manifold Network (Compact)

Key innovations retained from full HCRMN:
1. Dual backbone: Raw Inception + Transport branch with channel gating
2. Hierarchical regime manifold: z1→z2→z3 with cross-scale conditioning
3. FiLM modulation: regime modulates CNN features (bidirectional H↔Z)
4. Lightweight regime interaction (4 prototypes per scale, no graph GNN)
5. Second-order dynamics with change-aware gate
6. Shared continuous regime predictor (no separate experts)
7. 5 focused loss terms

Target: <400K parameters (vs 1.38M for full HCRMN, 262K for eTAI, 227K for InceptionTime)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ============================================================
# STAGE 1: Transport Channel Computation
# ============================================================
def compute_transport_channels_lite(X):
    """
    Compute [Q, D, M] from raw signal X.
    X: (N, L) raw signals
    Returns: (N, 3, L) — [quantile, drift_magnitude, signed_deformation]
    """
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    for i in range(N):
        s = np.sort(X[i])
        # Q: quantile function
        Q = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
        out[i, 0, :] = Q
        # D: drift magnitude (unsigned)
        D = np.zeros(L)
        # M: signed deformation
        M = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            D[lag:] += np.abs(diff) / 3.0
            M[lag:] += diff / 3.0
        out[i, 1, :] = D
        out[i, 2, :] = M
    # Normalize per channel
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


# ============================================================
# STAGE 2A: Compact Raw Inception Backbone (64ch × 4 blocks)
# ============================================================
class InceptionBlockLite(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(5, 10, 20)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, padding='same', bias=False) for k in kernels
        ])
        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
        )
        total = out_ch * len(kernels) + out_ch
        self.norm = nn.BatchNorm1d(total)
        self.shortcut = nn.Conv1d(in_ch, total, 1, bias=False) if in_ch != total else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        parts = [b(x) for b in self.branches] + [self.pool(x)]
        return self.act(self.norm(torch.cat(parts, dim=1)) + self.shortcut(x))


class RawInceptionLite(nn.Module):
    """Compact InceptionTime backbone: 64 channels × 4 blocks → H_raw (B, 64, L)"""
    def __init__(self, n_blocks=4, base_ch=16):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = 1
        for i in range(n_blocks):
            self.blocks.append(InceptionBlockLite(ch, base_ch, (5, 10, 20)))
            ch = base_ch * 4  # 64 channels
        self.hidden_ch = ch  # 64
        # Residual projection for skip connection
        self.res_proj = nn.Conv1d(1, ch, 1, bias=False) if True else nn.Identity()

    def forward(self, x):
        """x: (B, 1, L) → H_raw: (B, 64, L)"""
        shortcut = self.res_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x + shortcut  # residual connection


# ============================================================
# STAGE 2B: Lightweight Transport Encoder (32 channels)
# ============================================================
class TransportEncoderLite(nn.Module):
    """Compact multi-scale conv on [Q, D, M] → 32-channel transport features"""
    def __init__(self, out_ch=32, kernels=(3, 7, 15)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(3, out_ch, k, padding='same', bias=False) for k in kernels
        ])
        total = out_ch * len(kernels)
        self.norm = nn.BatchNorm1d(total)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(total, out_ch, 1, bias=False)
        self.out_ch = out_ch

    def forward(self, x_tr):
        """x_tr: (B, 3, L) → T_tr: (B, 32, L)"""
        parts = [c(x_tr) for c in self.convs]
        cat = torch.cat(parts, dim=1)
        return self.proj(self.act(self.norm(cat)))


# ============================================================
# STAGE 3: Transport–Temporal Fusion (Channel Gating)
# ============================================================
class ChannelGateFusion(nn.Module):
    """
    Lightweight transport→temporal fusion via channel gating.
    g_tr = σ(MLP(GAP(T_tr)))
    H_0 = H_raw + g_tr ⊙ P(T_tr)
    """
    def __init__(self, raw_ch=64, tr_ch=32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tr_ch, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv1d(tr_ch, raw_ch, 1, bias=False)
        self.norm = nn.BatchNorm1d(raw_ch)

    def forward(self, H_raw, T_tr):
        """
        H_raw: (B, 64, L)
        T_tr: (B, 32, L)
        Returns: H_0: (B, 64, L)
        """
        g = self.gate_net(T_tr)  # (B, 1)
        T_proj = self.proj(T_tr)  # (B, 64, L)
        H_0 = H_raw + g.unsqueeze(-1) * T_proj
        return self.norm(H_0)


# ============================================================
# STAGE 4: Hierarchical Regime Encoder (z1→z2→z3)
# ============================================================
class RegimeEncoderLite(nn.Module):
    def __init__(self, in_dim, regime_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, regime_dim),
        )

    def forward(self, x):
        return self.net(x)


class HierarchicalRegimeEncoderLite(nn.Module):
    """
    H → z^(1) → z^(2) → z^(3) with cross-scale conditioning.
    Fine: E1(GAP(H_f))
    Mid:  E2(GAP(H_m), z_fine)
    Coarse: E3(GAP(H_c), z_mid)
    """
    def __init__(self, feat_dim=64, regime_dim=16):
        super().__init__()
        self.regime_dim = regime_dim
        self.fine_encoder = RegimeEncoderLite(feat_dim, regime_dim)
        self.mid_encoder = RegimeEncoderLite(feat_dim + regime_dim, regime_dim)
        self.coarse_encoder = RegimeEncoderLite(feat_dim + regime_dim, regime_dim)

    def forward(self, h_f, h_m, h_c):
        """
        h_f, h_m, h_c: (B, feat_dim) — multi-scale pooled features
        Returns: (z_fine, z_mid, z_coarse) each (B, regime_dim)
        """
        z_fine = self.fine_encoder(h_f)
        z_mid = self.mid_encoder(torch.cat([h_m, z_fine], dim=-1))
        z_coarse = self.coarse_encoder(torch.cat([h_c, z_mid], dim=-1))
        return z_fine, z_mid, z_coarse


# ============================================================
# STAGE 5: Lightweight Regime Interaction
# ============================================================
class RegimeInteractionLite(nn.Module):
    """
    K=4 prototypes per scale.
    Prototype adjacency A^(s) ∈ R^{4×4}.
    Cross-scale: z1 += W12·z2, z2 += W23·z3
    """
    def __init__(self, regime_dim=16, K=4):
        super().__init__()
        self.K = K
        self.regime_dim = regime_dim

        # Prototypes at each scale
        self.prototypes_fine = nn.Parameter(torch.randn(K, regime_dim) * 0.1)
        self.prototypes_mid = nn.Parameter(torch.randn(K, regime_dim) * 0.1)
        self.prototypes_coarse = nn.Parameter(torch.randn(K, regime_dim) * 0.1)

        # Per-scale adjacency (learned)
        self.adj_fine = nn.Parameter(torch.eye(K) + torch.randn(K, K) * 0.01)
        self.adj_mid = nn.Parameter(torch.eye(K) + torch.randn(K, K) * 0.01)
        self.adj_coarse = nn.Parameter(torch.eye(K) + torch.randn(K, K) * 0.01)

        # Learnable temperatures per scale
        self.temp_fine = nn.Parameter(torch.tensor(1.0))
        self.temp_mid = nn.Parameter(torch.tensor(1.0))
        self.temp_coarse = nn.Parameter(torch.tensor(1.0))

        # Interaction strength per scale
        self.eta_fine = nn.Parameter(torch.tensor(0.3))
        self.eta_mid = nn.Parameter(torch.tensor(0.3))
        self.eta_coarse = nn.Parameter(torch.tensor(0.3))

        # Cross-scale projection
        self.W12 = nn.Linear(regime_dim, regime_dim, bias=False)
        self.W23 = nn.Linear(regime_dim, regime_dim, bias=False)

    def forward(self, z_fine, z_mid, z_coarse):
        """
        Returns: (z_fine, z_mid, z_coarse) after interaction
        """
        # Prototype interaction at each scale
        for z, protos, adj, temp, eta, scale in [
            (z_fine, self.prototypes_fine, self.adj_fine, self.temp_fine, self.eta_fine, 'fine'),
            (z_mid, self.prototypes_mid, self.adj_mid, self.temp_mid, self.eta_mid, 'mid'),
            (z_coarse, self.prototypes_coarse, self.adj_coarse, self.temp_coarse, self.eta_coarse, 'coarse'),
        ]:
            dists = torch.cdist(z, protos)  # (B, K)
            temp_clamped = torch.clamp(torch.exp(temp), min=0.1, max=5.0)
            alpha = F.softmax(-dists / temp_clamped, dim=-1)  # (B, K)
            # Propagate through adjacency
            P_proj = torch.matmul(adj, protos)  # (K, rd)
            propagated = torch.matmul(alpha, P_proj)  # (B, rd)
            z_new = z + torch.sigmoid(eta) * propagated

            if scale == 'fine':
                z_fine_new = z_new
            elif scale == 'mid':
                z_mid_new = z_new
            else:
                z_coarse_new = z_new

        # Cross-scale interaction: top-down
        z_fine_ctx = z_fine_new + self.W12(z_mid_new)
        z_mid_ctx = z_mid_new + self.W23(z_coarse_new)

        return z_fine_ctx, z_mid_ctx, z_coarse_new


# ============================================================
# STAGE 6: Lightweight FiLM
# ============================================================
class FiLMLite(nn.Module):
    """
    Regime-conditioned FiLM: [γ, β] = M(z_t)
    H* = H_0 + λ_f * (γ ⊙ H_0 + β)
    """
    def __init__(self, regime_dim=48, feat_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(regime_dim, 32),
            nn.GELU(),
            nn.Linear(32, feat_dim * 2),
        )
        self.feat_dim = feat_dim
        # Initialize near identity
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data[:feat_dim].fill_(1.0)
        self.net[-1].bias.data[feat_dim:].zero_()
        # Learnable residual strength
        self.lambda_f = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, H_0):
        """
        z: (B, 48) — combined regime state
        H_0: (B, 64, L)
        Returns: H*: (B, 64, L)
        """
        gb = self.net(z)  # (B, 128)
        gamma = gb[:, :self.feat_dim]  # (B, 64)
        beta = gb[:, self.feat_dim:]   # (B, 64)
        gamma = 0.5 + 0.5 * torch.tanh(gamma)  # (0, 1)
        # Apply with residual
        lam = torch.sigmoid(self.lambda_f)
        H_mod = gamma.unsqueeze(-1) * H_0 + beta.unsqueeze(-1)
        return H_0 + lam * H_mod


# ============================================================
# STAGE 7: Regime Dynamics
# ============================================================
class RegimeDynamicsLite(nn.Module):
    """
    v_t = V(H_t, z_t)
    a_t = A(H_t, z_t, v_t)
    ẑ_{t+1} = z_t + v_t + a_t
    """
    def __init__(self, feat_dim=64, regime_dim=48):
        super().__init__()
        self.velocity_net = nn.Sequential(
            nn.Linear(feat_dim + regime_dim, 32),
            nn.GELU(),
            nn.Linear(32, regime_dim),
        )
        self.acceleration_net = nn.Sequential(
            nn.Linear(regime_dim * 2 + feat_dim, 32),
            nn.GELU(),
            nn.Linear(32, regime_dim),
        )
        self.change_gate = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, h):
        """
        z: (B, 48) — current regime
        h: (B, 64) — pooled features
        Returns: z_next, v, a, change_gate
        """
        v = self.velocity_net(torch.cat([h, z], dim=-1))
        a = self.acceleration_net(torch.cat([z, v, h], dim=-1))
        z_next = z + v + a
        c = self.change_gate(h)
        return z_next, v, a, c


# ============================================================
# HCRMN-Lite: Full Model
# ============================================================
class HCRMNLite(nn.Module):
    """
    Compact HCRMN: Multi-scale CNN + Transport + Hierarchical Regimes + FiLM + Dynamics
    Target: <400K params
    """
    def __init__(self, in_channels=4, num_classes=5, feat_dim=64,
                 regime_dim=16, K=4, n_blocks=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.regime_dim = regime_dim
        self.num_classes = num_classes
        self.regime_dim_total = regime_dim * 3  # 48

        # Stage 2A: Raw Inception backbone
        self.raw_path = RawInceptionLite(n_blocks=n_blocks, base_ch=16)

        # Stage 2B: Transport encoder
        self.transport_encoder = TransportEncoderLite(out_ch=32)

        # Stage 3: Channel gate fusion
        self.fusion = ChannelGateFusion(raw_ch=feat_dim, tr_ch=32)

        # Stage 4: Multi-scale pooling (no learnable params)
        self.pool_2 = nn.AvgPool1d(2, stride=2)
        self.pool_4 = nn.AvgPool1d(4, stride=4)

        # Stage 5: Hierarchical regime encoder
        self.hier_regime = HierarchicalRegimeEncoderLite(feat_dim, regime_dim)

        # Stage 6: Regime interaction
        self.regime_interaction = RegimeInteractionLite(regime_dim, K)

        # Stage 7: FiLM modulation
        self.film = FiLMLite(self.regime_dim_total, feat_dim)

        # Stage 8: Dynamics
        self.dynamics = RegimeDynamicsLite(feat_dim, self.regime_dim_total)

        # Stage 9: Classification head
        # Input: GAP(H*) + z + v + a = 64 + 48 + 48 + 48 = 208
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + self.regime_dim_total * 3, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

        # Regime-specific bias (per prototype per class)
        self.prototype_bias = nn.Parameter(
            torch.randn(K, 3, num_classes) * 0.01  # K prototypes × 3 scales × C
        )

    def forward(self, x):
        """
        x: (B, 4, L) — [raw, Q, D, M]
        Returns: logits, info
        """
        B = x.shape[0]
        raw = x[:, :1, :]     # (B, 1, L)
        x_tr = x[:, 1:4, :]   # (B, 3, L)

        # Stage 2A: Raw path
        H_raw = self.raw_path(raw)  # (B, 64, L)

        # Stage 2B: Transport
        T_tr = self.transport_encoder(x_tr)  # (B, 32, L)

        # Stage 3: Fusion
        H_0 = self.fusion(H_raw, T_tr)  # (B, 64, L)

        # Stage 4: Multi-scale features
        h_f = H_0.mean(dim=-1)  # (B, 64)
        h_m = self.pool_2(H_0).mean(dim=-1)  # (B, 64)
        h_c = self.pool_4(H_0).mean(dim=-1)  # (B, 64)

        # Stage 5: Hierarchical regimes
        z_fine, z_mid, z_coarse = self.hier_regime(h_f, h_m, h_c)

        # Stage 6: Regime interaction
        z_f, z_m, z_c = self.regime_interaction(z_fine, z_mid, z_coarse)

        # Combine regime states
        z = torch.cat([z_f, z_m, z_c], dim=-1)  # (B, 48)

        # Stage 7: FiLM modulation
        H_star = self.film(z, H_0)  # (B, 64, L)

        # Stage 8: Dynamics
        z_next, vel, accel, change_gate = self.dynamics(z, h_f)

        # Stage 9: Classification
        h_pool = H_star.mean(dim=-1)  # (B, 64)
        r = torch.cat([h_pool, z, vel, accel], dim=-1)  # (B, 208)
        logits = self.classifier(r)  # (B, C)

        # Add prototype-specific bias
        # Compute per-scale alpha and interpolate prototype biases
        proto_biases = []
        for s, (zs, protos) in enumerate([
            (z_f, self.regime_interaction.prototypes_fine),
            (z_m, self.regime_interaction.prototypes_mid),
            (z_c, self.regime_interaction.prototypes_coarse),
        ]):
            dists = torch.cdist(zs, protos)  # (B, K)
            temps = [
                torch.clamp(torch.exp(self.regime_interaction.temp_fine), min=0.1),
                torch.clamp(torch.exp(self.regime_interaction.temp_mid), min=0.1),
                torch.clamp(torch.exp(self.regime_interaction.temp_coarse), min=0.1),
            ]
            alpha = F.softmax(-dists / temps[s], dim=-1)  # (B, K)
            # bias: (K, C) → (B, C)
            b_s = self.prototype_bias[:, s, :]  # (K, C)
            pb = torch.matmul(alpha, b_s)  # (B, C)
            proto_biases.append(pb)

        logits = logits + sum(proto_biases)

        # Collect info for loss
        info = {
            'h': h_pool,
            'H_star': H_star,
            'H_0': H_0,
            'z_fine': z_f, 'z_mid': z_m, 'z_coarse': z_c,
            'z_fine_raw': z_fine, 'z_mid_raw': z_mid, 'z_coarse_raw': z_coarse,
            'z': z,
            'z_next': z_next, 'velocity': vel, 'acceleration': accel,
            'change_gate': change_gate,
            'prototypes': [
                self.regime_interaction.prototypes_fine,
                self.regime_interaction.prototypes_mid,
                self.regime_interaction.prototypes_coarse,
            ],
        }
        return logits, info


# ============================================================
# HCRMN-Lite Loss: 5 focused losses
# ============================================================
class HCRMNLiteLoss(nn.Module):
    """
    L = L_task + λ_d*L_dyn + λ_h*L_hier + λ_p*L_proto + λ_s*L_smooth
    """
    def __init__(self, num_classes=5, feat_dim=64, regime_dim=16,
                 focal_gamma=0.0,
                 lambda_dyn=0.01, lambda_hier=0.01,
                 lambda_proto=0.01, lambda_smooth=0.01,
                 proto_sep=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.lambda_dyn = lambda_dyn
        self.lambda_hier = lambda_hier
        self.lambda_proto = lambda_proto
        self.lambda_smooth = lambda_smooth
        self.proto_sep = proto_sep
        self.regime_dim = regime_dim

        # Learned projections for hierarchical consistency
        self.g12 = nn.Linear(regime_dim, regime_dim)
        self.g23 = nn.Linear(regime_dim, regime_dim)

    def focal_ce(self, logits, targets):
        if self.focal_gamma == 0:
            return F.cross_entropy(logits, targets)
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.focal_gamma * ce).mean()

    def forward(self, logits, info, targets):
        B = logits.shape[0]

        # L_task
        L_task = self.focal_ce(logits, targets)

        z_f = info['z_fine']
        z_m = info['z_mid']
        z_c = info['z_coarse']
        z_f_raw = info['z_fine_raw']
        z_m_raw = info['z_mid_raw']
        z_c_raw = info['z_coarse_raw']
        z = info['z']

        # L_dyn: dynamics consistency
        z_next = info['z_next']
        L_dyn = F.mse_loss(z_next, z.detach())

        # L_hier: hierarchical consistency
        z2_pred = self.g12(z_f_raw)
        z3_pred = self.g23(z_m_raw)
        L_hier = F.mse_loss(z_m_raw.detach(), z2_pred) + F.mse_loss(z_c_raw.detach(), z3_pred)

        # L_proto: coverage + separation
        L_coverage = torch.tensor(0.0, device=logits.device)
        prototypes = info['prototypes']
        zs = [z_f, z_m, z_c]
        for s, (z_s, protos) in enumerate(zip(zs, prototypes)):
            dists = torch.cdist(z_s, protos)  # (B, K)
            L_coverage = L_coverage + dists.min(dim=1)[0].mean()

        L_sep = torch.tensor(0.0, device=logits.device)
        for protos in prototypes:
            pd = torch.cdist(protos, protos)
            mask = ~torch.eye(protos.shape[0], dtype=torch.bool, device=protos.device)
            if mask.any():
                L_sep = L_sep + F.relu(self.proto_sep - pd[mask].min())

        L_proto = L_coverage + 0.1 * L_sep

        # L_smooth: change-aware smoothness
        c = info['change_gate'].squeeze(-1)  # (B,)
        L_smooth = ((1 - c) * torch.sum(z ** 2, dim=-1)).mean()

        # Total
        L_total = (L_task
                   + self.lambda_dyn * L_dyn
                   + self.lambda_hier * L_hier
                   + self.lambda_proto * L_proto
                   + self.lambda_smooth * L_smooth)

        return L_total, {
            'task': L_task.item(),
            'dyn': L_dyn.item(),
            'hier': L_hier.item(),
            'proto': L_proto.item(),
            'smooth': L_smooth.item(),
        }
