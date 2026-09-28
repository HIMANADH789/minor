"""
HCRMN — Hierarchical Continuous Regime Manifold Network

Key innovations over CMRM:
1. Dual-path backbone: Raw Inception + Transport encoder with cross-attention
2. Hierarchical regime manifold: Fine→Mid→Coarse with conditional geometry
3. Regime-Adaptive FiLM modulation: Regime modulates CNN features (bidirectional H↔Z)
4. Multi-scale regime graph with cross-scale edges
5. Residual MoE with multi-scale expert specialization
6. Second-order dynamics with change-aware gate
7. Rich final representation preserving all information sources
8. 7 distinct regularization losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# STAGE 1: Transport Channel Computation
# ============================================================
def compute_transport_channels(X):
    """
    Compute [Q, D, M] from raw signal X.
    X: (N, L) raw signals
    Returns: (N, 3, L) — [quantile, signed_drift, drift_magnitude]
    """
    N, L = X.shape
    out = np.zeros((N, 3, L), dtype=np.float32)
    for i in range(N):
        s = np.sort(X[i])
        # Q: quantile function
        Q = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(s)), s)
        out[i, 0, :] = Q
        # D: signed drift (direction of deformation)
        D = np.zeros(L)
        M = np.zeros(L)
        for lag in [1, 2, 4]:
            diff = np.diff(s, n=lag)
            pos = np.maximum(diff, 0)
            neg = np.maximum(-diff, 0)
            D[lag:] += (pos - neg) / 3.0
            M[lag:] += (pos + neg) / 3.0
        out[i, 1, :] = D
        out[i, 2, :] = M
    # Normalize per channel
    for c in range(3):
        mu = np.mean(out[:, c, :], axis=-1, keepdims=True)
        sig = np.std(out[:, c, :], axis=-1, keepdims=True) + 1e-8
        out[:, c, :] = (out[:, c, :] - mu) / sig
    return out


# Needed for compute_transport_channels
import numpy as np


# ============================================================
# STAGE 2A: Raw Inception Path
# ============================================================
class InceptionBlock(nn.Module):
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


class RawInceptionPath(nn.Module):
    """Standard InceptionTime backbone → H_raw (B, d, L)"""
    def __init__(self, n_blocks=6, base_ch=32, feat_dim=128):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = 1
        for _ in range(n_blocks):
            self.blocks.append(InceptionBlock(ch, base_ch, (5, 10, 20)))
            ch = base_ch * 4
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.feat_dim = feat_dim
        self.hidden_ch = ch

    def forward(self, x):
        """x: (B, 1, L) → h: (B, feat_dim)"""
        for blk in self.blocks:
            x = blk(x)
        h_seq = x  # keep sequence for cross-attention
        x = self.gap(x).squeeze(-1)
        return self.norm(self.proj(x)), h_seq


# ============================================================
# STAGE 2B: Transport Encoder
# ============================================================
class TransportEncoder(nn.Module):
    """Multi-scale conv encoder on [Q, D, M] → transport features"""
    def __init__(self, out_ch=64, kernels=(3, 7, 15, 31)):
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
        """x_tr: (B, 3, L) → H_tr: (B, out_ch, L)"""
        parts = [c(x_tr) for c in self.convs]
        cat = torch.cat(parts, dim=1)
        return self.proj(self.act(self.norm(cat)))


# ============================================================
# STAGE 2C: Transport-to-Temporal Cross Attention
# ============================================================
class TransportCrossAttention(nn.Module):
    """
    Transport queries temporal: A = softmax(Q_tr K_raw^T / sqrt(d))
    H_cross = A V_raw
    H_fused = H_raw + γ_tr H_cross + γ_raw H_tr
    """
    def __init__(self, raw_ch, tr_ch, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_d = raw_ch // n_heads
        assert raw_ch % n_heads == 0

        self.q_proj = nn.Linear(tr_ch, raw_ch)
        self.k_proj = nn.Linear(raw_ch, raw_ch)
        self.v_proj = nn.Linear(raw_ch, raw_ch)
        self.out_proj = nn.Linear(raw_ch, raw_ch)

        # Learned gates
        self.gamma_tr = nn.Parameter(torch.tensor(0.1))
        self.gamma_raw = nn.Parameter(torch.tensor(0.5))

        # Project transport features to raw_ch for the residual add
        self.tr_proj = nn.Conv1d(tr_ch, raw_ch, 1, bias=False)

        self.norm = nn.BatchNorm1d(raw_ch)

    def forward(self, H_raw, H_tr):
        """
        H_raw: (B, raw_ch, L) — temporal features (sequence)
        H_tr: (B, tr_ch, L) — transport features (sequence)
        Returns: H_fused: (B, raw_ch, L)
        """
        B, d, L = H_raw.shape
        n_heads, head_d = self.n_heads, self.head_d

        # Transpose to (B, L, ch) for attention
        Hr = H_raw.permute(0, 2, 1)  # (B, L, raw_ch)
        Ht = H_tr.permute(0, 2, 1)  # (B, L, tr_ch)

        Q = self.q_proj(Ht).view(B, L, n_heads, head_d).transpose(1, 2)  # (B, nh, L, hd)
        K = self.k_proj(Hr).view(B, L, n_heads, head_d).transpose(1, 2)
        V = self.v_proj(Hr).view(B, L, n_heads, head_d).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(head_d)  # (B, nh, L, L)
        attn = F.softmax(attn, dim=-1)
        cross = torch.matmul(attn, V)  # (B, nh, L, hd)
        cross = cross.transpose(1, 2).contiguous().view(B, L, d)  # (B, L, raw_ch)
        cross = self.out_proj(cross).permute(0, 2, 1)  # (B, raw_ch, L)

        # Fused: H_raw + γ_tr * H_cross + γ_raw * H_tr (projected to raw_ch)
        H_tr_up = F.adaptive_avg_pool1d(H_tr, H_raw.shape[-1])  # match length
        H_tr_proj = self.tr_proj(H_tr_up)  # (B, raw_ch, L)
        H_fused = H_raw + self.gamma_tr * cross + self.gamma_raw * H_tr_proj
        return self.norm(H_fused)


# ============================================================
# STAGE 3: Hierarchical Regime Manifold
# ============================================================
class RegimeEncoder(nn.Module):
    def __init__(self, in_dim, regime_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, regime_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(regime_dim * 2, regime_dim),
        )
    def forward(self, x):
        return self.net(x)


class HierarchicalRegimeEncoder(nn.Module):
    """
    H → z^(1) → z^(2) → z^(3) with cross-scale conditioning.
    Fine: E1(H)
    Mid:  E2(H, z_fine)   — conditioned on fine
    Coarse: E3(H, z_mid)  — conditioned on mid
    """
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        self.regime_dim = regime_dim

        self.fine_encoder = RegimeEncoder(feat_dim, regime_dim)
        self.mid_encoder = RegimeEncoder(feat_dim + regime_dim, regime_dim)
        self.coarse_encoder = RegimeEncoder(feat_dim + regime_dim, regime_dim)

        # Cross-attention for conditioning
        self.mid_attn = nn.MultiheadAttention(regime_dim, 4, batch_first=True, dropout=0.1)
        self.mid_norm = nn.LayerNorm(regime_dim)
        self.coarse_attn = nn.MultiheadAttention(regime_dim, 4, batch_first=True, dropout=0.1)
        self.coarse_norm = nn.LayerNorm(regime_dim)

    def forward(self, h):
        """h: (B, feat_dim) → (z_fine, z_mid, z_coarse)"""
        z_fine = self.fine_encoder(h)

        # Mid: conditioned on fine
        z_mid_base = self.mid_encoder(torch.cat([h, z_fine], dim=-1))
        z_mid_q = z_mid_base.unsqueeze(1)
        z_fine_kv = z_fine.unsqueeze(1)
        mid_out, _ = self.mid_attn(z_mid_q, z_fine_kv, z_fine_kv)
        z_mid = self.mid_norm(z_mid_base + mid_out.squeeze(1))

        # Coarse: conditioned on mid
        z_coarse_base = self.coarse_encoder(torch.cat([h, z_mid], dim=-1))
        z_coarse_q = z_coarse_base.unsqueeze(1)
        z_mid_kv = z_mid.unsqueeze(1)
        coarse_out, _ = self.coarse_attn(z_coarse_q, z_mid_kv, z_mid_kv)
        z_coarse = self.coarse_norm(z_coarse_base + coarse_out.squeeze(1))

        return z_fine, z_mid, z_coarse


# ============================================================
# STAGE 4: Regime Graph with Cross-Scale Edges
# ============================================================
class RegimeGraph(nn.Module):
    """
    Multi-scale prototype graph with cross-scale edges.
    Adjacency learned from prototype geometry.
    """
    def __init__(self, regime_dim=32, K=8):
        super().__init__()
        self.K = K
        self.regime_dim = regime_dim

        # Prototypes at each scale
        self.prototypes_fine = nn.Parameter(torch.randn(K, regime_dim) * 0.1)
        self.prototypes_mid = nn.Parameter(torch.randn(K, regime_dim) * 0.1)
        self.prototypes_coarse = nn.Parameter(torch.randn(K, regime_dim) * 0.1)

        # Cross-scale projection matrices
        self.W_f2m = nn.Linear(regime_dim, regime_dim, bias=False)
        self.W_m2c = nn.Linear(regime_dim, regime_dim, bias=False)

        # Learnable temperatures
        self.temp_fine = nn.Parameter(torch.tensor(1.0))
        self.temp_mid = nn.Parameter(torch.tensor(1.0))
        self.temp_coarse = nn.Parameter(torch.tensor(1.0))
        self.temp_f2m = nn.Parameter(torch.tensor(1.0))
        self.temp_m2c = nn.Parameter(torch.tensor(1.0))

        # Propagation mix coefficients
        self.alpha_fine = nn.Parameter(torch.tensor(0.3))
        self.alpha_mid = nn.Parameter(torch.tensor(0.3))
        self.alpha_coarse = nn.Parameter(torch.tensor(0.3))
        self.beta_f2m = nn.Parameter(torch.tensor(0.2))
        self.beta_m2c = nn.Parameter(torch.tensor(0.2))

    def propagate(self, z, prototypes, adj, alpha):
        """1-hop graph propagation: each sample gets weighted sum of prototypes"""
        # z: (B, rd), prototypes: (K, rd), adj: (K, K)
        # Compute per-sample attention over prototypes based on distance
        dists = torch.cdist(z, prototypes)  # (B, K)
        attn = F.softmax(-dists, dim=-1)  # (B, K)
        propagated = torch.matmul(attn, prototypes)  # (B, rd)
        return z + torch.sigmoid(alpha) * propagated

    def forward(self, z_fine, z_mid, z_coarse):
        """
        Returns: (z_fine_prop, z_mid_prop, z_coarse_prop), adj_info
        """
        K = self.K

        # Intra-scale adjacency
        for name, protos, temp in [
            ('fine', self.prototypes_fine, self.temp_fine),
            ('mid', self.prototypes_mid, self.temp_mid),
            ('coarse', self.prototypes_coarse, self.temp_coarse)
        ]:
            dists = torch.cdist(protos, protos)
            adj = F.softmax(-dists / torch.clamp(torch.exp(temp), min=0.1), dim=-1)
            if name == 'fine':
                adj_fine = adj
            elif name == 'mid':
                adj_mid = adj
            else:
                adj_coarse = adj

        # Cross-scale adjacency (fine→mid, mid→coarse)
        pf_m = self.W_f2m(self.prototypes_fine)
        pm_c = self.W_m2c(self.prototypes_mid)

        dists_f2m = torch.cdist(pf_m, self.prototypes_mid)
        adj_f2m = F.softmax(-dists_f2m / torch.clamp(torch.exp(self.temp_f2m), min=0.1), dim=-1)

        dists_m2c = torch.cdist(pm_c, self.prototypes_coarse)
        adj_m2c = F.softmax(-dists_m2c / torch.clamp(torch.exp(self.temp_m2c), min=0.1), dim=-1)

        # Intra-scale propagation
        z_f = self.propagate(z_fine, self.prototypes_fine, adj_fine, self.alpha_fine)
        z_m = self.propagate(z_mid, self.prototypes_mid, adj_mid, self.alpha_mid)
        z_c = self.propagate(z_coarse, self.prototypes_coarse, adj_coarse, self.alpha_coarse)

        # Cross-scale propagation: fine gets info from mid, mid gets from coarse
        # adj_f2m: (K_fine, K_mid) — fine prototypes project to mid space
        # Each sample's fine regime gets info from mid prototypes weighted by cross-scale similarity
        dists_f2m_sample = torch.cdist(z_f, self.prototypes_mid)  # (B, K_mid)
        attn_f2m = F.softmax(-dists_f2m_sample, dim=-1)  # (B, K_mid)
        mid_info = torch.matmul(attn_f2m, self.prototypes_mid)  # (B, rd)
        z_f = z_f + torch.sigmoid(self.beta_f2m) * mid_info

        dists_m2c_sample = torch.cdist(z_m, self.prototypes_coarse)  # (B, K_coarse)
        attn_m2c = F.softmax(-dists_m2c_sample, dim=-1)  # (B, K_coarse)
        coarse_info = torch.matmul(attn_m2c, self.prototypes_coarse)  # (B, rd)
        z_m = z_m + torch.sigmoid(self.beta_m2c) * coarse_info

        adj_info = {
            'adj_fine': adj_fine, 'adj_mid': adj_mid, 'adj_coarse': adj_coarse,
            'adj_f2m': adj_f2m, 'adj_m2c': adj_m2c,
        }
        return z_f, z_m, z_c, adj_info


# ============================================================
# STAGE 5: Regime-Adaptive Feature Modulation (FiLM)
# ============================================================
class RegimeFiLM(nn.Module):
    """
    Regime-conditioned FiLM: γ,β = M(z_fine, z_mid, z_coarse)
    H_modulated = γ ⊙ H + β
    Creates bidirectional coupling: H→Z and Z→H.
    """
    def __init__(self, regime_dim=32, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(regime_dim * 3, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim * 2),  # output gamma and beta
        )
        self.feat_dim = feat_dim
        # Initialize near identity
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data[:feat_dim].fill_(1.0)  # gamma starts at 1
        self.net[-1].bias.data[feat_dim:].zero_()      # beta starts at 0

    def forward(self, z_fine, z_mid, z_coarse, h):
        """
        z_fine/z_mid/z_coarse: (B, rd)
        h: (B, feat_dim)
        Returns: h_modulated: (B, feat_dim)
        """
        z_cat = torch.cat([z_fine, z_mid, z_coarse], dim=-1)  # (B, 3*rd)
        gamma_beta = self.net(z_cat)  # (B, 2*feat_dim)
        gamma = gamma_beta[:, :self.feat_dim]  # (B, fd)
        beta = gamma_beta[:, self.feat_dim:]   # (B, fd)
        # Soft clamp gamma to prevent collapse
        gamma = 0.5 + 0.5 * torch.tanh(gamma)  # in (0, 1)
        return gamma * h + beta


# ============================================================
# STAGE 6: Residual MoE with Multi-Scale Expert Specialization
# ============================================================
class MultiScaleMoE(nn.Module):
    """
    Residual MoE: H_out = H + λ_moe * H_reg
    H_reg = Σ_s ρ_s Σ_k α_k^(s) E_k^(s)(H)
    Each scale has its own set of experts.
    """
    def __init__(self, feat_dim=128, regime_dim=32, num_classes=5, K=8, n_scales=3):
        super().__init__()
        self.K = K
        self.n_scales = n_scales
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Experts per scale
        self.experts = nn.ModuleDict()
        for s in range(n_scales):
            experts_s = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feat_dim, 64), nn.GELU(),
                    nn.Dropout(0.1), nn.Linear(64, feat_dim)
                ) for _ in range(K)
            ])
            self.experts[f'scale_{s}'] = experts_s

        # Scale mixing weights
        self.scale_mix = nn.Linear(regime_dim * 3, n_scales)

        # Attention temperature per scale
        self.temp = nn.Parameter(torch.ones(n_scales))

        # Residual coefficient
        self.lambda_moe = nn.Parameter(torch.tensor(0.3))

        # Classification head (operates on H_out)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, h, z_fine, z_mid, z_coarse, prototypes_list):
        """
        h: (B, fd)
        z_fine/z_mid/z_coarse: (B, rd)
        prototypes_list: [protos_fine, protos_mid, protos_coarse] each (K, rd)
        Returns: logits (B, nc), expert_info dict
        """
        B = h.shape[0]
        z_all = torch.cat([z_fine, z_mid, z_coarse], dim=-1)

        # Scale mixing weights
        rho = F.softmax(self.scale_mix(z_all), dim=-1)  # (B, n_scales)

        H_reg = torch.zeros_like(h)
        all_attn = []

        for s in range(self.n_scales):
            protos = prototypes_list[s]  # (K, rd)
            # Compute attention for this scale
            zs = [z_fine, z_mid, z_coarse][s]
            dists = torch.cdist(zs, protos)  # (B, K)
            temp = torch.clamp(self.temp[s], min=0.1, max=5.0)
            attn = F.softmax(-dists / temp, dim=-1)  # (B, K)
            all_attn.append(attn)

            # Expert outputs
            experts_s = self.experts[f'scale_{s}']
            expert_outs = torch.stack([e(h) for e in experts_s], dim=1)  # (B, K, fd)
            h_s = (expert_outs * attn.unsqueeze(-1)).sum(dim=1)  # (B, fd)

            H_reg = H_reg + rho[:, s:s+1] * h_s

        # Residual connection
        lam = torch.sigmoid(self.lambda_moe)
        H_out = h + lam * H_reg

        # Classification
        logits = self.head(H_out)

        info = {
            'H_out': H_out,
            'H_reg': H_reg,
            'all_attn': all_attn,
            'rho': rho,
            'lambda_moe': lam.item(),
        }
        return logits, info


# ============================================================
# STAGE 7: Second-Order Dynamics with Change-Aware Gate
# ============================================================
class SecondOrderDynamics(nn.Module):
    """
    v_t = z_t - z_{t-1}
    a_t = v_t - v_{t-1}
    ẑ_{t+1} = z_t + v_t + f_ξ(z_t, v_t, H_t)
    """
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        self.acceleration_net = nn.Sequential(
            nn.Linear(regime_dim * 2 + feat_dim, regime_dim * 2),
            nn.GELU(),
            nn.Linear(regime_dim * 2, regime_dim),
        )
        self.change_gate = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, h):
        """
        z: (B, rd) — current regime
        h: (B, fd) — current features
        Returns: z_next (B, rd), velocity (B, rd), acceleration (B, rd), change_gate (B, 1)
        """
        # Velocity (approximate from features)
        v = torch.zeros_like(z)  # placeholder for single-window mode
        a = self.acceleration_net(torch.cat([z, v, h], dim=-1))
        z_next = z + v + a
        c = self.change_gate(h)
        return z_next, v, a, c


# ============================================================
# HCRMN: Full Model
# ============================================================
class HCRMN(nn.Module):
    def __init__(self, in_channels=4, num_classes=5, feat_dim=128,
                 regime_dim=32, n_experts=8, n_blocks=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.regime_dim = regime_dim
        self.num_classes = num_classes
        self.n_experts = n_experts

        # Stage 2: Dual backbone
        self.raw_path = RawInceptionPath(n_blocks=n_blocks, base_ch=32, feat_dim=feat_dim)
        self.transport_encoder = TransportEncoder(out_ch=64)
        self.cross_attn = TransportCrossAttention(
            raw_ch=self.raw_path.hidden_ch, tr_ch=64, n_heads=4
        )

        # Projection from fused conv features to feat_dim for regime encoding
        self.fused_proj = nn.Linear(self.raw_path.hidden_ch, feat_dim)
        self.fused_norm = nn.LayerNorm(feat_dim)

        # Stage 3: Hierarchical regime encoder
        self.hier_regime = HierarchicalRegimeEncoder(feat_dim, regime_dim)

        # Stage 4: Regime graph
        self.regime_graph = RegimeGraph(regime_dim, K=n_experts)

        # Stage 5: FiLM modulation
        self.film = RegimeFiLM(regime_dim, feat_dim)

        # Stage 6: Multi-scale residual MoE
        self.moe = MultiScaleMoE(feat_dim, regime_dim, num_classes, K=n_experts)

        # Stage 7: Dynamics
        self.dynamics = SecondOrderDynamics(feat_dim, regime_dim)

    def forward(self, x):
        """
        x: (B, 4, L) — [raw, Q, D, M]
        Returns: logits, info_dict
        """
        B = x.shape[0]
        raw = x[:, :1, :]     # (B, 1, L)
        x_tr = x[:, 1:4, :]   # (B, 3, L)

        # Stage 2A: Raw path → H_raw (feat_dim,), h_seq (hidden_ch, L)
        H_raw, h_seq = self.raw_path(raw)

        # Stage 2B: Transport → H_tr (tr_ch, L)
        H_tr = self.transport_encoder(x_tr)

        # Stage 2C: Cross attention → H_fused (hidden_ch, L)
        H_fused = self.cross_attn(h_seq, H_tr)

        # Project fused conv features to feat_dim
        H_fused_pooled = H_fused.mean(dim=-1)  # (B, hidden_ch)
        H_fused_proj = self.fused_norm(self.fused_proj(H_fused_pooled))  # (B, feat_dim)

        # Combine raw path + fused path
        h = H_raw + H_fused_proj  # (B, feat_dim)

        # Stage 3: Hierarchical regime encoding
        z_fine, z_mid, z_coarse = self.hier_regime(h)

        # Stage 4: Regime graph propagation
        z_f, z_m, z_c, adj_info = self.regime_graph(z_fine, z_mid, z_coarse)

        # Stage 5: FiLM modulation
        h_modulated = self.film(z_f, z_m, z_c, h)

        # Stage 6: Multi-scale MoE
        prototypes = [
            self.regime_graph.prototypes_fine,
            self.regime_graph.prototypes_mid,
            self.regime_graph.prototypes_coarse,
        ]
        logits, moe_info = self.moe(h_modulated, z_f, z_m, z_c, prototypes)

        # Stage 7: Dynamics
        z_merged = torch.cat([z_f, z_m, z_c], dim=-1)
        z_next, vel, accel, change_gate = self.dynamics(z_f, h_modulated)

        info = {
            'h': h,
            'h_modulated': h_modulated,
            'H_raw': H_raw,
            'z_fine': z_f, 'z_mid': z_m, 'z_coarse': z_c,
            'z_fine_raw': z_fine, 'z_mid_raw': z_mid, 'z_coarse_raw': z_coarse,
            'z_next': z_next, 'velocity': vel, 'acceleration': accel,
            'change_gate': change_gate,
            'prototypes': prototypes,
            'adj_info': adj_info,
            'moe_info': moe_info,
            'H_reg': moe_info['H_reg'],
        }
        return logits, info


# ============================================================
# HCRMN Loss: 7 distinct regularization terms
# ============================================================
import numpy as np

class HCRMNLoss(nn.Module):
    """
    L = L_focal + λ1*L_smooth + λ2*L_dyn + λ3*L_hier
        + λ4*L_graph + λ5*L_proto + λ6*L_cross + λ7*L_rep
    """
    def __init__(self, num_classes=5, feat_dim=128,
                 focal_gamma=1.0,
                 lambda_smooth=0.01, lambda_dyn=0.005,
                 lambda_hier=0.01, lambda_graph=0.005,
                 lambda_proto=0.01, lambda_cross=0.005,
                 lambda_rep=0.005,
                 proto_sep=2.0, contrastive_margin=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.lambda_smooth = lambda_smooth
        self.lambda_dyn = lambda_dyn
        self.lambda_hier = lambda_hier
        self.lambda_graph = lambda_graph
        self.lambda_proto = lambda_proto
        self.lambda_cross = lambda_cross
        self.lambda_rep = lambda_rep
        self.proto_sep = proto_sep
        self.contrastive_margin = contrastive_margin

        # Representation preservation projection
        self.rep_proj = nn.Linear(feat_dim, feat_dim)

    def focal_ce(self, logits, targets):
        if self.focal_gamma == 0:
            return F.cross_entropy(logits, targets)
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.focal_gamma * ce).mean()

    def forward(self, logits, info, targets):
        B = logits.shape[0]

        # ---- L_focal ----
        L_task = self.focal_ce(logits, targets)

        h = info['h']
        h_mod = info['h_modulated']
        z_f = info['z_fine']
        z_m = info['z_mid']
        z_c = info['z_coarse']
        z_f_raw = info['z_fine_raw']
        z_m_raw = info['z_mid_raw']
        z_c_raw = info['z_coarse_raw']
        change_gate = info['change_gate']

        # ---- L_smooth: change-aware smoothness ----
        # Smooth where change_gate is low (stable regions)
        L_smooth = ((1 - change_gate.squeeze(-1)) * torch.sum(z_f ** 2, dim=-1)).mean()

        # ---- L_dyn: dynamics consistency ----
        z_next = info['z_next']
        L_dyn = F.mse_loss(z_next, z_f.detach())

        # ---- L_hier: hierarchical consistency ----
        # z^(2) should be explainable from z^(1), z^(3) from z^(2)
        # Use learned projections
        L_hier_fm = F.mse_loss(z_m_raw, z_f_raw.detach()[:, :z_m_raw.shape[-1]])
        L_hier_mc = F.mse_loss(z_c_raw, z_m_raw.detach())
        L_hier = L_hier_fm + L_hier_mc

        # ---- L_graph: nearby prototypes should have compatible predictions ----
        prototypes = info['prototypes']  # list of 3 (K, rd)
        proto_preds = info['moe_info']['all_attn']  # list of 3 (B, K)
        # This is per-sample, so we compute it as prototype distance regularization
        L_graph = torch.tensor(0.0, device=logits.device)
        for s, protos in enumerate(prototypes):
            adj = info['adj_info'][f'adj_{["fine", "mid", "coarse"][s]}']
            # Regularize: propagate and check consistency
            propagated = torch.matmul(adj, protos)
            L_graph = L_graph + F.mse_loss(propagated, protos.detach()) * 0.1

        # ---- L_proto: coverage + separation ----
        L_coverage = torch.tensor(0.0, device=logits.device)
        zs = [z_f, z_m, z_c]
        for s, (z, protos) in enumerate(zip(zs, prototypes)):
            dists = torch.cdist(z, protos)  # (B, K)
            L_coverage = L_coverage + dists.min(dim=1)[0].mean()

        L_sep = torch.tensor(0.0, device=logits.device)
        for protos in prototypes:
            pd = torch.cdist(protos, protos)
            mask = ~torch.eye(protos.shape[0], dtype=torch.bool, device=protos.device)
            L_sep = L_sep + F.relu(self.proto_sep - pd[mask].min())

        L_proto = L_coverage + 0.1 * L_sep

        # ---- L_cross: cross-scale contrastive ----
        with torch.no_grad():
            feat_dists = torch.cdist(h, h)
            feat_dists.fill_diagonal_(float('inf'))
            k = min(4, B - 1)
            if k > 0:
                _, pos_idx = feat_dists.topk(k, largest=False)
                _, neg_idx = feat_dists.topk(k, largest=True)
            else:
                pos_idx = neg_idx = torch.zeros(B, 1, dtype=torch.long, device=h.device)
                k = 0

        if k > 0:
            z_all = torch.cat([z_f, z_m, z_c], dim=-1)  # (B, 3*rd)
            z_pos = z_all[pos_idx]  # (B, k, 3*rd)
            z_neg = z_all[neg_idx]  # (B, k, 3*rd)
            z_exp = z_all.unsqueeze(1).expand(-1, k, -1)

            d_pos = F.mse_loss(z_exp, z_pos)
            d_neg = F.relu(self.contrastive_margin - F.mse_loss(z_exp, z_neg))
            L_cross = d_pos + 0.5 * d_neg
        else:
            L_cross = torch.tensor(0.0, device=logits.device)

        # ---- L_rep: representation preservation ----
        h_proj = self.rep_proj(h_mod)
        L_rep = 1 - F.cosine_similarity(h_proj, h.detach(), dim=-1).mean()

        # ---- Total ----
        L_total = (L_task
                   + self.lambda_smooth * L_smooth
                   + self.lambda_dyn * L_dyn
                   + self.lambda_hier * L_hier
                   + self.lambda_graph * L_graph
                   + self.lambda_proto * L_proto
                   + self.lambda_cross * L_cross
                   + self.lambda_rep * L_rep)

        return L_total, {
            'task': L_task.item(),
            'smooth': L_smooth.item(),
            'dyn': L_dyn.item(),
            'hier': L_hier.item(),
            'graph': L_graph.item(),
            'proto': L_proto.item(),
            'cross': L_cross.item(),
            'rep': L_rep.item(),
        }
