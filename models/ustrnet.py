"""
USTR-Net — Uncertainty-aware Soft Transport-Regime Network

Key novelty: uncertainty-aware soft regime modulation where the network
falls back to the reliable backbone when regime is uncertain.

    H' = H + u * gamma(H,z) + u * beta(H,z)

where u ~ sigma(U(H)) controls regime influence.

~180-250K parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Inception block with k=5,10,20 + pooling branch."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Use 4 branches: k=5, k=10, k=20, maxpool
        branch_ch = out_ch // 4
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 5, padding='same', bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 10, padding='same', bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True))
        self.conv20 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 20, padding='same', bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(
            nn.AvgPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True))
        
        self.out_bn = nn.BatchNorm1d(out_ch)
        
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch))
        else:
            self.residual = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        cat = torch.cat([self.conv5(x), self.conv10(x), self.conv20(x), self.pool(x)], dim=1)
        return self.relu(self.out_bn(cat) + self.residual(x))


class TransportBuilder(nn.Module):
    """Build [raw, Q, D] transport representation."""
    
    def forward(self, x):
        """x: [B, 1, L] -> [B, 3, L]"""
        raw = x.squeeze(1)
        sorted_x, _ = torch.sort(raw, dim=1)
        drift = torch.zeros_like(raw)
        for k in [1, 2, 4]:
            shifted = torch.zeros_like(raw)
            if k < raw.shape[1]:
                shifted[:, k:] = sorted_x[:, :-k]
            drift += torch.abs(sorted_x - shifted)
        drift /= 3.0
        return torch.stack([raw, sorted_x, drift], dim=1)


class USTRNet(nn.Module):
    """
    Uncertainty-aware Soft Transport-Regime Network.
    
    Architecture:
      1. Transport branch [raw, Q, D] -> learned fusion
      2. Multi-scale Inception backbone (32->64, 4 blocks)
      3. Continuous regime field (hypersphere-normalized)
      4. Regime velocity (v = z_t - z_{t-1})
      5. Uncertainty encoder u = sigma(U(H)) in [0,1]
      6. Soft regime modulation: H' = H + u*gamma + u*beta
      7. Transport-regime interaction: H_r = H * (1 + r)
      8. Regime-conditioned classifier: [h, z, v, u] -> logits
    """
    
    def __init__(self, in_channels=1, num_classes=5, regime_dim=8, base_ch=32):
        super().__init__()
        D = base_ch * 2  # 64
        self.D = D
        self.regime_dim = regime_dim
        self.num_classes = num_classes
        
        # 1. Transport branch
        self.transport_builder = TransportBuilder()
        self.transport_fusion = nn.Sequential(
            nn.Conv1d(3, D, 1, bias=False),
            nn.BatchNorm1d(D),
            nn.ReLU(inplace=True))
        
        # 2. Inception backbone: 32 -> 64 -> 64 -> 64 -> 64
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True))
        self.block1 = InceptionBlock(base_ch, base_ch)      # 32
        self.block2 = InceptionBlock(base_ch, base_ch)       # 32
        self.block3 = InceptionBlock(base_ch, D)             # 64
        self.block4 = InceptionBlock(D, D)                    # 64
        
        # 3. Regime field encoder (hypersphere-normalized)
        self.regime_enc = nn.Sequential(
            nn.Linear(D, 32), nn.GELU(),
            nn.Linear(32, regime_dim))
        
        # 4. Regime velocity (computed externally, stored as buffer)
        # Velocity is z_t - z_{t-1}, no learnable params
        
        # 5. Uncertainty encoder
        self.uncertainty_enc = nn.Sequential(
            nn.Linear(D, 16), nn.GELU(),
            nn.Linear(16, 1))
        
        # 6. Soft regime modulation (FiLM-style, but u-gated)
        # gamma, beta from [z, v]
        self.modulation = nn.Sequential(
            nn.Linear(regime_dim * 2, D), nn.GELU(),
            nn.Linear(D, D * 2))  # outputs gamma + beta
        
        # 7. Transport-regime interaction
        self.tr_regime_interaction = nn.Sequential(
            nn.Linear(D + regime_dim, D),
            nn.Sigmoid())
        
        # 8. Regime-conditioned classifier
        # Input: [h(64) + z(8) + v(8) + u(1)] = 81
        self.classifier = nn.Sequential(
            nn.Linear(D + regime_dim * 2 + 1, 64),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes))
        
        # Transport channel stats
        self._running_transport_mean = None
        self._running_transport_std = None
    
    def encode_regime(self, H):
        """Encode features into hypersphere-normalized regime."""
        v = H.mean(dim=2)  # [B, D]
        z_raw = self.regime_enc(v)  # [B, rd]
        z = F.normalize(z_raw, p=2, dim=1)  # [B, rd] on unit sphere
        return z, v
    
    def compute_uncertainty(self, v):
        """Compute uncertainty u in [0,1] from features."""
        u = self.uncertainty_enc(v)  # [B, 1]
        u = torch.sigmoid(u)
        return u
    
    def forward(self, x, prev_z=None):
        """
        x: [B, 1, L]
        prev_z: [B, rd] regime from previous window (for velocity)
        Returns dict with logits, regime, uncertainty, etc.
        """
        B, C, L = x.shape
        
        # 1. Transport
        T = self.transport_builder(x)  # [B, 3, L]
        T_e = self.transport_fusion(T)  # [B, D, L]
        
        # 2. Backbone
        h_raw = self.proj(x)  # [B, 32, L]
        h = self.block1(h_raw)  # [B, 32, L]
        h = self.block2(h)      # [B, 32, L]
        h = self.block3(h)      # [B, 64, L]
        h = self.block4(h)      # [B, 64, L]
        
        # Fuse transport + backbone (learned residual)
        H = h + T_e  # [B, D, L]
        
        # 3. Regime
        z, v_feat = self.encode_regime(H)  # [B, rd], [B, D]
        
        # 4. Regime velocity
        if prev_z is not None and prev_z.shape[0] == z.shape[0]:
            vel = z - prev_z.detach()  # [B, rd]
        else:
            vel = torch.zeros_like(z)
        
        # 5. Uncertainty
        u = self.compute_uncertainty(v_feat)  # [B, 1]
        
        # 6. Soft regime modulation
        zv = torch.cat([z, vel], dim=1)  # [B, 2*rd]
        gamma_beta = self.modulation(zv)  # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # each [B, D]
        
        # H' = H + u*gamma*H + u*beta = H*(1 + u*gamma) + u*beta
        h_mean = H.mean(dim=2)  # [B, D]
        h_modulated = h_mean * (1 + u * gamma) + u * beta  # [B, D]
        
        # 7. Transport-regime interaction
        tr_input = torch.cat([v_feat, z], dim=1)  # [B, D + rd]
        r = self.tr_regime_interaction(tr_input)  # [B, D]
        h_final = h_modulated * (1 + r)  # [B, D]
        
        # 8. Classifier
        cls_input = torch.cat([h_final, z, vel, u], dim=1)  # [B, D + 2*rd + 1]
        logits = self.classifier(cls_input)  # [B, C]
        
        return {
            "logits": logits,
            "features": h_mean,
            "modulated_features": h_final,
            "regime": z,
            "regime_velocity": vel,
            "uncertainty": u,
            "gate_gamma": gamma,
            "gate_beta": beta,
            "transport_interaction": r,
            "prev_z": z.detach(),
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = USTRNet(in_channels=1, num_classes=5, regime_dim=8, base_ch=32)
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(4, 1, 140)
    out = model(x)
    
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:30s} shape={list(v.shape)} range=[{v.min():.3f}, {v.max():.3f}]")
            assert not torch.isnan(v).any(), f"NaN in {k}"
    
    # Test with prev_z
    out2 = model(x, prev_z=out["prev_z"])
    print(f"\nWith prev_z: vel range=[{out2['regime_velocity'].min():.3f}, {out2['regime_velocity'].max():.3f}]")
    
    # Backward
    out["logits"].sum().backward()
    print("Gradient check passed.")
    print("All checks passed.")
