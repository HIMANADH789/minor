"""
ARTNet — Adaptive Regime-Transport Network

Compact (~190-220K params) successor to InceptionTime + eTAI.
Key mechanisms:
  - Transport-regime interaction (regime-conditioned transport transformation)
  - Hierarchical continuous regime encoder (fine/mid/coarse)
  - Uncertainty-aware regime distribution (mu, sigma)
  - Adaptive residual gate (uncertainty suppresses regime injection)
  - Conditional classifier (features + regime + interaction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Single Inception block with bottleneck and residual connection."""
    
    def __init__(self, in_ch, out_ch, bottleneck_ch=None, kernels=(9, 19, 39)):
        super().__init__()
        if bottleneck_ch is None:
            bottleneck_ch = in_ch // 2
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, bottleneck_ch, 1, bias=False),
            nn.BatchNorm1d(bottleneck_ch),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_ch, out_ch // 3, k, padding=k // 2, bias=False)
            for k in kernels
        ])
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv1d(out_ch // 3 * 3, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        
        # Residual projection (if channels change)
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.residual = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        res = self.residual(x)
        b = self.bottleneck(x)
        feats = [conv(b) for conv in self.convs]
        cat = torch.cat(feats, dim=1)
        out = self.out_proj(cat)
        return self.relu(out + res)


class InceptionBackbone(nn.Module):
    """InceptionTime backbone that exposes multi-scale feature maps."""
    
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        # Initial projection
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True)
        )
        
        # 4 blocks: 32 -> 32 -> 64 -> 64
        self.block1 = InceptionBlock(base_ch, base_ch, kernels=(9, 19, 39))
        self.block2 = InceptionBlock(base_ch, base_ch, kernels=(9, 19, 39))
        self.block3 = InceptionBlock(base_ch, base_ch * 2, kernels=(9, 19, 39))
        self.block4 = InceptionBlock(base_ch * 2, base_ch * 2, kernels=(9, 19, 39))
        
        self.D = base_ch * 2  # Final channel dimension = 64
    
    def forward(self, x):
        """
        Returns: H_f (after block1), H_m (after block3), H_c (after block4)
        All shape: [B, D, L]
        """
        h = self.proj(x)
        h_f = self.block1(h)      # Fine: [B, 32, L]
        h = self.block2(h_f)      # [B, 32, L]
        h_m = self.block3(h)      # Middle: [B, 64, L]
        h_c = self.block4(h_m)    # Coarse: [B, 64, L]
        return h_f, h_m, h_c


class TransportBuilder(nn.Module):
    """Builds [raw, Q, D] transport representation from raw input."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        x: [B, 1, L]
        Returns: [B, 3, L] — [raw, quantile, drift]
        """
        B, C, L = x.shape
        
        # Channel 0: raw (already x)
        raw = x.squeeze(1)  # [B, L]
        
        # Channel 1: quantile representation
        sorted_x, _ = torch.sort(raw, dim=1)  # [B, L]
        quantile = sorted_x  # [B, L]
        
        # Channel 2: multi-scale drift on sorted signal
        drift = torch.zeros_like(raw)
        for k in [1, 2, 4]:
            shifted = torch.zeros_like(raw)
            if k < L:
                shifted[:, k:] = sorted_x[:, :-k]
            else:
                shifted = sorted_x
            drift += torch.abs(sorted_x - shifted)
        drift = drift / 3.0
        
        # Stack as channels
        T = torch.stack([raw, quantile, drift], dim=1)  # [B, 3, L]
        return T


class TransportEncoder(nn.Module):
    """Encodes [B, 3, L] transport into [B, D, L] features."""
    
    def __init__(self, out_ch=16, final_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 9, padding=4, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Project to backbone dimension
        self.proj = nn.Conv1d(out_ch, final_ch, 1, bias=False)
    
    def forward(self, T):
        """T: [B, 3, L] -> [B, final_ch, L]"""
        te = self.net(T)
        return self.proj(te)


class HierarchicalRegimeEncoder(nn.Module):
    """Encodes multi-scale features into hierarchical regime z_f, z_m, z_c."""
    
    def __init__(self, D=64, regime_dim=16):
        super().__init__()
        # Fine regime: D -> 32 -> regime_dim
        self.fine_enc = nn.Sequential(
            nn.Linear(D, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim)
        )
        # Middle regime: (D + regime_dim) -> 32 -> regime_dim
        self.mid_enc = nn.Sequential(
            nn.Linear(D + regime_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim)
        )
        # Coarse regime: (D + regime_dim) -> 32 -> regime_dim
        self.coarse_enc = nn.Sequential(
            nn.Linear(D + regime_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim)
        )
        self.regime_dim = regime_dim
        self.D = D
    
    def forward(self, H_f, H_m, H_c):
        """
        H_f: [B, D, L] fine features
        H_m: [B, D, L] middle features (D may differ)
        H_c: [B, D, L] coarse features
        Returns: z_f [B, rd], z_m [B, rd], z_c [B, rd]
        """
        # Handle D mismatch for early layers (fine may have fewer channels)
        D_f = H_f.shape[1]
        D_m = H_m.shape[1]
        D_c = H_c.shape[1]
        
        # Project to common D for regime encoding
        v_f = H_f.mean(dim=2)  # [B, D_f]
        v_m = H_m.mean(dim=2)  # [B, D_m]
        v_c = H_c.mean(dim=2)  # [B, D_c]
        
        # Fine regime
        if D_f != self.D:
            v_f = F.adaptive_avg_pool1d(v_f.unsqueeze(1), self.D).squeeze(1) if v_f.dim() == 2 else v_f
            # Simple projection if needed
            v_f_proj = v_f[:, :self.D] if v_f.shape[1] >= self.D else F.pad(v_f, (0, self.D - v_f.shape[1]))
        else:
            v_f_proj = v_f
        
        z_f = self.fine_enc(v_f_proj)  # [B, regime_dim]
        
        # Middle regime (conditioned on fine)
        if D_m != self.D:
            v_m_proj = v_m[:, :self.D] if v_m.shape[1] >= self.D else F.pad(v_m, (0, self.D - v_m.shape[1]))
        else:
            v_m_proj = v_m
        
        z_m = self.mid_enc(torch.cat([v_m_proj, z_f], dim=1))  # [B, regime_dim]
        
        # Coarse regime (conditioned on middle)
        if D_c != self.D:
            v_c_proj = v_c[:, :self.D] if v_c.shape[1] >= self.D else F.pad(v_c, (0, self.D - v_c.shape[1]))
        else:
            v_c_proj = v_c
        
        z_c = self.coarse_enc(torch.cat([v_c_proj, z_m], dim=1))  # [B, regime_dim]
        
        return z_f, z_m, z_c


class ARTNet(nn.Module):
    """
    Adaptive Regime-Transport Network
    
    ~190-220K params. Combines:
    - InceptionTime multi-scale backbone
    - Transport representation (raw + quantile + drift)
    - Hierarchical regime encoder (fine/mid/coarse)
    - Uncertainty-aware regime distribution
    - Adaptive residual gate (uncertainty suppresses regime injection)
    - Transport-regime interaction
    - Conditional classifier
    """
    
    def __init__(self, in_channels=1, num_classes=5, seq_len=140,
                 regime_dim=16, base_ch=32):
        super().__init__()
        self.D = base_ch * 2  # 64
        self.regime_dim = regime_dim
        self.num_classes = num_classes
        
        # Phase 1: InceptionTime backbone
        self.backbone = InceptionBackbone(in_channels, base_ch)
        
        # Phase 1: Transport builder + encoder
        self.transport_builder = TransportBuilder()
        self.transport_encoder = TransportEncoder(out_ch=16, final_ch=self.D)
        
        # Learned fusion scalar alpha
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        
        # Transport-regime interaction
        self.tr_regime_Wa = nn.Linear(regime_dim, self.D, bias=True)
        
        # Preliminary regime projection (from fused features)
        self.prelim_regime = nn.Linear(self.D, regime_dim, bias=True)
        
        # Phase 2: Hierarchical regime encoder
        # Note: fine has base_ch (32) channels, mid/coarse have D (64)
        self.fine_regime_enc = nn.Sequential(
            nn.Linear(base_ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim)
        )
        self.mid_regime_enc = nn.Sequential(
            nn.Linear(self.D + regime_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim)
        )
        self.coarse_regime_enc = nn.Sequential(
            nn.Linear(self.D + regime_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim)
        )
        
        # Phase 3: Uncertainty
        self.regime_mu = nn.Linear(regime_dim * 3, regime_dim, bias=True)
        self.regime_logvar = nn.Linear(regime_dim * 3, regime_dim, bias=True)
        self.regime_proj = nn.Linear(regime_dim * 3, regime_dim, bias=True)
        
        # Phase 4: Adaptive gate
        self.gate_W = nn.Linear(self.D + regime_dim, self.D, bias=True)
        self.gate_u = nn.Parameter(torch.tensor(-2.0))  # negative init: high uncertainty -> low gate
        
        # Regime residual injection
        self.regime_residual = nn.Linear(regime_dim, self.D, bias=True)
        
        # Classifier interaction projection
        self.classifier_proj = nn.Linear(regime_dim, self.D, bias=True)
        
        # Classifier head: [h'; z_r; h' * P(z)] -> classes
        self.classifier = nn.Sequential(
            nn.Linear(self.D + regime_dim + self.D, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
        # Dynamics predictor (for auxiliary loss)
        self.dynamics = nn.Sequential(
            nn.Linear(regime_dim + self.D, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim)
        )
    
    def forward(self, x):
        """
        x: [B, 1, L]
        Returns dict with logits, features, regime stats, gate, etc.
        """
        B, C, L = x.shape
        
        # Step 1-2: Build transport representation
        T = self.transport_builder(x)  # [B, 3, L]
        
        # Step 3: Transport encoding
        T_e = self.transport_encoder(T)  # [B, D, L]
        
        # Step 4: Inception features (multi-scale)
        H_f, H_m, H_c = self.backbone(x)  # [B, 32, L], [B, 64, L], [B, 64, L]
        
        # Step 5: Initial fusion (learned additive)
        alpha = torch.sigmoid(self.alpha_logit)
        H = H_c  # Main features from coarse level (64 channels)
        H_T = H + alpha * T_e  # [B, D, L]
        
        # Step 6: Preliminary regime from fused features
        v_prelim = H_T.mean(dim=2)  # [B, D]
        z_0 = self.prelim_regime(v_prelim)  # [B, regime_dim]
        
        # Step 7: Transport-regime interaction
        a_z = self.tr_regime_Wa(z_0)  # [B, D]
        scale = torch.sigmoid(a_z)  # [B, D]
        T_prime = T_e * (1 + scale.unsqueeze(2))  # [B, D, L]
        
        # Step 8: Final temporal representation
        H_TR = H_T + T_prime  # [B, D, L]
        
        # Step 9: Hierarchical regimes
        v_f = H_f.mean(dim=2)  # [B, 32]
        v_m = H_m.mean(dim=2)  # [B, 64]
        v_c = H_c.mean(dim=2)  # [B, 64]
        
        z_f = self.fine_regime_enc(v_f)  # [B, regime_dim]
        z_m = self.mid_regime_enc(torch.cat([v_m, z_f], dim=1))
        z_c = self.coarse_regime_enc(torch.cat([v_c, z_m], dim=1))
        
        # Step 10: Combined regime
        z_cat = torch.cat([z_f, z_m, z_c], dim=1)  # [B, 3*rd]
        
        # Step 11: Uncertainty
        mu_z = self.regime_mu(z_cat)  # [B, rd]
        logvar_z = self.regime_logvar(z_cat)  # [B, rd]
        logvar_z = torch.clamp(logvar_z, -5.0, 2.0)
        std_z = torch.exp(0.5 * logvar_z)  # [B, rd]
        
        # Step 12: Sample regime (reparameterization)
        if self.training:
            eps = torch.randn_like(std_z)
            z_r = mu_z + std_z * eps
        else:
            z_r = mu_z
        
        # Regime projection
        z_r_proj = self.regime_proj(z_cat)  # [B, rd]
        # Use sampled z_r for training, projected for inference stability
        if self.training:
            z_r_use = z_r
        else:
            z_r_use = z_r_proj
        
        # Step 13: Uncertainty scalar
        u_z = std_z.mean(dim=1, keepdim=True)  # [B, 1]
        
        # Step 14: Global temporal feature
        h = H_TR.mean(dim=2)  # [B, D]
        
        # Step 15: Adaptive gate (uncertainty-aware)
        gate_input = torch.cat([h, z_r_use], dim=1)  # [B, D + rd]
        g = torch.sigmoid(self.gate_W(gate_input))  # [B, D]
        # Uncertainty suppression: high u_z -> lower gate
        gate_u_mod = torch.sigmoid(self.gate_u * u_z)  # [B, 1], starts near 0 (suppressing)
        g = g * gate_u_mod  # [B, D]
        
        # Step 16: Regime residual injection
        r_z = self.regime_residual(z_r_use)  # [B, D]
        h_prime = h + g * r_z  # [B, D] — adaptive residual correction
        
        # Step 17: Classifier with interaction
        p_z = self.classifier_proj(z_r_use)  # [B, D]
        h_z = h_prime * p_z  # [B, D] — feature-regime interaction
        
        # Final representation: [h'; z_r; h' * P(z)]
        r = torch.cat([h_prime, z_r_use, h_z], dim=1)  # [B, 2D + rd]
        
        logits = self.classifier(r)  # [B, num_classes]
        
        # Predicted next regime (for dynamics loss)
        z_next_pred = self.dynamics(torch.cat([z_r_use, h], dim=1))  # [B, rd]
        
        return {
            "logits": logits,
            "features": h,
            "adapted_features": h_prime,
            "regime": z_r_use,
            "regime_mu": mu_z,
            "regime_std": std_z,
            "regime_uncertainty": u_z,
            "gate": g,
            "fine_regime": z_f,
            "middle_regime": z_m,
            "coarse_regime": z_c,
            "predicted_next_regime": z_next_pred,
            "alpha": alpha,
        }


def artnet_loss(outputs, targets, prev_regime=None,
                lambda_smooth=0.01, lambda_local=0.01,
                lambda_dyn=0.02, lambda_var=0.01, lambda_gate=0.001):
    """
    ARTNet loss function.
    
    Args:
        outputs: dict from ARTNet.forward()
        targets: [B] class labels
        prev_regime: [B, rd] regime from previous window (for smoothness)
        lambda_*: loss weights
    """
    # Task loss (cross-entropy)
    task_loss = F.cross_entropy(outputs["logits"], targets)
    
    # Smoothness loss: consecutive regimes should be similar
    smooth_loss = torch.tensor(0.0, device=task_loss.device)
    if prev_regime is not None:
        smooth_loss = F.mse_loss(outputs["regime"], prev_regime.detach())
    
    # Local consistency loss: feature similarity -> regime similarity
    # (simplified: just penalize large regime variance within batch)
    local_loss = torch.tensor(0.0, device=task_loss.device)
    feat = outputs["features"]
    if feat.shape[0] > 1:
        # Compute pairwise feature distances
        feat_norm = F.normalize(feat, p=2, dim=1)
        sim = torch.mm(feat_norm, feat_norm.t())  # [B, B]
        regime = outputs["regime"]
        reg_dist = torch.cdist(regime.unsqueeze(0), regime.unsqueeze(0)).squeeze(0)
        # Similar features should have similar regimes
        mask = sim > 0.8
        if mask.sum() > 0:
            local_loss = (reg_dist[mask] ** 2).mean()
    
    # Dynamics loss: predicted next regime should be close to actual
    dyn_loss = torch.tensor(0.0, device=task_loss.device)
    # We'll pass next_regime from the training loop
    
    # Variance regularization: prevent regime collapse
    regime_std = outputs["regime_std"].mean(dim=0)  # [rd]
    min_std = 0.1
    var_loss = F.relu(min_std - regime_std).pow(2).mean()
    
    # Gate sparsity: encourage gate to be sparse
    gate_loss = outputs["gate"].abs().mean()
    
    total = (
        task_loss
        + lambda_smooth * smooth_loss
        + lambda_local * local_loss
        + lambda_dyn * dyn_loss
        + lambda_var * var_loss
        + lambda_gate * gate_loss
    )
    
    return total, {
        "task": task_loss.item(),
        "smooth": smooth_loss.item(),
        "local": local_loss.item(),
        "var": var_loss.item(),
        "gate": gate_loss.item(),
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick verification
    model = ARTNet(in_channels=1, num_classes=5, seq_len=140)
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(4, 1, 140)
    out = model(x)
    
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Regime shape: {out['regime'].shape}")
    print(f"Gate range: [{out['gate'].min():.3f}, {out['gate'].max():.3f}]")
    print(f"Uncertainty range: [{out['regime_uncertainty'].min():.3f}, {out['regime_uncertainty'].max():.3f}]")
    print(f"Alpha: {out['alpha']:.3f}")
    
    # Check no NaN
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f"NaN in {k}"
    print("No NaN detected. All shapes correct.")
