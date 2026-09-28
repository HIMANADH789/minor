"""
CRMN — Continuous Regime Manifold Network

Architecture:
  1. Base feature extractor F_θ: multi-scale 1D CNN → feature vector h_t
  2. Regime encoder E_φ: h_t → regime coordinates z_t in R^d
  3. Multi-scale regime structure: z^(1) fine, z^(2) mid, z^(3) coarse
  4. Regime-conditioned MoE predictor: G_ψ(h_t, z_t) via prototype interpolation
  5. Regime transition model: z_{t+1} = f_ξ(z_t, h_t)
  6. Joint loss: task + smooth + feat_sim + prototype + dynamics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Base Feature Extractor: Multi-scale 1D CNN
# ============================================================
class MultiScaleConv(nn.Module):
    """Single multi-scale conv block with residual."""
    def __init__(self, in_ch, out_ch, kernels=(5, 11, 21)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False) for k in kernels
        ])
        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
        )
        out_total = out_ch * len(kernels) + out_ch
        self.norm = nn.BatchNorm1d(out_total)
        self.shortcut = nn.Conv1d(in_ch, out_total, 1, bias=False) if in_ch != out_total else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        branch_outs = [b(x) for b in self.branches]
        pool_out = self.pool(x)
        cat = torch.cat(branch_outs + [pool_out], dim=1)
        return self.act(self.norm(cat) + self.shortcut(x))


class FeatureExtractor(nn.Module):
    """Multi-scale CNN: raw signal → feature vector h_t ∈ R^{feat_dim}."""
    def __init__(self, in_channels=1, base_ch=32, n_blocks=4, feat_dim=128):
        super().__init__()
        ch = in_channels
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(MultiScaleConv(ch, base_ch, kernels=(5, 11, 21)))
            ch = base_ch * 4  # 3 branches + pool
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x):
        """x: (B, C, L) → h: (B, feat_dim)"""
        for blk in self.blocks:
            x = blk(x)
        x = self.gap(x).squeeze(-1)  # (B, ch)
        x = self.proj(x)              # (B, feat_dim)
        return self.norm(x)


# ============================================================
# Regime Encoder: h_t → z_t in R^{regime_dim}
# ============================================================
class RegimeEncoder(nn.Module):
    """Maps feature vector to regime coordinates."""
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, regime_dim * 2),
            nn.GELU(),
            nn.Linear(regime_dim * 2, regime_dim),
        )
        self.regime_dim = regime_dim

    def forward(self, h):
        """h: (B, feat_dim) → z: (B, regime_dim)"""
        return self.net(h)


# ============================================================
# Multi-Scale Regime Structure
# ============================================================
class MultiScaleRegimeEncoder(nn.Module):
    """Three-scale regime: fine (direct), mid (reduced-dim), coarse (heavily compressed).
    Each scale uses a different non-linear transform of the feature vector
    to simulate capturing information at different granularity levels."""
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        # Fine scale: direct encoding — captures raw feature patterns
        self.fine_encoder = RegimeEncoder(feat_dim, regime_dim)
        # Mid scale: features passed through a learned compression, then encode
        self.mid_compress = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, feat_dim),
        )
        self.mid_encoder = RegimeEncoder(feat_dim, regime_dim)
        # Coarse scale: heavy compression → reconstruction → encode
        self.coarse_compress = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, feat_dim),
        )
        self.coarse_encoder = RegimeEncoder(feat_dim, regime_dim)
        self.regime_dim = regime_dim

    def forward(self, h):
        """h: (B, feat_dim) → (z_fine, z_mid, z_coarse) each (B, regime_dim)"""
        z_fine = self.fine_encoder(h)
        z_mid = self.mid_encoder(self.mid_compress(h) + h)   # residual compression
        z_coarse = self.coarse_encoder(self.coarse_compress(h) + h)
        return z_fine, z_mid, z_coarse


# ============================================================
# Regime-Conditioned MoE Predictor
# ============================================================
class RegimeMoEPredictor(nn.Module):
    """
    Mixture-of-experts conditioned on regime coordinates.
    K prototypes define expert centers; each prototype has its own classifier head.
    Final prediction is an attention-weighted combination based on regime proximity.
    """
    def __init__(self, feat_dim=128, regime_dim=32, num_classes=5, K=8):
        super().__init__()
        self.K = K
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.regime_dim = regime_dim

        # Learnable prototype regimes
        self.prototypes = nn.Parameter(torch.randn(K, regime_dim) * 0.1)

        # Expert heads: each is a small MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.GELU(),
                nn.Linear(64, num_classes)
            ) for _ in range(K)
        ])

        # Regime-to-attention weights
        self.attn_proj = nn.Linear(regime_dim, K)

    def forward(self, h, z):
        """
        h: (B, feat_dim)
        z: (B, regime_dim) — combined regime from multi-scale
        logits: (B, num_classes)
        """
        B = h.shape[0]

        # Compute attention weights: how close is z to each prototype
        # prototypes: (K, regime_dim), z: (B, regime_dim)
        dists = torch.cdist(z.unsqueeze(1), self.prototypes.unsqueeze(0))  # (B, 1, K)
        dists = dists.squeeze(1)  # (B, K)
        attn = F.softmax(-dists * 2.0, dim=-1)  # (B, K) — closer prototypes get more weight

        # Compute expert outputs
        expert_outs = []
        for k in range(self.K):
            expert_outs.append(self.experts[k](h))  # (B, num_classes)
        expert_stack = torch.stack(expert_outs, dim=1)  # (B, K, num_classes)

        # Weighted combination
        attn_expanded = attn.unsqueeze(-1)  # (B, K, 1)
        logits = (expert_stack * attn_expanded).sum(dim=1)  # (B, num_classes)

        return logits, attn  # return attn for analysis


# ============================================================
# Regime Transition Model
# ============================================================
class RegimeTransition(nn.Module):
    """Learns dynamics: z_{t+1} = f(z_t, h_t)."""
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + regime_dim, regime_dim * 2),
            nn.GELU(),
            nn.Linear(regime_dim * 2, regime_dim),
        )

    def forward(self, z, h):
        """z: (B, regime_dim), h: (B, feat_dim) → z_next: (B, regime_dim)"""
        return self.net(torch.cat([h, z], dim=-1))


# ============================================================
# CRMN: Full Model
# ============================================================
class CRMN(nn.Module):
    """
    Continuous Regime Manifold Network.

    Forward pass:
      1. h = FeatureExtractor(x)           — (B, feat_dim)
      2. (z1, z2, z3) = MultiScaleRegime(h) — three regime coords
      3. z = cat([z1, z2, z3])             — combined regime
      4. z_combined = Linear(3*regime_dim → regime_dim)
      5. logits, attn = MoEPredictor(h, z_combined)
      6. z_next = Transition(z_combined, h) — predicted next regime

    Total parameters scale: ~500K-1M (comparable to InceptionTime).
    """
    def __init__(self, in_channels=1, num_classes=5, feat_dim=128,
                 regime_dim=32, n_experts=8, n_blocks=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.regime_dim = regime_dim
        self.num_classes = num_classes

        self.extractor = FeatureExtractor(in_channels, base_ch=32, n_blocks=n_blocks, feat_dim=feat_dim)
        self.ms_regime = MultiScaleRegimeEncoder(feat_dim, regime_dim)
        self.regime_merge = nn.Linear(regime_dim * 3, regime_dim)
        self.regime_merge_norm = nn.LayerNorm(regime_dim)
        self.predictor = RegimeMoEPredictor(feat_dim, regime_dim, num_classes, K=n_experts)
        self.transition = RegimeTransition(feat_dim, regime_dim)

    def forward(self, x):
        """
        x: (B, C, L)
        Returns: logits, info_dict (for computing regime losses)
        """
        B = x.shape[0]

        # 1. Extract features
        h = self.extractor(x)  # (B, feat_dim)

        # 2. Multi-scale regime encoding
        z_fine, z_mid, z_coarse = self.ms_regime(h)

        # 3. Combine regimes
        z_cat = torch.cat([z_fine, z_mid, z_coarse], dim=-1)  # (B, 3*regime_dim)
        z = self.regime_merge_norm(self.regime_merge(z_cat))   # (B, regime_dim)

        # 4. Regime-conditioned prediction
        logits, attn_weights = self.predictor(h, z)

        # 5. Transition prediction (for dynamics loss)
        z_next_pred = self.transition(z, h)

        info = {
            "h": h,
            "z": z,
            "z_fine": z_fine,
            "z_mid": z_mid,
            "z_coarse": z_coarse,
            "z_next_pred": z_next_pred,
            "attn_weights": attn_weights,
            "prototypes": self.predictor.prototypes,
        }
        return logits, info


# ============================================================
# CRMN Loss
# ============================================================
class CRMNLoss(nn.Module):
    """
    Joint loss: L = L_task + λ1*L_smooth + λ2*L_feat + λ3*L_proto + λ4*L_dyn

    L_task:   cross-entropy (or focal) for classification
    L_smooth: neighboring z's should be close (unused in single-window mode)
    L_feat:   similar features → similar regimes (in-batch pairs)
    L_proto:  regime points cluster around prototypes; prototypes separated
    L_dyn:    transition model consistency
    """
    def __init__(self, num_classes=5, lambda_smooth=0.01, lambda_feat=0.01,
                 lambda_proto=0.01, lambda_dyn=0.005, focal_gamma=0,
                 proto_sep=2.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_feat = lambda_feat
        self.lambda_proto = lambda_proto
        self.lambda_dyn = lambda_dyn
        self.focal_gamma = focal_gamma
        self.proto_sep = proto_sep
        self.num_classes = num_classes

    def focal_ce(self, logits, targets):
        if self.focal_gamma == 0:
            return F.cross_entropy(logits, targets)
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.focal_gamma * ce).mean()

    def forward(self, logits, info, targets):
        """
        logits: (B, C)
        info: dict from CRMN forward
        targets: (B,)
        """
        B = logits.shape[0]

        # Task loss
        L_task = self.focal_ce(logits, targets)

        # Feature similarity loss: in-batch similar features → similar regimes
        z = info["z"]  # (B, regime_dim)
        h = info["h"]  # (B, feat_dim)
        with torch.no_grad():
            feat_dists = torch.cdist(h, h)  # (B, B)
            k = min(4, B - 1)
            _, topk_idx = feat_dists.topk(k + 1, dim=-1, largest=False)
            topk_idx = topk_idx[:, 1:]  # exclude self — shape (B, k), values in [0, B-1]
        z_expanded = z.unsqueeze(1).expand(-1, k, -1)  # (B, k, regime_dim)
        z_neighbors = z[topk_idx]  # advanced indexing: (B, k, regime_dim)
        L_feat = F.mse_loss(z_expanded, z_neighbors)

        # Prototype loss: regimes should be close to their nearest prototype
        prototypes = info["prototypes"]  # (K, regime_dim)
        proto_dists = torch.cdist(z, prototypes)  # (B, K)
        min_proto_dist = proto_dists.min(dim=1)[0]  # (B,)
        L_proto_cluster = min_proto_dist.mean()

        # Prototype separation: different prototypes should be far apart
        proto_pair_dists = torch.cdist(prototypes, prototypes)  # (K, K)
        # Mask diagonal
        mask = ~torch.eye(prototypes.shape[0], dtype=torch.bool, device=prototypes.device)
        proto_min_dist = proto_pair_dists[mask].min()
        L_proto_sep = F.relu(self.proto_sep - proto_min_dist)

        L_proto = L_proto_cluster + 0.1 * L_proto_sep

        # Dynamics consistency: z_next_pred should be close to shifted z
        # (In single-sample mode, this is approximate)
        z_next_pred = info["z_next_pred"]  # (B, regime_dim)
        # Use h as proxy for next-z (shifted features); approximate
        # In a real sequential model, we'd use actual next-step z
        # Here we use self-consistency: z_next should be smooth w.r.t. current z
        L_dyn = F.mse_loss(z_next_pred, z.detach())  # predict next regime ≈ current regime (smoothness)

        # Total
        L_total = (L_task
                   + self.lambda_feat * L_feat
                   + self.lambda_proto * L_proto
                   + self.lambda_dyn * L_dyn)

        return L_total, {
            "task": L_task.item(),
            "feat": L_feat.item(),
            "proto": L_proto.item(),
            "dyn": L_dyn.item(),
        }
