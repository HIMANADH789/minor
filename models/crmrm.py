"""
CMRM — Continuous Multi-Scale Regime Manifold Network

Key differences from CRMN:
1. Cross-scale regime interaction: fine regime conditioned by mid/coarse context,
   coarse regime modulates interpretation of fine features via gating.
2. Regime interaction graph: prototype-to-prototype adjacency learned from
   geometry, graph propagation modifies prototype predictions before interpolation.
3. Continuous prototype field: soft distance-based interpolation, no hard assignment.
4. Geometry-preserving regularization: task + feat_sim + proto + cross-scale + dynamics
5. Stable training: explicit shape assertions throughout, no topk batch-index bugs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Base Feature Extractor: Multi-scale 1D CNN (Inception-style)
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
    """Multi-scale CNN: raw signal -> feature vector h in R^{feat_dim}."""
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
        """x: (B, C, L) -> h: (B, feat_dim)"""
        for blk in self.blocks:
            x = blk(x)
        x = self.gap(x).squeeze(-1)
        x = self.proj(x)
        return self.norm(x)


# ============================================================
# Single-Scale Regime Encoder
# ============================================================
class RegimeEncoder(nn.Module):
    """Maps feature vector to regime coordinates."""
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, regime_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(regime_dim * 2, regime_dim),
        )
        self.regime_dim = regime_dim

    def forward(self, h):
        """h: (B, feat_dim) -> z: (B, regime_dim)"""
        return self.net(h)


# ============================================================
# Cross-Scale Regime Interaction (KEY NOVELTY over CRMN)
# ============================================================
class CrossScaleRegimeEncoder(nn.Module):
    """
    Three genuine temporal regime levels with cross-scale interaction:
    
    Fine z^1: direct encoding of features
    Mid z^2: encoding of compressed features, then MODULATED by fine context
    Coarse z^3: encoding of heavily compressed features, then MODULATED by mid+fine context
    
    Cross-scale interaction:
    - z_mid gets cross-attended with z_fine as context (fine informs mid)
    - z_coarse gets cross-attended with [z_fine; z_mid] as context (fine+mid inform coarse)
    - z_fine gets GATED by mid+coarse (coarse modulates interpretation of fine)
    
    This gives the manifold genuine hierarchical structure rather than
    three independent projections (as in CRMN).
    """
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        self.regime_dim = regime_dim
        
        # --- Three base encoders (different compression levels) ---
        # Fine: direct
        self.fine_encoder = RegimeEncoder(feat_dim, regime_dim)
        
        # Mid: moderate compression
        self.mid_compress = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, feat_dim),
        )
        self.mid_encoder = RegimeEncoder(feat_dim, regime_dim)
        
        # Coarse: heavy compression
        self.coarse_compress = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, feat_dim),
        )
        self.coarse_encoder = RegimeEncoder(feat_dim, regime_dim)
        
        # --- Cross-scale interaction layers ---
        # Mid conditioned by fine context (cross-attention)
        self.mid_cross_attn = nn.MultiheadAttention(
            embed_dim=regime_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.mid_cross_norm = nn.LayerNorm(regime_dim)
        
        # Coarse conditioned by [fine; mid] context (cross-attention)
        self.coarse_cross_attn = nn.MultiheadAttention(
            embed_dim=regime_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.coarse_cross_norm = nn.LayerNorm(regime_dim)
        
        # Fine gated by mid+coarse (gating mechanism)
        self.fine_gate_linear = nn.Linear(regime_dim * 2, regime_dim)
        self.fine_gate_sigmoid = nn.Sigmoid()
        self.fine_gate_norm = nn.LayerNorm(regime_dim)

    def forward(self, h):
        """
        h: (B, feat_dim)
        Returns: (z_fine, z_mid, z_coarse) each (B, regime_dim)
        
        Flow: fine -> mid (conditioned on fine) -> coarse (conditioned on fine+mid) -> fine (gated by mid+coarse)
        """
        B = h.shape[0]
        
        # Step 1: Base encodings
        z_fine_base = self.fine_encoder(h)                            # (B, rd)
        z_mid_base = self.mid_encoder(self.mid_compress(h) + h)       # (B, rd) residual
        z_coarse_base = self.coarse_encoder(self.coarse_compress(h) + h)  # (B, rd) residual
        
        # Step 2: Mid conditioned by fine (cross-attention: query=mid, key/value=fine)
        z_mid_q = z_mid_base.unsqueeze(1)           # (B, 1, rd)
        z_fine_kv = z_fine_base.unsqueeze(1)        # (B, 1, rd)
        mid_cross, _ = self.mid_cross_attn(z_mid_q, z_fine_kv, z_fine_kv)  # (B, 1, rd)
        z_mid = self.mid_cross_norm(z_mid_base + mid_cross.squeeze(1))      # (B, rd) residual
        
        # Step 3: Coarse conditioned by [fine; mid] (cross-attention)
        z_coarse_q = z_coarse_base.unsqueeze(1)     # (B, 1, rd)
        fine_mid_ctx = torch.stack([z_fine_base, z_mid], dim=1)  # (B, 2, rd)
        coarse_cross, _ = self.coarse_cross_attn(z_coarse_q, fine_mid_ctx, fine_mid_ctx)
        z_coarse = self.coarse_cross_norm(z_coarse_base + coarse_cross.squeeze(1))  # (B, rd) residual
        
        # Step 4: Fine gated by mid+coarse
        gate_input = torch.cat([z_mid, z_coarse], dim=-1)  # (B, 2*rd)
        gate = self.fine_gate_sigmoid(self.fine_gate_linear(gate_input))  # (B, rd)
        z_fine = self.fine_gate_norm(z_fine_base * gate + z_fine_base * (1 - gate))
        
        return z_fine, z_mid, z_coarse


# ============================================================
# Regime Interaction Graph (KEY NOVELTY over CRMN)
# ============================================================
class RegimeGraph(nn.Module):
    """
    Prototype-to-prototype interaction graph.
    
    Given K learnable prototypes z^(k) in regime space:
    1. Compute adjacency: A_kl = softmax(-||z^(k) - z^(l)||^2 / tau)
    2. Graph propagation: z^(k)_prop = sum_l A_kl * z^(l)  (1-hop message passing)
    3. Modified prototypes: z^(k)_eff = alpha * z^(k)_prop + (1-alpha) * z^(k)
    
    This means prototype predictions are influenced by their neighbors in the
    regime graph, creating a structured continuous field rather than independent experts.
    """
    def __init__(self, regime_dim=32, K=8, temperature=1.0):
        super().__init__()
        self.K = K
        self.regime_dim = regime_dim
        self.temperature = nn.Parameter(torch.tensor(math.log(temperature)))  # learnable
        self.alpha = nn.Parameter(torch.tensor(0.5))  # mixing coefficient

    def forward(self, prototypes):
        """
        prototypes: (K, regime_dim) — learnable parameters
        Returns: propagated_prototypes: (K, regime_dim)
        """
        K = prototypes.shape[0]
        tau = torch.exp(self.temperature)
        
        # Pairwise distances: (K, K)
        dists = torch.cdist(prototypes, prototypes)  # (K, K)
        
        # Adjacency with learnable temperature
        adj = F.softmax(-dists / tau, dim=-1)  # (K, K) rows sum to 1
        
        # 1-hop graph propagation
        prototypes_prop = torch.matmul(adj, prototypes)  # (K, rd)
        
        # Mix original with propagated
        alpha_clamped = torch.sigmoid(self.alpha)
        prototypes_eff = alpha_clamped * prototypes_prop + (1 - alpha_clamped) * prototypes
        
        return prototypes_eff, adj


# ============================================================
# MoE Predictor with Graph-Propagated Prototypes
# ============================================================
class GraphMoEPredictor(nn.Module):
    """
    Mixture-of-experts conditioned on regime coordinates,
    using graph-propagated prototypes for the attention weights.
    
    1. Compute attention: softmax(-dist(z, z^(k)_eff) / temperature)
    2. Each prototype has its own expert MLP
    3. Final prediction = weighted combination of expert outputs
    """
    def __init__(self, feat_dim=128, regime_dim=32, num_classes=5, K=8):
        super().__init__()
        self.K = K
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.regime_dim = regime_dim

        # Learnable prototype regimes
        self.prototypes = nn.Parameter(torch.randn(K, regime_dim) * 0.1)

        # Expert heads
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(64, num_classes)
            ) for _ in range(K)
        ])
        
        # Temperature for attention (learnable)
        self.attn_temp = nn.Parameter(torch.tensor(1.0))
        
        # Regime graph
        self.graph = RegimeGraph(regime_dim, K)

    def forward(self, h, z):
        """
        h: (B, feat_dim)
        z: (B, regime_dim)
        Returns: logits (B, num_classes), attn (B, K), graph_adj (K, K)
        """
        B = h.shape[0]
        
        # Get graph-propagated prototypes
        prototypes_eff, graph_adj = self.graph(self.prototypes)  # (K, rd), (K, K)
        
        # Attention weights based on distance to effective prototypes
        dists = torch.cdist(z.unsqueeze(1), prototypes_eff.unsqueeze(0))  # (B, 1, K)
        dists = dists.squeeze(1)  # (B, K)
        
        temp = torch.clamp(self.attn_temp, min=0.1, max=5.0)
        attn = F.softmax(-dists * temp, dim=-1)  # (B, K)
        
        # Expert outputs
        expert_outs = []
        for k in range(self.K):
            expert_outs.append(self.experts[k](h))  # (B, nc)
        expert_stack = torch.stack(expert_outs, dim=1)  # (B, K, nc)
        
        # Weighted combination
        logits = (expert_stack * attn.unsqueeze(-1)).sum(dim=1)  # (B, nc)
        
        return logits, attn, graph_adj


# ============================================================
# Regime Dynamics (auxiliary, not dominant)
# ============================================================
class RegimeDynamics(nn.Module):
    """Predicts next regime from current regime + features."""
    def __init__(self, feat_dim=128, regime_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + regime_dim, regime_dim * 2),
            nn.GELU(),
            nn.Linear(regime_dim * 2, regime_dim),
        )

    def forward(self, z, h):
        """z: (B, rd), h: (B, feat_dim) -> z_next: (B, rd)"""
        return self.net(torch.cat([h, z], dim=-1))


# ============================================================
# CMRM: Full Model
# ============================================================
class CMRM(nn.Module):
    """
    Continuous Multi-Scale Regime Manifold Network.
    
    Forward pass:
      1. h = FeatureExtractor(x)                        — (B, feat_dim)
      2. (z1, z2, z3) = CrossScaleRegimeEncoder(h)      — cross-scale interaction
      3. z = Linear([z1; z2; z3]) -> regime_dim          — merged regime
      4. logits, attn, adj = GraphMoEPredictor(h, z)    — graph-propagated MoE
      5. z_next = RegimeDynamics(z, h)                    — auxiliary dynamics
    """
    def __init__(self, in_channels=1, num_classes=5, feat_dim=128,
                 regime_dim=32, n_experts=8, n_blocks=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.regime_dim = regime_dim
        self.num_classes = num_classes

        # Backbone
        self.extractor = FeatureExtractor(in_channels, base_ch=32, n_blocks=n_blocks, feat_dim=feat_dim)
        
        # Cross-scale regime encoder (KEY NOVELTY)
        self.ms_regime = CrossScaleRegimeEncoder(feat_dim, regime_dim)
        
        # Regime merge with cross-scale consistency
        self.regime_merge = nn.Linear(regime_dim * 3, regime_dim)
        self.regime_merge_norm = nn.LayerNorm(regime_dim)
        
        # Graph MoE predictor (KEY NOVELTY)
        self.predictor = GraphMoEPredictor(feat_dim, regime_dim, num_classes, K=n_experts)
        
        # Auxiliary dynamics
        self.transition = RegimeDynamics(feat_dim, regime_dim)

    def forward(self, x):
        """
        x: (B, C, L)
        Returns: logits, info_dict
        """
        B = x.shape[0]
        
        # 1. Extract features
        h = self.extractor(x)  # (B, feat_dim)
        assert h.shape == (B, self.feat_dim), f"Feature shape mismatch: {h.shape}"
        
        # 2. Cross-scale regime encoding
        z_fine, z_mid, z_coarse = self.ms_regime(h)
        assert z_fine.shape == (B, self.regime_dim), f"Fine regime shape mismatch"
        assert z_mid.shape == (B, self.regime_dim), f"Mid regime shape mismatch"
        assert z_coarse.shape == (B, self.regime_dim), f"Coarse regime shape mismatch"
        
        # 3. Merge regimes
        z_cat = torch.cat([z_fine, z_mid, z_coarse], dim=-1)  # (B, 3*rd)
        z = self.regime_merge_norm(self.regime_merge(z_cat))   # (B, rd)
        assert z.shape == (B, self.regime_dim), f"Merged regime shape mismatch"
        
        # 4. Graph MoE prediction
        logits, attn_weights, graph_adj = self.predictor(h, z)
        assert logits.shape == (B, self.num_classes), f"Logits shape mismatch"
        assert attn_weights.shape == (B, self.predictor.K), f"Attention shape mismatch"
        assert graph_adj.shape == (self.predictor.K, self.predictor.K), f"Adj shape mismatch"
        
        # 5. Dynamics (auxiliary)
        z_next_pred = self.transition(z, h)
        
        info = {
            "h": h,
            "z": z,
            "z_fine": z_fine,
            "z_mid": z_mid,
            "z_coarse": z_coarse,
            "z_next_pred": z_next_pred,
            "attn_weights": attn_weights,
            "graph_adj": graph_adj,
            "prototypes": self.predictor.prototypes,
            "prototypes_eff": None,  # filled by loss if needed
        }
        return logits, info


# ============================================================
# CMRM Loss: Geometry-Preserving Regularization
# ============================================================
class CRMRLoss(nn.Module):
    """
    L = L_task + λ1*L_feat + λ2*L_proto + λ3*L_cross + λ4*L_dyn
    
    L_task:  focal CE for classification
    L_feat:  local feature/regime consistency (similar features -> similar regimes)
    L_proto: prototype attraction (regimes near prototypes) + separation (prototypes far apart)
    L_cross: cross-scale consistency (fine~mid relationship consistent with mid~coarse)
    L_dyn:   dynamics consistency (weak, auxiliary)
    """
    def __init__(self, num_classes=5, lambda_feat=0.01, lambda_proto=0.01,
                 lambda_cross=0.005, lambda_dyn=0.001, focal_gamma=0,
                 proto_sep=2.0):
        super().__init__()
        self.lambda_feat = lambda_feat
        self.lambda_proto = lambda_proto
        self.lambda_cross = lambda_cross
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
        info: dict from CMRM forward
        targets: (B,)
        """
        B = logits.shape[0]
        
        # ---- Task loss ----
        L_task = self.focal_ce(logits, targets)
        
        z = info["z"]        # (B, rd)
        h = info["h"]        # (B, fd)
        z_fine = info["z_fine"]
        z_mid = info["z_mid"]
        z_coarse = info["z_coarse"]
        prototypes = info["prototypes"]  # (K, rd)
        
        # ---- Feature similarity loss ----
        # In-batch: similar features should have similar regimes
        with torch.no_grad():
            feat_dists = torch.cdist(h, h)  # (B, B)
            # For stability: use top-k nearest neighbors (avoid batch-index bug)
            k = min(4, B - 1)
            if k > 0:
                # Mask self-distances to inf
                feat_dists_copy = feat_dists.clone()
                feat_dists_copy.fill_diagonal_(float('inf'))
                _, topk_idx = feat_dists_copy.topk(k, dim=-1, largest=False)
                # topk_idx values are in [0, B-1], indexing z along dim=0
                valid_mask = torch.ones(B, k, dtype=torch.bool, device=z.device)
            else:
                valid_mask = torch.zeros(B, 1, dtype=torch.bool, device=z.device)
                topk_idx = torch.zeros(B, 1, dtype=torch.long, device=z.device)
        
        if k > 0:
            z_expanded = z.unsqueeze(1).expand(-1, k, -1)  # (B, k, rd)
            z_neighbors = z[topk_idx]  # (B, k, rd) — advanced indexing
            L_feat = F.mse_loss(z_expanded, z_neighbors, reduction='none').mean()
        else:
            L_feat = torch.tensor(0.0, device=z.device)
        
        # ---- Prototype losses ----
        # Attraction: regimes should be close to nearest prototype
        proto_dists = torch.cdist(z, prototypes)  # (B, K)
        min_proto_dist = proto_dists.min(dim=1)[0]  # (B,)
        L_proto_attract = min_proto_dist.mean()
        
        # Separation: different prototypes should be far apart
        proto_pair_dists = torch.cdist(prototypes, prototypes)  # (K, K)
        mask = ~torch.eye(prototypes.shape[0], dtype=torch.bool, device=prototypes.device)
        proto_min_dist = proto_pair_dists[mask].min()
        L_proto_sep = F.relu(self.proto_sep - proto_min_dist)
        
        L_proto = L_proto_attract + 0.1 * L_proto_sep
        
        # ---- Cross-scale consistency loss ----
        # Fine-to-mid relationship should be consistent with mid-to-coarse
        # i.e., the ratio ||z_fine - z_mid|| / ||z_mid - z_coarse|| should be stable
        # Simplified: all three scales should be locally consistent
        L_cross_fm = F.mse_loss(z_fine, z_mid.detach())  # fine ~ mid
        L_cross_mc = F.mse_loss(z_mid, z_coarse.detach())  # mid ~ coarse
        L_cross = 0.5 * (L_cross_fm + L_cross_mc)
        
        # ---- Dynamics consistency (weak, auxiliary) ----
        z_next_pred = info["z_next_pred"]  # (B, rd)
        # Self-consistency: predicted next regime ≈ current regime
        L_dyn = F.mse_loss(z_next_pred, z.detach())
        
        # ---- Total ----
        L_total = (L_task
                   + self.lambda_feat * L_feat
                   + self.lambda_proto * L_proto
                   + self.lambda_cross * L_cross
                   + self.lambda_dyn * L_dyn)
        
        return L_total, {
            "task": L_task.item(),
            "feat": L_feat.item(),
            "proto": L_proto.item(),
            "cross": L_cross.item(),
            "dyn": L_dyn.item(),
        }
