"""
TURS-AMR — Transport–Uncertainty Regime Network with Adaptive Multiscale Response Field

Central innovation: Multiscale response field used as internal sensor for
continuous regime transitions and scale selection, not as classifier features.

  X → R_multiscale → ΔR → ṽ → z → g_scale → R_adaptive

Response field participates in:
  1. Regime encoding (H, F_M, q, C) → (z, u)
  2. Regime velocity: v_t + g_v ⊙ P(q_t) → ṽ_t
  3. Adaptive scale selection: regime determines which scales matter
  4. Cross-scale response interaction: pairwise scale-group dot products
  5. Transport-regime fusion: same TURS principle

Variants (incremental, seed-42 comparable):
  rv   : reference (TURS-RV baseline)
  d    : AMR-D — richer response dynamics (R, ΔR, Δ²R)
  cs   : AMR-CS — cross-scale response interaction
  as   : AMR-AS — adaptive scale selection (regime-conditioned scale gating)
  rk   : AMR-RK — conditional kernel residual
  full : AMR-Full — combined best components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tursnet import (TURSNet, RegimeEncoder, TransportBuilder,
                             TransportEncoder, TURSLoss, InceptionBlock)


# ============================================================
# Adaptive Multiscale Response Field
# ============================================================
class MultiscaleResponseBank(nn.Module):
    """Compact ROCKET-inspired response bank with diverse scales.

    Each filter is fixed-random (frozen). Response statistics (PPV, mean,
    std, abs_mean, max_abs, energy) are computed per filter, then projected.

    Scale groups enable cross-scale and adaptive-scale mechanisms.
    """

    def __init__(self, in_ch=1,
                 # Scale configs: (kernel, dilation, n_filters)
                 scales=None,
                 seed=0):
        super().__init__()
        if scales is None:
            scales = [
                (7, 1, 4),   # scale 0: short-range, no dilation
                (9, 2, 4),   # scale 1: short-range, dilated
                (13, 1, 4),  # scale 2: medium-range
                (19, 2, 4),  # scale 3: medium-range, dilated
                (25, 1, 4),  # scale 4: long-range
                (35, 2, 4),  # scale 5: very-long-range, dilated
            ]
        self.scales = scales
        self.n_scales = len(scales)
        self.filters_per_scale = scales[0][2]
        self.n_total = sum(s[2] for s in scales)
        self.n_stat = 6  # PPV, mean, std, abs_mean, max_abs, energy
        self.raw_dim = self.n_total * self.n_stat  # descriptors per sample

        self.convs = nn.ModuleList()
        self.scale_indices = []  # which scale group each conv belongs to
        saved_seed = torch.initial_seed()
        torch.manual_seed(seed)
        for si, (k, d, nf) in enumerate(scales):
            for _ in range(nf):
                conv = nn.Conv1d(in_ch, 1, k, padding='same',
                                 dilation=d, bias=True)
                with torch.no_grad():
                    conv.weight.normal_(0.0, 1.0)
                    conv.bias.uniform_(-1.0, 1.0)
                for p in conv.parameters():
                    p.requires_grad = False
                self.convs.append(conv)
                self.scale_indices.append(si)
        torch.manual_seed(saved_seed)  # restore

    def forward(self, x):
        """x: [B, 1, L] → R: [B, n_total], R_raw: [B, n_total, L]"""
        B, _, L = x.shape
        responses = []
        for conv in self.convs:
            r = conv(x).squeeze(1)  # [B, L]
            responses.append(r)
        # R_raw: [B, n_total, L]
        R_raw = torch.stack(responses, dim=1)
        return R_raw

    def describe(self, R_raw):
        """Compute 6 statistics per filter. R_raw: [B, F, L] → [B, F * 6]"""
        ppv = (R_raw > 0).float().mean(dim=2)         # positive proportion
        mu = R_raw.mean(dim=2)                          # mean
        std = R_raw.std(dim=2) + 1e-6                   # std
        abs_mu = R_raw.abs().mean(dim=2)                # abs mean
        max_abs = R_raw.abs().max(dim=2)[0]             # max abs
        energy = (R_raw ** 2).mean(dim=2)               # energy
        desc = torch.cat([ppv, mu, std, abs_mu, max_abs, energy], dim=1)
        return desc  # [B, F * 6]


# ============================================================
# Cross-Scale Interaction Module
# ============================================================
class CrossScaleInteraction(nn.Module):
    """Pairwise dot-product interactions between scale-group projections."""

    def __init__(self, n_scales, proj_dim=4, out_dim=16):
        super().__init__()
        self.n_scales = n_scales
        self.n_pairs = n_scales * (n_scales - 1) // 2
        self.proj = nn.Linear(proj_dim, proj_dim, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(self.n_pairs * proj_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, scale_features):
        """scale_features: list of [B, proj_dim] for each scale group
           → [B, out_dim]"""
        B = scale_features[0].shape[0]
        pairs = []
        for i in range(self.n_scales):
            for j in range(i + 1, self.n_scales):
                pi = self.proj(scale_features[i])
                pj = self.proj(scale_features[j])
                pairs.append(pi * pj)  # [B, proj_dim]
        cat = torch.cat(pairs, dim=1)  # [B, n_pairs * proj_dim]
        return self.out_proj(cat)  # [B, out_dim]


# ============================================================
# Adaptive Scale Selector
# ============================================================
class AdaptiveScaleSelector(nn.Module):
    """Regime-conditioned softmax over scale groups.

    Learns which response scales matter given the current regime.
    """

    def __init__(self, regime_dim, n_scales):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(regime_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_scales),
        )

    def forward(self, z):
        """z: [B, regime_dim] → weights: [B, n_scales]"""
        return F.softmax(self.net(z), dim=1)


# ============================================================
# Response Dynamics Encoder
# ============================================================
class ResponseDynamicsEncoder(nn.Module):
    """Encodes R, ΔR, Δ²R into a compact dynamics embedding q.

    Uses fixed-statistics descriptors projected to a compact dim.
    """

    def __init__(self, raw_dim, n_pairs, cs_proj_dim, out_dim):
        super().__init__()
        # Input: descriptors of R, ΔR, Δ²R + cross-scale structure
        in_dim = raw_dim * 3 + n_pairs * cs_proj_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, desc_R, desc_dR, desc_ddR, cs_struct):
        """All inputs: [B, *]"""
        cat = torch.cat([desc_R, desc_dR, desc_ddR, cs_struct], dim=1)
        return self.net(cat)


# ============================================================
# TURS-AMR Model
# ============================================================
class TURSAMR(nn.Module):
    """Transport–Uncertainty Regime Network with Adaptive Multiscale Response.

    variant='rv'   : TURS-RV reference (no AMR components)
    variant='d'    : AMR-D — richer response dynamics
    variant='cs'   : AMR-CS — cross-scale interaction
    variant='as'   : AMR-AS — adaptive scale selection
    variant='rk'   : AMR-RK — conditional kernel residual
    variant='full' : AMR-Full — combined selected mechanisms
    """

    def __init__(self, in_channels=1, num_classes=5, regime_dim=16,
                 variant='full', dropout=0.1, rocket_dim=24,
                 cs_proj_dim=4, cs_out_dim=16,
                 **kw):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.regime_dim = regime_dim
        self.rocket_dim = rocket_dim

        base_ch = 32
        D = base_ch * 2  # 64

        # --- Transport (same as TURS) ---
        self.transport_builder = TransportBuilder()
        self.transport_encoder = TransportEncoder(out_ch=16, final_ch=D)
        self.C_T = D

        # --- Inception backbone (same as TURS) ---
        self.proj_in = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True),
        )
        self.block1 = InceptionBlock(base_ch, base_ch)
        self.block2 = InceptionBlock(base_ch, base_ch)
        self.block3 = InceptionBlock(base_ch, D)
        self.block4 = InceptionBlock(D, D)

        # --- Multiscale Response Bank ---
        self.response_bank = MultiscaleResponseBank(in_ch=in_channels, seed=42)
        n_filters = self.response_bank.n_total
        raw_dim = self.response_bank.raw_dim  # n_filters * 6
        n_scales = self.response_bank.n_scales

        # Response descriptor projection
        self.response_proj = nn.Sequential(
            nn.Linear(raw_dim, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, rocket_dim),
        )

        # Scale-group projections (for cross-scale / dynamics)
        self.scale_proj = nn.ModuleList()
        if variant in ('cs', 'd', 'full'):
            self.scale_proj = nn.ModuleList([
                nn.Linear(self.response_bank.filters_per_scale * 6, cs_proj_dim)
                for _ in range(n_scales)
            ])

        # --- Variant-specific AMR modules ---

        # D: Response dynamics encoder
        if variant in ('d', 'full'):
            n_pairs = n_scales * (n_scales - 1) // 2
            self.dynamics_enc = ResponseDynamicsEncoder(
                raw_dim, n_pairs, cs_proj_dim, rocket_dim)

        # CS: Cross-scale interaction
        if variant in ('cs', 'full'):
            self.cross_scale = CrossScaleInteraction(
                n_scales, cs_proj_dim, cs_out_dim)
            self.cs_merge = nn.Linear(cs_out_dim, rocket_dim, bias=False)

        # AS: Adaptive scale selector
        if variant in ('as', 'full'):
            self.scale_selector = AdaptiveScaleSelector(regime_dim, n_scales)
            self.adaptive_pool = nn.ModuleList([
                nn.Linear(self.response_bank.filters_per_scale * 6, rocket_dim)
                for _ in range(n_scales)
            ])

        # RK: Conditional kernel residual (only in rk variant)
        if variant == 'rk':
            self.kernel_residual_net = nn.Sequential(
                nn.Linear(regime_dim, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, n_filters),
            )

        # --- Regime encoder (sees [H, F_M, q, C]) ---
        regime_in_dim = D + rocket_dim  # H + response
        if variant in ('d', 'full'):
            regime_in_dim += rocket_dim  # + dynamics q
        if variant in ('cs', 'full'):
            regime_in_dim += cs_out_dim  # + cross-scale structure
        self.regime_main = RegimeEncoder(regime_in_dim, regime_dim)

        # --- Velocity (base + rocket-informed) ---
        self.vel_linear = nn.Linear(regime_dim, regime_dim)
        self.vel_q = nn.Linear(rocket_dim, regime_dim)
        self.vel_gate = nn.Sequential(
            nn.Linear(regime_dim * 2, 16),
            nn.ReLU(inplace=True), nn.Linear(16, 1))

        # --- Acceleration (optional, for 'full') ---
        if variant in ('full',):
            self.acc_q = nn.Linear(rocket_dim, regime_dim)
            self.acc_gate = nn.Sequential(
                nn.Linear(regime_dim * 2, 16),
                nn.ReLU(inplace=True), nn.Linear(16, 1))

        # --- Transport-regime fusion (same as TURS) ---
        self.transport_proj = nn.Linear(self.C_T, D, bias=False)
        self.regime_proj = nn.Linear(regime_dim, D, bias=False)

        gate_in_dim = self.C_T + regime_dim + regime_dim + regime_dim
        self.gate_T = nn.Sequential(
            nn.Linear(gate_in_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
        self.gate_R = nn.Sequential(
            nn.Linear(gate_in_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
        self.alpha_net = nn.Sequential(
            nn.Linear(gate_in_dim + 2, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))

        self.interact_T = nn.Linear(self.C_T, D, bias=False)
        self.interact_R = nn.Linear(regime_dim, D, bias=False)
        self.lambda_I = nn.Sequential(
            nn.Linear(regime_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

        # Rocket-mediated interaction
        self.interact_RM = nn.Linear(rocket_dim, D, bias=False)
        self.lambda_RM = nn.Sequential(
            nn.Linear(rocket_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

        # Residual injection gate
        self.res_gate = nn.Sequential(
            nn.Linear(regime_dim * 2, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
        self.res_proj = nn.Linear(D, D, bias=False)

        # --- FiLM modulation ---
        mod_in = regime_dim + regime_dim  # z + v
        self.film = nn.Sequential(
            nn.Linear(mod_in, D), nn.ReLU(inplace=True), nn.Linear(D, D * 2))
        self.film_gate = nn.Sequential(
            nn.Linear(regime_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

        # --- Classifier ---
        cls_in = D * 2 + self.C_T + regime_dim + regime_dim + 1 + regime_dim + D
        # extra inputs depending on variant
        if variant in ('cs', 'full'):
            cls_in += cs_out_dim  # cross-scale struct
        if variant in ('d', 'full'):
            cls_in += rocket_dim  # dynamics q
        if variant in ('as', 'full'):
            cls_in += rocket_dim  # adaptive response
        if variant == 'full':
            cls_in += regime_dim  # acceleration

        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))

    def _compute_velocity(self, z, lin):
        """Feature-space velocity: v = W*z (learned projection, no temporal info)."""
        return lin(z)

    def _get_scale_features(self, R_raw):
        """Partition response bank outputs by scale group, compute per-group descriptors.

        Returns list of [B, filters_per_scale * 6] tensors.
        """
        fps = self.response_bank.filters_per_scale
        B = R_raw.shape[0]
        scale_feats = []
        for si in range(self.response_bank.n_scales):
            R_si = R_raw[:, si * fps:(si + 1) * fps, :]  # [B, fps, L]
            desc = self.response_bank.describe(R_si)       # [B, fps * 6]
            scale_feats.append(desc)
        return scale_feats

    def _adaptive_scale_response(self, R_raw, z):
        """Regime-conditioned scale selection.

        Returns adaptive response: [B, rocket_dim]
        """
        scale_feats = self._get_scale_features(R_raw)
        scale_weights = self.scale_selector(z)  # [B, n_scales]

        adapted = torch.zeros(R_raw.shape[0], self.rocket_dim, device=R_raw.device)
        for si in range(self.response_bank.n_scales):
            proj = self.adaptive_pool[si](scale_feats[si])  # [B, rocket_dim]
            adapted = adapted + scale_weights[:, si:si+1] * proj
        return adapted

    def forward(self, x, return_aux=False):
        """
        x: [B, 1, L]
        Returns: logits [B, K] and optionally aux dict
        """
        B, _, L = x.shape
        variant = self.variant

        # === Transport ===
        T = self.transport_builder(x)  # [B, 3, L]
        T_e = self.transport_encoder(T)  # [B, C_T, L]
        t_pooled = T_e.mean(dim=2)  # [B, C_T]

        # === Inception backbone ===
        h = self.proj_in(x)
        H1 = self.block1(h)
        H2 = self.block2(H1)
        H3 = self.block3(H2)
        H4 = self.block4(H3)

        # === Multiscale Response Bank ===
        R_raw = self.response_bank(x)  # [B, n_total, L]
        desc_R = self.response_bank.describe(R_raw)  # [B, raw_dim]
        F_M = self.response_proj(desc_R)  # [B, rocket_dim]

        # === Response Dynamics (D / full) ===
        q = torch.zeros(B, self.rocket_dim, device=x.device)
        if variant in ('d', 'full'):
            # First-order: ΔR (difference along time)
            dR_raw = R_raw[:, :, 1:] - R_raw[:, :, :-1]
            if dR_raw.shape[2] > 0:
                desc_dR = self.response_bank.describe(dR_raw)
            else:
                desc_dR = torch.zeros(B, self.response_bank.raw_dim, device=x.device)

            # Second-order: Δ²R
            ddR_raw = dR_raw[:, :, 1:] - dR_raw[:, :, :-1]
            if ddR_raw.shape[2] > 0:
                desc_ddR = self.response_bank.describe(ddR_raw)
            else:
                desc_ddR = torch.zeros(B, self.response_bank.raw_dim, device=x.device)

            # Cross-scale structure for dynamics (scale pairwise products)
            scale_feats = self._get_scale_features(R_raw)
            cs_struct_list = []
            for i in range(len(scale_feats)):
                for j in range(i + 1, len(scale_feats)):
                    cs_struct_list.append(
                        self.scale_proj[i](scale_feats[i]) *
                        self.scale_proj[j](scale_feats[j]))
            if cs_struct_list:
                cs_struct = torch.cat(cs_struct_list, dim=1)
            else:
                cs_struct = torch.zeros(B, 0, device=x.device)

            q = self.dynamics_enc(desc_R, desc_dR, desc_ddR, cs_struct)

        # === Cross-Scale Interaction (CS / full) ===
        c_struct = torch.zeros(B, 0, device=x.device)
        if variant in ('cs', 'full'):
            scale_feats = self._get_scale_features(R_raw)
            scale_projs = [self.scale_proj[si](sf)
                           for si, sf in enumerate(scale_feats)]
            c_struct = self.cross_scale(scale_projs)  # [B, cs_out_dim]
            F_M = F_M + self.cs_merge(c_struct)

        # === Adaptive Scale Selection (AS / full) ===
        F_M_adaptive = None
        if variant in ('as', 'full'):
            F_M_adaptive = None  # Will compute after regime is known

        # === Regime encoder ===
        # For RK: compute preliminary regime, then modulate response
        if variant in ('rk',):
            regime_in_pre = [H4.mean(dim=2), F_M]
            if variant in ('d', 'full'):
                regime_in_pre.append(q)
            if variant in ('cs', 'full'):
                regime_in_pre.append(c_struct)
            z_pre, u_pre = self.regime_main(torch.cat(regime_in_pre, dim=1))
            # RK: regime-conditioned per-filter gain modulation
            rk_gains = torch.sigmoid(self.kernel_residual_net(z_pre))  # [B, n_total]
            # Apply gains to raw responses, then re-describe
            # rk_gains: [B, n_total] -> [B, n_total, 1] for broadcasting
            R_raw_mod = R_raw * (1.0 + rk_gains.unsqueeze(2))  # [B, n_total, L]
            desc_R = self.response_bank.describe(R_raw_mod)  # re-describe
            F_M = self.response_proj(desc_R)  # re-project modulated response

        regime_in = [H4.mean(dim=2), F_M]
        if variant in ('d', 'full'):
            regime_in.append(q)
        if variant in ('cs', 'full'):
            regime_in.append(c_struct)
        z, u = self.regime_main(torch.cat(regime_in, dim=1))

        # === Adaptive Scale Response (AS / full) — after regime ===
        if variant in ('as', 'full'):
            F_M_adaptive = self._adaptive_scale_response(R_raw, z)
            F_M = F_M + F_M_adaptive

        # === Velocity ===
        v = self._compute_velocity(z, self.vel_linear)
        q_v = self.vel_q(F_M)
        g_v = torch.sigmoid(self.vel_gate(torch.cat([z, q_v], dim=1)))
        v = v + g_v * q_v
        s = torch.norm(v, dim=1, keepdim=True)

        # === Acceleration (full only) ===
        a = None
        if variant in ('full',):
            a = self.vel_linear(v)  # base acceleration
            q_a = self.acc_q(F_M)
            g_a = torch.sigmoid(self.acc_gate(torch.cat([z, q_a], dim=1)))
            a = a + g_a * q_a

        # === Transport-regime fusion ===
        F_T = self.transport_proj(t_pooled)
        F_R = self.regime_proj(z)

        gate_in = torch.cat([t_pooled, z, v, u], dim=1)
        g_T = torch.sigmoid(self.gate_T(gate_in))
        g_R = torch.sigmoid(self.gate_R(gate_in))

        alpha_in = torch.cat([gate_in, g_T, g_R], dim=1)
        alpha = torch.sigmoid(self.alpha_net(alpha_in))

        F_fused = alpha * (g_T * F_T) + (1 - alpha) * (g_R * F_R)

        # Transport-regime interaction
        I_T = self.interact_T(t_pooled)
        I_R = self.interact_R(z)
        lam_I = torch.sigmoid(self.lambda_I(z))
        I_TR = I_T * I_R
        u_gate = torch.sigmoid(u.mean(dim=1, keepdim=True))
        I_TR = (1 - u_gate) * I_TR

        # Rocket-mediated regime interaction
        I_RM = self.interact_RM(F_M)
        lam_RM = torch.sigmoid(self.lambda_RM(F_M))

        F_fused = F_fused + lam_I * I_TR + lam_RM * I_RM

        # Residual injection into H4
        res_g = torch.sigmoid(self.res_gate(torch.cat([z, v], dim=1)))
        H4 = H4 + res_g.unsqueeze(2) * self.res_proj(F_fused).unsqueeze(2)

        # === FiLM modulation ===
        mod_cat = torch.cat([z, v], dim=1)
        film_out = self.film(mod_cat)
        gamma, beta = film_out.chunk(2, dim=1)
        u_gate2 = torch.sigmoid(self.film_gate(u))
        H4 = H4 * (1 + u_gate2.unsqueeze(2) * gamma.unsqueeze(2)) + \
             u_gate2.unsqueeze(2) * beta.unsqueeze(2)

        # === Global pooling ===
        h_gap = H4.mean(dim=2)
        h_gmp = H4.max(dim=2)[0]

        # === Classifier input ===
        parts = [h_gap, h_gmp, t_pooled, z, v, s, u, I_TR]
        if variant in ('cs', 'full'):
            parts.append(c_struct)
        if variant in ('d', 'full'):
            parts.append(q)
        if variant in ('as', 'full') and F_M_adaptive is not None:
            parts.append(F_M_adaptive)
        if variant == 'full' and a is not None:
            parts.append(a)

        G = torch.cat(parts, dim=1)
        logits = self.classifier(G)

        if return_aux:
            aux = {
                "regime": z,
                "regime_velocity": v,
                "regime_speed": s,
                "uncertainty": u,
                "alpha": alpha,
                "transport_gate": g_T,
                "regime_gate": g_R,
                "F_M": F_M,
                "response_q": q,
            }
            if variant in ('cs', 'full'):
                aux["cross_scale"] = c_struct
            if variant in ('as', 'full'):
                aux["F_M_adaptive"] = F_M_adaptive
                aux["scale_weights"] = self.scale_selector(z) if hasattr(self, 'scale_selector') else None
            if a is not None:
                aux["acceleration"] = a
            return logits, aux
        return logits


# ============================================================
# Utilities
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("TURS-AMR Shape Verification")
    print("=" * 60)

    for variant in ['rv', 'd', 'cs', 'as', 'rk', 'full']:
        for L, nc in [(140, 5), (1024, 4)]:
            model = TURSAMR(in_channels=1, num_classes=nc,
                            regime_dim=16, variant=variant)
            n = count_parameters(model)
            x = torch.randn(4, 1, L)
            try:
                logits, aux = model(x, return_aux=True)
                assert logits.shape == (4, nc), f"{variant} L{L}: {logits.shape}"
                assert not torch.isnan(logits).any(), f"NaN in logits"
                logits.sum().backward()
                no_grad = [nm for nm, p in model.named_parameters()
                           if p.requires_grad and p.grad is None]
                if no_grad:
                    print(f"  WARNING: no grad for {no_grad}")
                print(f"  {variant:6s} L={L:4d} C={nc}: params={n:,} OK")
            except Exception as e:
                print(f"  {variant:6s} L={L:4d} C={nc}: FAILED — {e}")
                import traceback; traceback.print_exc()

    print("\nDone.")
