"""
TURS-Net — Transport–Uncertainty Regime Synergy Network

Central innovation: uncertainty-aware adaptive fusion between transport/
distribution information and continuous regime information.

  F_TR = α · F_T + (1-α) · F_R + λ_I · I_TR

where α = sigmoid(f(T, z, v, u)) is learned dynamically per sample.

Two variants:
  - TURS-Lite (~145K): single-scale fusion at H3
  - TURS-Strong (~220K): multi-scale fusion at H1, H2, H3

Ablation flags:
  use_transport, use_regime, use_velocity, use_uncertainty,
  use_adaptive_fusion, use_transport_regime_interaction,
  multi_scale_fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Transport Representation Builder
# ============================================================
class TransportBuilder(nn.Module):
    """Build [raw, quantile, drift] from raw input."""

    def forward(self, x):
        """x: [B, 1, L] -> [B, 3, L]"""
        raw = x.squeeze(1)  # [B, L]
        sorted_x, _ = torch.sort(raw, dim=1)
        drift = torch.zeros_like(raw)
        for k in [1, 2, 4]:
            shifted = torch.zeros_like(raw)
            if k < raw.shape[1]:
                shifted[:, k:] = sorted_x[:, :-k]
            drift += torch.abs(sorted_x - shifted)
        drift /= 3.0
        return torch.stack([raw, sorted_x, drift], dim=1)  # [B, 3, L]


# ============================================================
# InceptionTime Block
# ============================================================
class InceptionBlock(nn.Module):
    """Inception block with bottleneck, k=9/19/39 + maxpool branch."""

    def __init__(self, in_ch, out_ch, bottleneck_ch=None):
        super().__init__()
        if bottleneck_ch is None:
            bottleneck_ch = in_ch // 2

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_ch, bottleneck_ch, 1, bias=False),
            nn.BatchNorm1d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # 4 branches: 3 conv + 1 pool, each out_ch//4 channels -> total = out_ch
        branch_ch = out_ch // 4
        rem_ch = out_ch - 3 * branch_ch  # remainder for pool branch

        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_ch, branch_ch, k, padding=k // 2, bias=False)
            for k in [9, 19, 39]
        ])

        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, rem_ch, 1, bias=False),
            nn.BatchNorm1d(rem_ch),
            nn.ReLU(inplace=True),
        )

        self.out_bn = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b = self.bottleneck(x)
        cat = torch.cat([c(b) for c in self.convs] + [self.pool_branch(x)], dim=1)
        return self.relu(self.out_bn(cat) + self.residual(x))


# ============================================================
# Transport Encoder
# ============================================================
class TransportEncoder(nn.Module):
    """Encode [B, 3, L] transport into [B, C_T, L] features."""

    def __init__(self, out_ch=16, final_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 9, padding=4, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv1d(out_ch, final_ch, 1, bias=False)

    def forward(self, T):
        return self.proj(self.net(T))  # [B, final_ch, L]


class TransportEncoderDeep(nn.Module):
    """Deeper transport encoder with hierarchical features for Strong."""

    def __init__(self, out_ch=16, final_ch=64):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv1d(3, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
        self.scale2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 9, padding=4, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.scale3 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 9, padding=4, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.proj1 = nn.Conv1d(out_ch, final_ch, 1, bias=False)
        self.proj2 = nn.Conv1d(out_ch, final_ch, 1, bias=False)
        self.proj3 = nn.Conv1d(out_ch, final_ch, 1, bias=False)

    def forward(self, T):
        t1 = self.scale1(T)
        t2 = self.scale2(t1)
        t3 = self.scale3(t2)
        return self.proj1(t1), self.proj2(t2), self.proj3(t3)


# ============================================================
# Regime Encoder
# ============================================================
class RegimeEncoder(nn.Module):
    """Encode features into regime z, velocity v, speed s, uncertainty u."""

    def __init__(self, in_ch, regime_dim=16):
        super().__init__()
        self.regime_dim = regime_dim

        self.mu_net = nn.Sequential(
            nn.Linear(in_ch, 32), nn.ReLU(inplace=True), nn.Linear(32, regime_dim)
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(in_ch, 32), nn.ReLU(inplace=True), nn.Linear(32, regime_dim)
        )

    def forward(self, h):
        """h: [B, C] -> z: [B, d], u: [B, d]"""
        z = self.mu_net(h)
        logvar = self.logvar_net(h)
        logvar = torch.clamp(logvar, -5.0, 2.0)
        sigma = F.softplus(logvar) + 1e-6
        return z, sigma

    def sample(self, z, sigma, training=True):
        if training:
            eps = torch.randn_like(sigma)
            return z + sigma * eps
        return z


# ============================================================
# TURS-Net
# ============================================================
class TURSNet(nn.Module):
    """
    Transport–Uncertainty Regime Synergy Network.

    variant="lite": ~145K params, single-scale fusion at H3
    variant="strong": ~220K params, multi-scale fusion at H1/H2/H3
    """

    def __init__(self, in_channels=1, num_classes=5, regime_dim=16,
                 variant="lite", dropout=0.1,
                 use_transport=True, use_regime=True, use_velocity=True,
                 use_uncertainty=True, use_adaptive_fusion=True,
                 use_transport_regime_interaction=True,
                 multi_scale_fusion=False):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.regime_dim = regime_dim

        # Ablation flags
        self.use_transport = use_transport
        self.use_regime = use_regime
        self.use_velocity = use_velocity
        self.use_uncertainty = use_uncertainty
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_interaction = use_transport_regime_interaction
        self.multi_scale = multi_scale_fusion

        base_ch = 32
        D = base_ch * 2  # 64

        # --- Transport ---
        if use_transport:
            self.transport_builder = TransportBuilder()
            self.transport_encoder = TransportEncoder(out_ch=16, final_ch=D)
            self.C_T = D

        # --- Inception backbone ---
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True),
        )
        self.block1 = InceptionBlock(base_ch, base_ch)
        self.block2 = InceptionBlock(base_ch, base_ch)
        self.block3 = InceptionBlock(base_ch, D)
        self.block4 = InceptionBlock(D, D)

        # Channel dims at each scale
        self.C1 = base_ch   # 32
        self.C2 = base_ch   # 32
        self.C3 = D          # 64
        self.C4 = D          # 64

        # --- Regime encoder(s) ---
        if use_regime:
            if multi_scale_fusion:
                self.regime_1 = RegimeEncoder(self.C1, regime_dim)
                self.regime_2 = RegimeEncoder(self.C2 + regime_dim, regime_dim)
                self.regime_3 = RegimeEncoder(self.C3 + regime_dim, regime_dim)
            else:
                self.regime_main = RegimeEncoder(self.C4, regime_dim)

        # --- Transport-regime fusion components ---
        if use_transport and use_regime:
            rd = regime_dim
            # For single-scale (Lite): fusion at H3 level
            if not multi_scale_fusion:
                C_fuse = self.C4
                self.transport_proj = nn.Linear(self.C_T, C_fuse, bias=False)
                self.regime_proj = nn.Linear(rd, C_fuse, bias=False)

                # Transport gate
                self.gate_T = nn.Sequential(
                    nn.Linear(self.C_T + rd + (rd if use_velocity else 0) +
                              (rd if use_uncertainty else 0), 16),
                    nn.ReLU(inplace=True), nn.Linear(16, 1),
                )
                # Regime gate
                self.gate_R = nn.Sequential(
                    nn.Linear(self.C_T + rd + (rd if use_velocity else 0) +
                              (rd if use_uncertainty else 0), 16),
                    nn.ReLU(inplace=True), nn.Linear(16, 1),
                )
                # Adaptive alpha
                n_extra = 2  # g_T, g_R scalars
                self.alpha_net = nn.Sequential(
                    nn.Linear(self.C_T + rd + (rd if use_velocity else 0) +
                              (rd if use_uncertainty else 0) + n_extra, 16),
                    nn.ReLU(inplace=True), nn.Linear(16, 1),
                )
                # Interaction projection
                if use_transport_regime_interaction:
                    self.interact_T = nn.Linear(self.C_T, C_fuse, bias=False)
                    self.interact_R = nn.Linear(rd, C_fuse, bias=False)
                    self.lambda_I = nn.Sequential(
                        nn.Linear(rd, 8), nn.ReLU(inplace=True), nn.Linear(8, 1),
                    )
                # Residual injection gate
                self.res_gate = nn.Sequential(
                    nn.Linear(rd + (rd if use_velocity else 0) +
                              (rd if use_uncertainty else 0), 16),
                    nn.ReLU(inplace=True), nn.Linear(16, 1),
                )
                self.res_proj = nn.Linear(C_fuse, self.C4, bias=False)
            else:
                # Multi-scale fusion (Strong)
                for s, (c_in, c_T) in enumerate(
                    [(self.C1, D), (self.C2, D), (self.C3, D)], start=1
                ):
                    setattr(self, f'transport_proj_{s}',
                            nn.Linear(self.C_T, c_in, bias=False))
                    setattr(self, f'regime_proj_{s}',
                            nn.Linear(rd, c_in, bias=False))
                    setattr(self, f'gate_T_{s}', nn.Sequential(
                        nn.Linear(self.C_T + rd + (rd if use_velocity else 0) +
                                  (rd if use_uncertainty else 0), 16),
                        nn.ReLU(inplace=True), nn.Linear(16, 1),
                    ))
                    setattr(self, f'gate_R_{s}', nn.Sequential(
                        nn.Linear(self.C_T + rd + (rd if use_velocity else 0) +
                                  (rd if use_uncertainty else 0), 16),
                        nn.ReLU(inplace=True), nn.Linear(16, 1),
                    ))
                    setattr(self, f'alpha_{s}', nn.Sequential(
                        nn.Linear(self.C_T + rd + (rd if use_velocity else 0) +
                                  (rd if use_uncertainty else 0) + 2, 16),
                        nn.ReLU(inplace=True), nn.Linear(16, 1),
                    ))
                    if use_transport_regime_interaction:
                        setattr(self, f'interact_T_{s}',
                                nn.Linear(self.C_T, c_in, bias=False))
                        setattr(self, f'interact_R_{s}',
                                nn.Linear(rd, c_in, bias=False))
                        setattr(self, f'lambda_I_{s}', nn.Sequential(
                            nn.Linear(rd, 8), nn.ReLU(inplace=True), nn.Linear(8, 1),
                        ))
                    setattr(self, f'res_gate_{s}', nn.Sequential(
                        nn.Linear(rd + (rd if use_velocity else 0) +
                                  (rd if use_uncertainty else 0), 16),
                        nn.ReLU(inplace=True), nn.Linear(16, 1),
                    ))
                    setattr(self, f'res_proj_{s}', nn.Linear(c_in, c_in, bias=False))

        # --- Uncertainty-aware FiLM modulation (lightweight) ---
        if use_uncertainty and use_regime:
            mod_in = regime_dim + (regime_dim if use_velocity else 0)
            self.film = nn.Sequential(
                nn.Linear(mod_in, D), nn.ReLU(inplace=True), nn.Linear(D, D * 2),
            )
            self.film_gate = nn.Sequential(
                nn.Linear(regime_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1),
            )

        # --- Classifier ---
        cls_in = D * 2  # GAP + GMP of H4
        if use_transport:
            cls_in += D  # transport global
        if use_regime:
            cls_in += regime_dim  # z
            if use_velocity:
                cls_in += regime_dim  # v
                cls_in += 1  # speed scalar
            if use_uncertainty:
                cls_in += regime_dim  # u
        if use_transport and use_regime and use_transport_regime_interaction:
            cls_in += D  # interaction global

        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # --- Auxiliary: regime velocity from features ---
        if use_velocity and use_regime:
            if multi_scale_fusion:
                # Velocity from local regime embeddings
                self.vel_linear_1 = nn.Linear(regime_dim, regime_dim)
                self.vel_linear_2 = nn.Linear(regime_dim, regime_dim)
                self.vel_linear_3 = nn.Linear(regime_dim, regime_dim)
            else:
                self.vel_linear = nn.Linear(regime_dim, regime_dim)

    def _compute_velocity(self, Z, lin):
        """Z: [B, d] -> v: [B, d]. For single sample, velocity is zero."""
        # Since we process batches independently (no temporal sequence),
        # velocity is derived from feature-space local embeddings.
        # We approximate velocity as the gradient of regime w.r.t. features
        # using the residual connection: v = W*z (learned velocity projection)
        return lin(Z)

    def _fuse_single_scale(self, H, T_e, z, v, u, scale_idx, H_orig):
        """Single-scale adaptive transport-regime fusion."""
        B, C, L = H.shape

        # Transport features projected to regime dim
        t_pooled = T_e.mean(dim=2)  # [B, C_T]
        F_T = getattr(self, f'transport_proj_{scale_idx}')(t_pooled)  # [B, C]
        F_T = F_T.unsqueeze(2).expand(-1, -1, L)  # [B, C, L]

        # Regime features projected
        F_R = getattr(self, f'regime_proj_{scale_idx}')(z)  # [B, C]
        F_R = F_R.unsqueeze(2).expand(-1, -1, L)  # [B, C, L]

        # Gate inputs
        gate_in = [t_pooled, z]
        if self.use_velocity:
            gate_in.append(v)
        if self.use_uncertainty:
            gate_in.append(u)
        gate_cat = torch.cat(gate_in, dim=1)

        # Transport gate
        g_T = torch.sigmoid(getattr(self, f'gate_T_{scale_idx}')(gate_cat))  # [B, 1]
        # Regime gate
        g_R = torch.sigmoid(getattr(self, f'gate_R_{scale_idx}')(gate_cat))  # [B, 1]

        # Adaptive alpha
        alpha_in = torch.cat([gate_cat, g_T, g_R], dim=1)
        alpha = torch.sigmoid(getattr(self, f'alpha_{scale_idx}')(alpha_in))  # [B, 1]

        # Fuse (broadcast gates to [B, 1, 1] for [B, C, L] features)
        g_T_3d = g_T.unsqueeze(2)  # [B, 1, 1]
        g_R_3d = g_R.unsqueeze(2)
        alpha_3d = alpha.unsqueeze(2)
        if self.use_adaptive_fusion:
            F_fused = alpha_3d * (g_T_3d * F_T) + (1 - alpha_3d) * (g_R_3d * F_R)
        else:
            F_fused = 0.5 * (g_T_3d * F_T) + 0.5 * (g_R_3d * F_R)

        # Interaction
        if self.use_interaction:
                I_T = getattr(self, f'interact_T_{scale_idx}')(t_pooled)  # [B, C]
                I_R = getattr(self, f'interact_R_{scale_idx}')(z)  # [B, C]
                lam_I = torch.sigmoid(getattr(self, f'lambda_I_{scale_idx}')(z))  # [B, 1]
                I = (I_T * I_R).unsqueeze(2).expand(-1, -1, L)  # [B, C, L]
                if self.use_uncertainty:
                    u_gate = torch.sigmoid(u.mean(dim=1, keepdim=True))  # [B, 1]
                    I = (1 - u_gate.unsqueeze(2)) * I
                F_fused = F_fused + lam_I.unsqueeze(2) * I

        # Residual injection
        res_g = torch.sigmoid(getattr(self, f'res_gate_{scale_idx}')(
            torch.cat([z] + ([v] if self.use_velocity else []) +
                      ([u] if self.use_uncertainty else []), dim=1)
        ))  # [B, 1]
        F_proj = getattr(self, f'res_proj_{scale_idx}')(
            F_fused.mean(dim=2)
        )  # [B, C]
        H_out = H + res_g.unsqueeze(2) * F_proj.unsqueeze(2)

        return H_out, alpha, g_T, g_R, F_fused

    def forward(self, x, return_aux=False):
        """
        x: [B, 1, L]
        Returns: logits [B, K] and optionally aux dict
        """
        B, _, L = x.shape

        # === Transport ===
        T_e = None
        if self.use_transport:
            T = self.transport_builder(x)  # [B, 3, L]
            T_e = self.transport_encoder(T)  # [B, C_T, L]

        # === Inception backbone ===
        h = self.proj(x)  # [B, 32, L]
        H1 = self.block1(h)   # [B, 32, L]
        H2 = self.block2(H1)  # [B, 32, L]
        H3 = self.block3(H2)  # [B, 64, L]
        H4 = self.block4(H3)  # [B, 64, L]

        # === Regime encoder ===
        z = v = u = s = None
        alpha_all = g_T_all = g_R_all = None

        if self.use_regime:
            if self.multi_scale:
                # Multi-scale regime
                v1 = H1.mean(dim=2)  # [B, 32]
                z1, u1 = self.regime_1(v1)

                v2 = H2.mean(dim=2)
                z2, u2 = self.regime_2(torch.cat([v2, z1], dim=1))

                v3 = H3.mean(dim=2)
                z3, u3 = self.regime_3(torch.cat([v3, z2], dim=1))

                # Final regime from H4 (additional)
                v4 = H4.mean(dim=2)
                # Use regime_3 output as final
                z, u = z3, u3

                # Velocity from feature-space local embeddings
                vel_1 = self._compute_velocity(z1, self.vel_linear_1)
                vel_2 = self._compute_velocity(z2, self.vel_linear_2)
                vel_3 = self._compute_velocity(z3, self.vel_linear_3)
                v = vel_3  # final velocity
            else:
                v_feat = H4.mean(dim=2)  # [B, 64]
                z, u = self.regime_main(v_feat)  # [B, d], [B, d]
                v = self._compute_velocity(z, self.vel_linear)

            s = torch.norm(v, dim=1, keepdim=True)  # [B, 1]

            # === Multi-scale fusion (Strong) ===
            if self.multi_scale and self.use_transport:
                H1, a1, g1t, g1r, _ = self._fuse_single_scale(
                    H1, T_e, z1, vel_1, u1, 1, H1)
                # Re-propagate through remaining blocks so gradients flow
                H2 = self.block2(H1)
                H2, a2, g2t, g2r, _ = self._fuse_single_scale(
                    H2, T_e, z2, vel_2, u2, 2, H2)
                H3 = self.block3(H2)
                H3, a3, g3t, g3r, _ = self._fuse_single_scale(
                    H3, T_e, z3, vel_3, u3, 3, H3)
                alpha_all = torch.stack([a1.squeeze(), a2.squeeze(), a3.squeeze()], dim=1)
                g_T_all = torch.stack([g1t.squeeze(), g2t.squeeze(), g3t.squeeze()], dim=1)
                g_R_all = torch.stack([g1r.squeeze(), g2r.squeeze(), g3r.squeeze()], dim=1)
                H4 = self.block4(H3)

        # === Single-scale fusion at H4 (Lite) ===
        I_global = None
        if self.use_regime and self.use_transport and not self.multi_scale:
            t_pooled = T_e.mean(dim=2)
            F_T = self.transport_proj(t_pooled)  # [B, D]

            F_R = self.regime_proj(z)  # [B, D]

            gate_in = [t_pooled, z]
            if self.use_velocity:
                gate_in.append(v)
            if self.use_uncertainty:
                gate_in.append(u)
            gate_cat = torch.cat(gate_in, dim=1)

            g_T = torch.sigmoid(self.gate_T(gate_cat))  # [B, 1]
            g_R = torch.sigmoid(self.gate_R(gate_cat))  # [B, 1]

            alpha_in = torch.cat([gate_cat, g_T, g_R], dim=1)
            alpha = torch.sigmoid(self.alpha_net(alpha_in))  # [B, 1]

            if self.use_adaptive_fusion:
                F_fused = alpha * (g_T * F_T) + (1 - alpha) * (g_R * F_R)
            else:
                F_fused = 0.5 * g_T * F_T + 0.5 * g_R * F_R

            if self.use_interaction:
                I_T = self.interact_T(t_pooled)
                I_R = self.interact_R(z)
                lam_I = torch.sigmoid(self.lambda_I(z))
                I_TR = I_T * I_R  # [B, D]
                if self.use_uncertainty:
                    u_gate = torch.sigmoid(u.mean(dim=1, keepdim=True))
                    I_TR = (1 - u_gate) * I_TR
                F_fused = F_fused + lam_I * I_TR
                I_global = I_TR

            # Residual injection
            res_g = torch.sigmoid(self.res_gate(
                torch.cat([z] + ([v] if self.use_velocity else []) +
                          ([u] if self.use_uncertainty else []), dim=1)
            ))
            H4 = H4 + res_g.unsqueeze(2) * self.res_proj(F_fused).unsqueeze(2)

            alpha_all = alpha
            g_T_all = g_T
            g_R_all = g_R

        # === Uncertainty-aware FiLM modulation ===
        if self.use_uncertainty and self.use_regime:
            mod_in = [z]
            if self.use_velocity:
                mod_in.append(v)
            mod_cat = torch.cat(mod_in, dim=1)
            film_out = self.film(mod_cat)  # [B, 2*D]
            gamma, beta = film_out.chunk(2, dim=1)  # each [B, D]
            u_gate = torch.sigmoid(self.film_gate(u))  # [B, 1]
            H4 = H4 * (1 + u_gate.unsqueeze(2) * gamma.unsqueeze(2)) + \
                 u_gate.unsqueeze(2) * beta.unsqueeze(2)

        # === Global pooling ===
        h_gap = H4.mean(dim=2)  # [B, D]
        h_gmp = H4.max(dim=2)[0]  # [B, D]

        # === Classifier input ===
        parts = [h_gap, h_gmp]
        if self.use_transport:
            parts.append(T_e.mean(dim=2))  # transport global
        if self.use_regime:
            parts.append(z)
            if self.use_velocity:
                parts.append(v)
                parts.append(s)
            if self.use_uncertainty:
                parts.append(u)
        if self.use_transport and self.use_regime and self.use_interaction:
            if I_global is not None:
                parts.append(I_global)
            elif T_e is not None and z is not None:
                # For multi-scale: use deepest interaction
                I_T_final = self.transport_proj_3(T_e.mean(dim=2)) if self.multi_scale else t_pooled
                I_R_final = self.regime_proj_3(z) if self.multi_scale else self.regime_proj(z)
                parts.append(I_T_final * I_R_final)

        G = torch.cat(parts, dim=1)
        logits = self.classifier(G)

        if return_aux:
            aux = {
                "regime": z,
                "regime_velocity": v,
                "regime_speed": s,
                "uncertainty": u,
                "alpha": alpha_all,
                "transport_gate": g_T_all,
                "regime_gate": g_R_all,
            }
            return logits, aux
        return logits


# ============================================================
# Loss Function
# ============================================================
class TURSLoss(nn.Module):
    """
    L = L_task + λ_s L_smooth + λ_v L_velocity + λ_u L_uncertainty + λ_i L_interaction
    """

    def __init__(self, num_classes=5, focal_gamma=2.0, use_focal=False,
                 lambda_smooth=0.01, lambda_vel=0.005,
                 lambda_unc=0.01, lambda_inter=0.005):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        self.ls = lambda_smooth
        self.lv = lambda_vel
        self.lu = lambda_unc
        self.li = lambda_inter
        self.num_classes = num_classes

    def forward(self, logits, targets, aux=None):
        """Returns (total_loss, loss_dict)"""
        # Task loss
        if self.use_focal:
            ce = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce)
            task_loss = ((1 - pt) ** self.focal_gamma * ce).mean()
        else:
            task_loss = F.cross_entropy(logits, targets)

        losses = {"task": task_loss.item()}
        total = task_loss

        if aux is not None:
            # Smoothness
            if self.ls > 0 and "regime" in aux and aux["regime"] is not None:
                z = aux["regime"]
                if z.shape[0] > 1:
                    # Penalize variance collapse
                    z_var = z.var(dim=0).mean()
                    smooth_loss = F.relu(0.1 - z_var)
                    losses["smooth"] = smooth_loss.item()
                    total = total + self.ls * smooth_loss

            # Velocity consistency
            if self.lv > 0 and "regime_velocity" in aux and aux["regime_velocity"] is not None:
                v = aux["regime_velocity"]
                v_norm = v.norm(dim=1)
                # Penalize extremely large velocities
                vel_loss = F.relu(v_norm - 3.0).mean()
                losses["velocity"] = vel_loss.item()
                total = total + self.lv * vel_loss

            # Uncertainty calibration
            if self.lu > 0 and "uncertainty" in aux and aux["uncertainty"] is not None:
                u = aux["uncertainty"]  # [B, d]
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1)
                    p_max = probs.max(dim=1)[0]  # [B]
                    target_u = (1 - p_max).unsqueeze(1)  # [B, 1]
                u_scalar = torch.sigmoid(u.mean(dim=1, keepdim=True))  # [B, 1]
                unc_loss = F.mse_loss(u_scalar, target_u)
                losses["uncertainty"] = unc_loss.item()
                total = total + self.lu * unc_loss

            # Interaction regularization (prevent gate collapse)
            if self.li > 0 and "alpha" in aux and aux["alpha"] is not None:
                a = aux["alpha"].squeeze()
                if a.dim() > 0:
                    gate_loss = (a * (1 - a)).mean()  # encourage diversity
                    losses["interaction"] = gate_loss.item()
                    total = total + self.li * gate_loss

        return total, losses


# ============================================================
# Utilities
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_by_subsystem(model):
    """Count parameters by architectural subsystem."""
    counts = {
        "backbone": 0,
        "transport": 0,
        "regime": 0,
        "fusion": 0,
        "classifier": 0,
    }
    for name, p in model.named_parameters():
        if "block" in name or "proj" in name and "transport" not in name and "regime" not in name:
            if "transport" not in name and "regime" not in name and "gate" not in name:
                counts["backbone"] += p.numel()
        if "transport" in name:
            counts["transport"] += p.numel()
        if "regime" in name or "vel" in name:
            counts["regime"] += p.numel()
        if "gate" in name or "alpha" in name or "interact" in name or "film" in name:
            counts["fusion"] += p.numel()
        if "classifier" in name:
            counts["classifier"] += p.numel()
    # Fallback: anything not counted
    total = sum(p.numel() for p in model.parameters())
    accounted = sum(counts.values())
    counts["other"] = total - accounted
    return counts


if __name__ == "__main__":
    print("=" * 60)
    print("TURS-Net Shape Verification")
    print("=" * 60)

    for variant, rd in [("lite", 16), ("strong", 24)]:
        for L, nc in [(140, 5), (1024, 4)]:
            model = TURSNet(
                in_channels=1, num_classes=nc, regime_dim=rd,
                variant=variant,
                multi_scale_fusion=(variant == "strong"),
            )
            n = count_parameters(model)
            print(f"\n  {variant.upper()} | L={L} C={nc} | Params: {n:,}")

            x = torch.randn(4, 1, L)
            logits, aux = model(x, return_aux=True)

            print(f"    logits: {logits.shape}")
            for k, v in aux.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k:25s}: {list(v.shape)}")

            # Check no NaN
            for k, v in aux.items():
                if isinstance(v, torch.Tensor):
                    assert not torch.isnan(v).any(), f"NaN in {k}"
            assert not torch.isnan(logits).any(), "NaN in logits"

            # Gradient check
            logits.sum().backward()
            no_grad = [n for n, p in model.named_parameters()
                       if p.grad is None]
            if no_grad:
                print(f"    WARNING: no gradient for {no_grad}")
            else:
                print(f"    Gradient check: OK")

            counts = count_by_subsystem(model)
            print(f"    Subsystems: {counts}")

    print("\nAll checks passed.")
