"""
TURS-Rocket — TURS-Lite + compact ROCKET-inspired multiscale response path.

Reference points (seed 42, corrected protocol):
    TURS-Lite : 0.7709 avg MF1
    MiniROCKET: 0.8089 avg MF1

The ROCKET inductive bias (diverse fixed random multiscale convolutional
projections summarized by PPV / mean-positive statistics) is integrated as an
internal representation pathway that *participates in* TURS's adaptive
transport-regime reasoning — not concatenated as external features.

Variants (all single-scale Lite-style, seed-42 comparable):
  rr   : Rocket Response      — gated rocket rep added to classifier input
  rtr  : Rocket Temporal Res. — conditional residual modulation of H4
  3f   : Three-way fusion     — softmax(alpha_T, alpha_R, alpha_M) fusion
  rcf  : Rocket-Conditioned F — rocket conditions the 2-way fusion + I_TM, I_RM
  rs   : Rocket-Sensed Regime — regime encoder sees [H, F_M]
  rv   : Rocket-informed Vel. — rocket responses inform regime velocity
  full : rs + rcf + rocket-gated classifier input (integrated model)

Original TURS-Lite (models/tursnet.py) is untouched.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tursnet import TURSNet, RegimeEncoder, TURSLoss


class RocketResponseBank(nn.Module):
    """Compact ROCKET-inspired fixed random convolution bank (frozen).

    Diverse (kernel, dilation) grid, weights ~ N(0,1), biases ~ U(-1,1)
    (MiniROCKET-style init). Each filter response is summarized by:
        PPV  = mean(1[R_k(t) > 0])
        MPV  = mean(ReLU(R_k(t)))
    Output [B, 2*F] is projected to a small rocket_dim by the caller.
    """

    def __init__(self, in_ch=1, kernels=(7, 9, 13, 19), dilations=(1, 2, 4),
                 filters_per_cfg=4, seed=0):
        super().__init__()
        self.convs = nn.ModuleList()
        cfgs = [(k, d) for k in kernels for d in dilations]
        for (k, d) in cfgs:
            for _ in range(filters_per_cfg):
                conv = nn.Conv1d(in_ch, 1, k, padding='same', dilation=d, bias=True)
                with torch.no_grad():
                    conv.weight.normal_(0.0, 1.0)
                    conv.bias.uniform_(-1.0, 1.0)
                for p in conv.parameters():
                    p.requires_grad = False
                self.convs.append(conv)
        self.n_filters = len(cfgs) * filters_per_cfg
        self.out_dim = self.n_filters * 2  # PPV + MPV

    def forward(self, x):
        """x: [B, 1, L] -> [B, 2F]"""
        ppvs, mpvs = [], []
        for conv in self.convs:
            r = conv(x).squeeze(1)  # [B, L]
            ppvs.append((r > 0).float().mean(dim=1))
            mpvs.append(F.relu(r).mean(dim=1))
        return torch.stack(ppvs + mpvs, dim=1)


class TURSRocket(TURSNet):
    """TURS-Lite + rocket-response path, one of the six variants or 'full'."""

    def __init__(self, in_channels=1, num_classes=5, regime_dim=16,
                 variant='rr', rocket_dim=32, dropout=0.1,
                 fixed_alpha=None, seed=0, **kw):
        super().__init__(in_channels=in_channels, num_classes=num_classes,
                         regime_dim=regime_dim, variant='lite', dropout=dropout, **kw)
        assert not self.multi_scale, "TURS-Rocket variants are Lite (single-scale)"
        self.variant = variant
        self.fixed_alpha = fixed_alpha
        D = self.C4            # 64
        rd = regime_dim
        v_off = rd if self.use_velocity else 0
        u_off = rd if self.use_uncertainty else 0
        gate_in_dim = self.C_T + rd + v_off + u_off

        # ---- Rocket response path ----
        self.rocket = RocketResponseBank(in_ch=in_channels, seed=seed)
        self.rocket_proj = nn.Sequential(
            nn.Linear(self.rocket.out_dim, rocket_dim), nn.ReLU(inplace=True))
        self.rocket_dim = rocket_dim

        # rs / full: regime encoder sees [H4_pool, F_M]
        if variant in ('rs', 'full'):
            self.regime_main_r = RegimeEncoder(D + rocket_dim, rd)

        # rtr: conditional residual rocket modulation of H4
        if variant == 'rtr':
            self.rtr_gate = nn.Sequential(
                nn.Linear(D + rd + u_off + rocket_dim, 16),
                nn.ReLU(inplace=True), nn.Linear(16, 1))
            self.rtr_proj = nn.Linear(rocket_dim, D, bias=False)

        # 3f: three-way softmax fusion
        if variant == '3f':
            self.alpha3 = nn.Sequential(
                nn.Linear(gate_in_dim + rocket_dim, 16),
                nn.ReLU(inplace=True), nn.Linear(16, 3))
            self.FM_to_D = nn.Linear(rocket_dim, D, bias=False)

        # rcf / full: rocket-conditioned two-way fusion + rocket interactions
        if variant in ('rcf', 'full'):
            self.alpha2 = nn.Sequential(
                nn.Linear(gate_in_dim + rocket_dim, 16),
                nn.ReLU(inplace=True), nn.Linear(16, 2))
            self.interact_M = nn.Linear(rocket_dim, D, bias=False)
            self.interact_TM = nn.Linear(self.C_T, D, bias=False)
            self.interact_RM = nn.Linear(rd, D, bias=False)
            self.lambda_TM = nn.Sequential(
                nn.Linear(rocket_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))
            self.lambda_RM = nn.Sequential(
                nn.Linear(rocket_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

        # rv: rocket-informed regime velocity
        if variant == 'rv':
            self.vel_q = nn.Linear(rocket_dim, rd)
            self.vel_gate = nn.Sequential(
                nn.Linear(rd + u_off + rocket_dim, 16),
                nn.ReLU(inplace=True), nn.Linear(16, 1))

        # uncertainty-gated rocket contribution to the classifier
        if variant in ('rr', 'rcf', 'full'):
            self.gM = nn.Sequential(
                nn.Linear(rd + rocket_dim, 16),
                nn.ReLU(inplace=True), nn.Linear(16, 1))

        # Variants with their own alpha net don't use super's alpha_net
        if variant in ('3f', 'rcf', 'full'):
            del self.alpha_net
        # Variants with rocket-sensed regime use regime_main_r, not regime_main
        if variant in ('rs', 'full'):
            del self.regime_main

        # ---- Rebuild classifier with rocket input where applicable ----
        cls_in = D * 2 + self.C_T + rd + v_off + 1 + u_off + D
        if variant in ('rr', 'rcf', 'full'):
            cls_in += rocket_dim
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes))

    # ------------------------------------------------------------------
    def forward(self, x, return_aux=False):
        B, _, L = x.shape

        # Transport
        T = self.transport_builder(x)
        T_e = self.transport_encoder(T)
        t_pooled = T_e.mean(dim=2)  # [B, C_T]

        # Inception backbone
        h = self.proj(x)
        H1 = self.block1(h)
        H2 = self.block2(H1)
        H3 = self.block3(H2)
        H4 = self.block4(H3)

        # Rocket response path (frozen bank -> learned projection)
        F_M_raw = self.rocket(x)                 # [B, 2F]
        F_M = self.rocket_proj(F_M_raw)          # [B, rocket_dim]

        # Regime
        v_feat = H4.mean(dim=2)                  # [B, D]
        if self.variant in ('rs', 'full'):
            z, u = self.regime_main_r(torch.cat([v_feat, F_M], dim=1))
        else:
            z, u = self.regime_main(v_feat)
        v = None
        s = None
        if self.use_velocity:
            v = self._compute_velocity(z, self.vel_linear)
            if self.variant == 'rv':
                q = self.vel_q(F_M)
                gq = torch.sigmoid(self.vel_gate(torch.cat([z, u, F_M], dim=1)))
                v = v + gq * q
            s = torch.norm(v, dim=1, keepdim=True)

        # rtr: conditional residual rocket modulation of H4
        if self.variant == 'rtr':
            g_rtr = torch.sigmoid(self.rtr_gate(torch.cat([v_feat, z, u, F_M], dim=1)))
            H4 = H4 + g_rtr.unsqueeze(2) * self.rtr_proj(F_M).unsqueeze(2)

        # ---- Fusion ----
        F_T = self.transport_proj(t_pooled)      # [B, D]
        F_R = self.regime_proj(z)                # [B, D]

        gate_in = [t_pooled, z]
        if self.use_velocity:
            gate_in.append(v)
        if self.use_uncertainty:
            gate_in.append(u)
        gate_cat = torch.cat(gate_in, dim=1)

        g_T = torch.sigmoid(self.gate_T(gate_cat))
        g_R = torch.sigmoid(self.gate_R(gate_cat))

        if self.use_interaction:
            I_T = self.interact_T(t_pooled)
            I_R = self.interact_R(z)
            lam_I = torch.sigmoid(self.lambda_I(z))
            I_TR = I_T * I_R
            if self.use_uncertainty:
                u_gate = torch.sigmoid(u.mean(dim=1, keepdim=True))
                I_TR = (1 - u_gate) * I_TR
        else:
            I_TR = None
            lam_I = 0.0

        # Per-variant fusion
        if self.variant == '3f':
            if self.fixed_alpha is not None:
                a3 = torch.full((B, 3), 1.0 / 3.0, device=x.device)
            else:
                a3 = F.softmax(self.alpha3(torch.cat([gate_cat, F_M], dim=1)), dim=1)
            F_M_D = self.FM_to_D(F_M)
            F_fused = a3[:, 0:1] * g_T * F_T + a3[:, 1:2] * g_R * F_R + \
                      a3[:, 2:3] * F_M_D
            if self.use_interaction:
                F_fused = F_fused + lam_I * I_TR
            alpha = a3
        elif self.variant in ('rcf', 'full'):
            if self.fixed_alpha is not None:
                a2 = torch.full((B, 2), 0.5, device=x.device)
            else:
                a2 = F.softmax(self.alpha2(torch.cat([gate_cat, F_M], dim=1)), dim=1)
            P_M = self.interact_M(F_M)
            I_TM = self.interact_TM(t_pooled) * P_M
            I_RM = self.interact_RM(z) * P_M
            lam_TM = torch.sigmoid(self.lambda_TM(F_M))
            lam_RM = torch.sigmoid(self.lambda_RM(F_M))
            if self.variant == 'rcf':
                F_fused = a2[:, 0:1] * (g_T * F_T + lam_TM * I_TM) + \
                          a2[:, 1:2] * (g_R * F_R + lam_RM * I_RM)
            else:
                F_fused = a2[:, 0:1] * g_T * F_T + a2[:, 1:2] * g_R * F_R + \
                          lam_TM * I_TM + lam_RM * I_RM
            if self.use_interaction:
                F_fused = F_fused + lam_I * I_TR
            alpha = a2
        else:
            if self.fixed_alpha is not None:
                alpha = torch.full((B, 1), float(self.fixed_alpha), device=x.device)
            else:
                alpha_in = torch.cat([gate_cat, g_T, g_R], dim=1)
                alpha = torch.sigmoid(self.alpha_net(alpha_in))
            F_fused = alpha * (g_T * F_T) + (1 - alpha) * (g_R * F_R)
            if self.use_interaction:
                F_fused = F_fused + lam_I * I_TR

        # Residual injection
        res_g = torch.sigmoid(self.res_gate(
            torch.cat([z] + ([v] if self.use_velocity else []) +
                      ([u] if self.use_uncertainty else []), dim=1)))
        H4 = H4 + res_g.unsqueeze(2) * self.res_proj(F_fused).unsqueeze(2)

        # FiLM modulation
        if self.use_uncertainty and self.use_regime:
            mod_cat = torch.cat([z] + ([v] if self.use_velocity else []), dim=1)
            film_out = self.film(mod_cat)
            gamma, beta = film_out.chunk(2, dim=1)
            u_gate2 = torch.sigmoid(self.film_gate(u))
            H4 = H4 * (1 + u_gate2.unsqueeze(2) * gamma.unsqueeze(2)) + \
                 u_gate2.unsqueeze(2) * beta.unsqueeze(2)

        # Global pooling
        h_gap = H4.mean(dim=2)
        h_gmp = H4.max(dim=2)[0]

        # Classifier input
        parts = [h_gap, h_gmp, t_pooled, z]
        if self.use_velocity:
            parts += [v, s]
        if self.use_uncertainty:
            parts.append(u)
        if self.use_interaction and I_TR is not None:
            parts.append(I_TR)
        g_M = None
        if self.variant in ('rr', 'rcf', 'full'):
            g_M = torch.sigmoid(self.gM(torch.cat([z, F_M], dim=1)))
            parts.append(g_M * F_M)
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
                "rocket_gate": g_M,
                "F_M": F_M,
            }
            return logits, aux
        return logits


if __name__ == "__main__":
    print("Verifying TURS-Rocket variants...")
    for variant in ['rr', 'rtr', '3f', 'rcf', 'rs', 'rv', 'full']:
        for L, nc in [(140, 5), (1024, 4)]:
            model = TURSRocket(in_channels=1, num_classes=nc, regime_dim=16,
                               variant=variant)
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            x = torch.randn(4, 1, L)
            logits, aux = model(x, return_aux=True)
            assert logits.shape == (4, nc), f"{variant} L{L}: {logits.shape}"
            assert not torch.isnan(logits).any()
            logits.sum().backward()
            no_grad = [nm for nm, p in model.named_parameters()
                       if p.requires_grad and p.grad is None]
            assert not no_grad, f"{variant} L{L}: no grad {no_grad}"
            print(f"  {variant:6s} L={L:4d} C={nc}: params={n:,} OK")
    print("All variants verified.")