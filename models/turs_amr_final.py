"""
TURS-AMR — FINAL FROZEN ARCHITECTURE
Transport–Uncertainty Regime Network with Adaptive Multiscale Response Dynamics

One unified model (no variants). Combines the strongest mechanisms discovered
in the TURS research arc while avoiding redundancy:

  X → [Transport T=[X,Q,D]] + [Inception-lite backbone H] + [Fixed multiscale response bank R]
  R → regime-conditioned multiplicative modulation (kernels stay FIXED)
  Ṙ → response dynamics ΔR (+ compact Δ²R inside ONE transition encoder) → q
  adjacent-pair cross-scale coordination → c
  Regime: (z, u) = E_Z([H_pool, F_M, q, c])          (response as regime sensor)
  Velocity: v = W·z + g_v ⊙ P_v(q)                    (RV mechanism, inherited)
  Fusion: F = α·g_T·F_T + (1-α)·g_R·F_R + λ_I·I_TR + λ_M·I_MR   (TURS principle)
  Residual injection + uncertainty-gated FiLM on H4
  Classifier: [GAP, GMP, t_pooled, z, v, s, u, I_TR, g_M·F_M, q, c]

Explicitly EXCLUDED (redundant per final spec):
  - adaptive kernel residuals (W_eff = W + ΔW)
  - separate acceleration branch
  - adaptive scale selector
  - second response pathway / duplicate classifiers
  - MiniROCKET as second classifier

Frozen design constants (selected once, fixed across datasets):
  - response bank: 6 scales × 4 filters = 24 filters, kernels {7,9,13,19,25,35},
    dilations {1,2}, stats {PPV, MPV} (MiniROCKET-style, matches TURS-RV bank)
  - rocket_dim=32, regime_dim=16, D=64, transport C_T=64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tursnet import (TURSNet, RegimeEncoder, TransportBuilder,
                             TransportEncoder, TURSLoss, InceptionBlock)


# ============================================================
# Fixed Multiscale Response Bank (Rocket-inspired, frozen)
# ============================================================
class ResponseBank(nn.Module):
    """Fixed diverse random convolution bank, MiniROCKET-style init.

    6 scale groups × 4 filters = 24 filters. Weights N(0,1), biases U(-1,1),
    frozen. Each response summarized by PPV + MPV (matches the bank that
    produced the TURS-RV result, keeping comparability).
    """

    def __init__(self, in_ch=1, kernels=(7, 9, 13, 19, 25, 35),
                 dilations=(1, 2), filters_per_scale=4, seed=42):
        super().__init__()
        self.kernels = kernels
        self.dilations = dilations
        self.filters_per_scale = filters_per_scale
        self.n_scales = len(kernels)
        self.convs = nn.ModuleList()
        cfgs = [(k, d) for k in kernels for d in dilations]
        saved = torch.initial_seed()
        torch.manual_seed(seed)
        for (k, d) in cfgs:
            for _ in range(filters_per_scale):
                conv = nn.Conv1d(in_ch, 1, k, padding='same',
                                 dilation=d, bias=True)
                with torch.no_grad():
                    conv.weight.normal_(0.0, 1.0)
                    conv.bias.uniform_(-1.0, 1.0)
                for p in conv.parameters():
                    p.requires_grad = False
                self.convs.append(conv)
        torch.manual_seed(saved)
        self.n_total = len(self.convs)
        self.out_dim = self.n_total * 2  # PPV + MPV per filter
        # scale group index per filter (dilation-major order matches cfgs loop)
        self.scale_of = []
        for si, k in enumerate(kernels):
            for _ in dilations:
                for _ in range(filters_per_scale):
                    self.scale_of.append(si)

    def forward(self, x):
        """x: [B, 1, L] → R_raw: [B, F, L] (per-filter responses)"""
        rs = [conv(x).squeeze(1) for conv in self.convs]
        return torch.stack(rs, dim=1)

    def describe(self, R):
        """R: [B, F, L] → [B, F, 2] (PPV, MPV)"""
        ppv = (R > 0).float().mean(dim=2)
        mpv = F.relu(R).mean(dim=2)
        return torch.stack([ppv, mpv], dim=2)

    def flat(self, R):
        """R: [B, F, L] → [B, 2F] (PPV block then MPV block, TURS-RV order)"""
        d = self.describe(R)  # [B, F, 2]
        return torch.cat([d[..., 0], d[..., 1]], dim=1)

    def scale_features(self, R):
        """Per-scale-group descriptors: list of [B, fs*2]"""
        fs = self.filters_per_scale
        return [self.flat(R[:, si * fs * 2 // 2:(si + 1) * fs, :])
                if False else
                torch.cat([(R[:, si * fs:(si + 1) * fs] > 0).float().mean(dim=2),
                           F.relu(R[:, si * fs:(si + 1) * fs]).mean(dim=2)], dim=1)
                for si in range(self.n_scales)]


# ============================================================
# Response Transition Encoder (unified dynamics pathway)
# ============================================================
class ResponseTransitionEncoder(nn.Module):
    """q = E_R(R, ΔR, Δ²R) — ONE unified encoder for response dynamics.

    Δ²R is computed internally and compactly; no separate branch.
    Input: per-filter descriptors of R, ΔR, Δ²R + cross-scale vector c.
    """

    def __init__(self, flat_dim, cs_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(flat_dim * 3 + cs_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, desc_R, desc_dR, desc_ddR, c):
        return self.net(torch.cat([desc_R, desc_dR, desc_ddR, c], dim=1))


# ============================================================
# Cross-Scale Coordination (adjacent pairs only — compact)
# ============================================================
class CrossScaleCoordination(nn.Module):
    """C = W_s[e_s ⊙ e_{s+1}] over adjacent scale pairs (S-1 pairs).

    e_s: [B, e_dim] per-scale projections; interaction is elementwise product
    projected back to e_dim, then concatenated over pairs.
    """

    def __init__(self, n_scales, e_dim=8):
        super().__init__()
        self.n_scales = n_scales
        self.e_dim = e_dim
        self.proj = nn.ModuleList([nn.Linear(8, e_dim, bias=False)
                                   for _ in range(n_scales)])
        self.mix = nn.Linear((n_scales - 1) * e_dim, e_dim)

    def forward(self, scale_feats):
        """scale_feats: list of [B, fs*2] → [B, e_dim]"""
        e = [self.proj[i](F.adaptive_avg_pool1d(
            sf.unsqueeze(1), 8).squeeze(1)) for i, sf in enumerate(scale_feats)]
        pairs = [e[i] * e[i + 1] for i in range(self.n_scales - 1)]
        return self.mix(torch.cat(pairs, dim=1))


# ============================================================
# FINAL TURS-AMR
# ============================================================
class TURSAMRFinal(nn.Module):
    """Transport–Uncertainty Regime Network with Adaptive Multiscale
    Response Dynamics — the single frozen final model.

    Pipeline:
      1. Transport path (eTAI-style, unchanged): T=[X,Q,D] → T_e → t_pooled
      2. Temporal path (Inception-lite, unchanged): x → H4
      3. Fixed response bank: R_raw = Bank(x)
      4. Regime-conditioned response modulation: R̃ = R ⊙ (1 + g(z_pre))
         (multiplicative gate per filter; kernels remain FIXED)
      5. Response dynamics: ΔR, Δ²R (internal) → q = E_R(R̃, ΔR, Δ²R, c)
      6. Cross-scale coordination: c from adjacent scale pairs
      7. Regime: (z, u) = E_Z([H4_pool, F_M, q, c])
      8. Velocity: v = W·z + g_v ⊙ P_v(q)  (Rocket-informed, RV mechanism)
      9. TURS fusion: F = α·g_T·F_T + (1-α)·g_R·F_R + λ_I·I_TR + λ_M·I_MR
     10. Residual injection + uncertainty-gated FiLM on H4
     11. Classifier on [GAP, GMP, t_pooled, z, v, s, u, I_TR, g_M·F_M, q, c]
    """

    def __init__(self, in_channels=1, num_classes=5, regime_dim=16,
                 rocket_dim=32, dropout=0.1, seed=42, **kw):
        super().__init__()
        self.num_classes = num_classes
        self.regime_dim = regime_dim
        self.rocket_dim = rocket_dim

        base_ch = 32
        D = base_ch * 2  # 64
        self.D = D

        # --- 1. Transport (identical to TURS-Lite) ---
        self.transport_builder = TransportBuilder()
        self.transport_encoder = TransportEncoder(out_ch=16, final_ch=D)
        self.C_T = D

        # --- 2. Inception backbone (identical to TURS-Lite) ---
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, 1, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True))
        self.block1 = InceptionBlock(base_ch, base_ch)
        self.block2 = InceptionBlock(base_ch, base_ch)
        self.block3 = InceptionBlock(base_ch, D)
        self.block4 = InceptionBlock(D, D)

        # --- 3. Fixed response bank ---
        self.bank = ResponseBank(in_ch=in_channels, seed=seed)
        F_n = self.bank.n_total
        flat_dim = self.bank.out_dim  # 2F

        # --- 4. Regime-conditioned multiplicative response modulation ---
        # preliminary regime embedding from [H4_pool, F_M] only (compact pre-stage;
        # plain MLP — uncertainty is only meaningful at the main regime stage)
        self.regime_pre = nn.Sequential(
            nn.Linear(D + rocket_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, regime_dim))
        self.resp_gate = nn.Sequential(
            nn.Linear(regime_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, F_n))

        # --- 5+6. Unified transition encoder + cross-scale coordination ---
        self.cs = CrossScaleCoordination(self.bank.n_scales, e_dim=8)
        self.trans_enc = ResponseTransitionEncoder(flat_dim, 8, rocket_dim)

        # F_M projection (modulated response → rocket_dim)
        self.resp_proj = nn.Sequential(
            nn.Linear(flat_dim, 48), nn.ReLU(inplace=True),
            nn.Linear(48, rocket_dim))

        # --- 7. Main regime encoder: sees [H4_pool, F_M, q, c] ---
        self.regime_main = RegimeEncoder(D + rocket_dim * 2 + 8, regime_dim)

        # --- 8. Rocket-informed velocity (RV mechanism) ---
        self.vel_linear = nn.Linear(regime_dim, regime_dim)
        self.vel_q = nn.Linear(rocket_dim, regime_dim)
        self.vel_gate = nn.Sequential(
            nn.Linear(regime_dim + regime_dim + rocket_dim, 16),
            nn.ReLU(inplace=True), nn.Linear(16, 1))

        # --- 9. TURS fusion (identical structure to TURS-Lite) ---
        self.transport_proj = nn.Linear(self.C_T, D, bias=False)
        self.regime_proj = nn.Linear(regime_dim, D, bias=False)
        gate_in_dim = self.C_T + regime_dim * 3
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
        # Rocket-mediated regime interaction I_MR = P_R(z) ⊙ P_M(q)
        self.interact_M = nn.Linear(rocket_dim, D, bias=False)
        self.lambda_M = nn.Sequential(
            nn.Linear(rocket_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

        # --- 10. Residual injection + FiLM (identical to TURS-Lite) ---
        self.res_gate = nn.Sequential(
            nn.Linear(regime_dim * 2, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
        self.res_proj = nn.Linear(D, D, bias=False)
        self.film = nn.Sequential(
            nn.Linear(regime_dim * 2, D), nn.ReLU(inplace=True),
            nn.Linear(D, D * 2))
        self.film_gate = nn.Sequential(
            nn.Linear(regime_dim, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

        # --- Uncertainty-gated rocket contribution to classifier ---
        self.gM_net = nn.Sequential(
            nn.Linear(regime_dim + rocket_dim, 16),
            nn.ReLU(inplace=True), nn.Linear(16, 1))

        # --- 11. Classifier ---
        cls_in = D * 2 + self.C_T + regime_dim + regime_dim + 1 + regime_dim + D
        cls_in += rocket_dim          # g_M · F_M
        cls_in += rocket_dim          # q
        cls_in += 8                   # c (cross-scale)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x, return_aux=False):
        B, _, L = x.shape

        # === 1. Transport ===
        T = self.transport_builder(x)
        T_e = self.transport_encoder(T)
        t_pooled = T_e.mean(dim=2)  # [B, C_T]

        # === 2. Temporal backbone ===
        h = self.proj(x)
        H1 = self.block1(h)
        H2 = self.block2(H1)
        H3 = self.block3(H2)
        H4 = self.block4(H3)
        h_pool = H4.mean(dim=2)  # [B, D]

        # === 3. Fixed response bank ===
        R_raw = self.bank(x)  # [B, F, L]
        F_M_raw = self.bank.flat(R_raw)  # [B, 2F]
        F_M = self.resp_proj(F_M_raw)  # [B, rocket_dim]

        # === 4. Regime-conditioned response modulation (kernels FIXED) ===
        z_pre = self.regime_pre(torch.cat([h_pool, F_M], dim=1))
        g_resp = torch.sigmoid(self.resp_gate(z_pre))  # [B, F]
        R_mod = R_raw * (1.0 + g_resp.unsqueeze(2))    # [B, F, L]

        # === 5. Response dynamics (unified) ===
        F_M_mod = self.resp_proj(self.bank.flat(R_mod))
        dR = R_mod[:, :, 1:] - R_mod[:, :, :-1]
        ddR = dR[:, :, 1:] - dR[:, :, :-1]
        desc_R = self.bank.flat(R_mod)
        desc_dR = self.bank.flat(dR) if dR.shape[2] > 0 else \
            torch.zeros(B, self.bank.out_dim, device=x.device)
        desc_ddR = self.bank.flat(ddR) if ddR.shape[2] > 0 else \
            torch.zeros(B, self.bank.out_dim, device=x.device)

        # === 6. Cross-scale coordination ===
        scale_feats = self.bank.scale_features(R_mod)
        c = self.cs(scale_feats)  # [B, 8]

        # === unified transition encoding ===
        q = self.trans_enc(desc_R, desc_dR, desc_ddR, c)  # [B, rocket_dim]

        # === 7. Main regime ===
        z, u = self.regime_main(torch.cat([h_pool, F_M_mod, q, c], dim=1))

        # === 8. Rocket-informed velocity ===
        v = self.vel_linear(z)
        q_v = self.vel_q(q)
        g_v = torch.sigmoid(self.vel_gate(torch.cat([z, u, q], dim=1)))
        v = v + g_v * q_v
        s = torch.norm(v, dim=1, keepdim=True)

        # === 9. TURS fusion ===
        F_T = self.transport_proj(t_pooled)
        F_R = self.regime_proj(z)
        gate_in = torch.cat([t_pooled, z, v, u], dim=1)
        g_T = torch.sigmoid(self.gate_T(gate_in))
        g_R = torch.sigmoid(self.gate_R(gate_in))
        alpha = torch.sigmoid(self.alpha_net(
            torch.cat([gate_in, g_T, g_R], dim=1)))
        F_fused = alpha * (g_T * F_T) + (1 - alpha) * (g_R * F_R)

        I_T = self.interact_T(t_pooled)
        I_R = self.interact_R(z)
        lam_I = torch.sigmoid(self.lambda_I(z))
        I_TR = I_T * I_R
        u_gate = torch.sigmoid(u.mean(dim=1, keepdim=True))
        I_TR = (1 - u_gate) * I_TR

        # Rocket-mediated regime interaction: P_M(q) gates regime evidence
        P_M = self.interact_M(q)
        I_MR = I_R * P_M
        lam_M = torch.sigmoid(self.lambda_M(q))
        F_fused = F_fused + lam_I * I_TR + lam_M * I_MR

        # === 10. Residual injection + FiLM ===
        res_g = torch.sigmoid(self.res_gate(torch.cat([z, v], dim=1)))
        H4 = H4 + res_g.unsqueeze(2) * self.res_proj(F_fused).unsqueeze(2)

        film_out = self.film(torch.cat([z, v], dim=1))
        gamma, beta = film_out.chunk(2, dim=1)
        u_gate2 = torch.sigmoid(self.film_gate(u))
        H4 = H4 * (1 + u_gate2.unsqueeze(2) * gamma.unsqueeze(2)) + \
            u_gate2.unsqueeze(2) * beta.unsqueeze(2)

        # === 11. Classifier ===
        h_gap = H4.mean(dim=2)
        h_gmp = H4.max(dim=2)[0]
        g_M = torch.sigmoid(self.gM_net(torch.cat([z, F_M_mod], dim=1)))
        parts = [h_gap, h_gmp, t_pooled, z, v, s, u, I_TR, g_M * F_M_mod, q, c]
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
                "F_M": F_M_mod,
                "response_q": q,
                "cross_scale": c,
                "response_gate": g_resp,
            }
            return logits, aux
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("TURS-AMR FINAL — Shape Verification")
    print("=" * 60)
    for L, nc in [(140, 5), (1024, 4)]:
        model = TURSAMRFinal(in_channels=1, num_classes=nc, regime_dim=16)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x = torch.randn(4, 1, L)
        logits, aux = model(x, return_aux=True)
        assert logits.shape == (4, nc)
        assert not torch.isnan(logits).any()
        logits.sum().backward()
        print(f"  L={L:4d} C={nc}: params={n:,} OK")
    print("Done.")
