import torch

def extract_spectral_signatures(Xi_t: torch.Tensor, K_top=8):
    """
    Xi_t: [B, T-1, K, K]
    """
    B, T_minus_1, K, _ = Xi_t.shape
    
    G_t = torch.matmul(Xi_t, Xi_t.transpose(-1, -2))
    # PSD symmetrize
    G_sym = (G_t + G_t.transpose(-1, -2)) / 2.0
    
    eigvals = torch.linalg.eigvalsh(G_sym)
    lam_t = torch.flip(eigvals[..., -K_top:], dims=[-1])
    
    mu_lam = torch.mean(lam_t, dim=1)
    var_lam = torch.var(lam_t, dim=1)
    max_lam = torch.max(lam_t, dim=1)[0]
    
    delta_lam = torch.zeros((B, T_minus_1 - 1, K_top), device=Xi_t.device, dtype=Xi_t.dtype)
    delta_lam.copy_(lam_t[:, 1:] - lam_t[:, :-1])
    trend_lam = torch.mean(delta_lam, dim=1)
    
    Phi_spec = torch.cat([mu_lam, var_lam, max_lam, trend_lam], dim=-1)
    return Phi_spec
