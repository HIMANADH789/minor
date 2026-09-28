import torch

def compute_persistent_spectrum(Xi_t: torch.Tensor, K_top: int = 8):
    """
    Xi_t: [B, T-1, M, K, K]
    
    Returns:
        Phi_spec: [B, M, 4 * K_top]
        delta_lam: [B, T-1-1, M, K_top] (trend over time)
    """
    B, T_minus_1, M, K, _ = Xi_t.shape
    
    # H10 - Flatten [B,T,M] for all eigs
    Xi_flat = Xi_t.reshape(B * T_minus_1 * M, K, K)
    
    # G_t = Xi * Xi^T
    G_flat = torch.bmm(Xi_flat, Xi_flat.transpose(1, 2))
    
    # Symmetrize for eigvalsh (H8)
    G_sym = (G_flat + G_flat.transpose(1, 2)) / 2.0
    
    # torch.linalg.eigvalsh is faster and designed for symmetric matrices
    eigvals = torch.linalg.eigvalsh(G_sym) # [B*T*M, K]
    
    # Top-K
    lam_flat = torch.flip(eigvals[:, -K_top:], dims=[1])
    
    # Reshape back to [B, T-1, M, K_top]
    lam_t = lam_flat.view(B, T_minus_1, M, K_top)
    
    # Features aggregated over time (T_minus_1 dimension which is dim=1)
    mu_lam = torch.mean(lam_t, dim=1) # [B, M, K_top]
    var_lam = torch.var(lam_t, dim=1) # [B, M, K_top]
    max_lam = torch.max(lam_t, dim=1)[0] # [B, M, K_top]
    
    # Trend
    delta_lam = torch.zeros((B, T_minus_1 - 1, M, K_top), device=Xi_t.device, dtype=Xi_t.dtype)
    delta_lam.copy_(lam_t[:, 1:] - lam_t[:, :-1])
    trend_lam = torch.mean(delta_lam, dim=1) # [B, M, K_top]
    
    # Concatenate features
    Phi_spec = torch.cat([mu_lam, var_lam, max_lam, trend_lam], dim=-1) # [B, M, 4 * K_top]
    
    return Phi_spec, delta_lam


