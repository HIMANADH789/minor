import torch
import torch.nn.functional as F

def compute_resolution_router(h_t: torch.Tensor, K_t: torch.Tensor, Xi_t: torch.Tensor, F_t: torch.Tensor, tau: float = 1.0):
    """
    h_t: [B, T-1, M]
    K_t: [B, T-1, M]
    Xi_t: [B, T-1, M, K, K]
    F_t: [B, T-1, M, K, K] (This is F_num = P_t - Q_t, or wait, the fisher field F_t?
          The fisher field F_t in fisher_field.py is F_num / sqrt(Q_t + tau).
          I should pass the normalized F_t to this, or compute the norm of F_t inside.)
    
    Returns:
        Xi_star: [B, T-1, K, K]
        F_star: [B, T-1, K, K]
        idx: [B, T-1] (the routed m_t^*)
    """
    B, T_minus_1, M, K, _ = Xi_t.shape
    
    # Fixed normalized coefficients
    a1 = 0.35
    a2 = 0.30
    a3 = 0.35
    
    # Compute norms
    norm_Xi = torch.norm(Xi_t, p='fro', dim=(-1, -2)) # [B, T-1, M]
    norm_F = torch.norm(F_t, p='fro', dim=(-1, -2)) # [B, T-1, M]
    
    # Fisher confidence routing boost (U6)
    # C_t^(m) = ||F_t^(m)||_F / (||Xi_t^(m)||_F + tau)
    C_t = norm_F / (norm_Xi + tau)
    
    # Scoring field
    Phi_t = a1 * h_t + a2 * K_t + a3 * C_t # [B, T-1, M]
    
    # Hard routing index
    idx = Phi_t.argmax(dim=2) # [B, T-1]
    
    # Batched gather for Xi_t and F_t
    idx_expand = idx[..., None, None, None].expand(-1, -1, 1, K, K)
    
    Xi_star = torch.gather(Xi_t, 2, idx_expand).squeeze(2) # [B, T-1, K, K]
    F_star = torch.gather(F_t, 2, idx_expand).squeeze(2) # [B, T-1, K, K]
    
    return Xi_star, F_star, idx
