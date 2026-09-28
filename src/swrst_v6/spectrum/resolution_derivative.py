import torch
from src.swrst_v4.transport.weighted_field import pack_upper_triangular

def compute_resolution_derivative(Xi_t: torch.Tensor, Q_t: torch.Tensor, inv_du_m: torch.Tensor, tau: float = 1e-4):
    """
    Xi_t: [B, T-1, M, K, K]
    Q_t: [B, T-1, K, K]
    inv_du_m: [M-1]
    
    Returns:
        V_packed: [B, T-1, M-1, K(K-1)/2] (H12: Packed upper triangle)
        norm_V: [B, T-1, M-1] (H7: Fused Frobenius norm)
    """
    # H11: Fuse derivative + Fisher normalization in one pass
    # Q_t is broadcasted across M
    Q_t_expanded = Q_t.unsqueeze(2) # [B, T-1, 1, K, K]
    fisher_norm = 1.0 / torch.sqrt(Q_t_expanded + tau) # [B, T-1, 1, K, K]
    
    Xi_fisher = Xi_t * fisher_norm # [B, T-1, M, K, K]
    
    diff_Xi = Xi_fisher[:, :, 1:] - Xi_fisher[:, :, :-1] # [B, T-1, M-1, K, K]
    
    # H13: Precompute inverse du multiplication
    # inv_du_m has shape [M-1], broadcast to [B, T-1, M-1, K, K]
    inv_du_expanded = inv_du_m.view(1, 1, -1, 1, 1)
    
    V_t = diff_Xi * inv_du_expanded # [B, T-1, M-1, K, K]
    
    # H12: Pack symmetric upper triangle
    V_packed = pack_upper_triangular(V_t) # [B, T-1, M-1, K(K-1)/2]
    
    # H7: Fused V norm computation
    # ||V||_F = sqrt(2) * ||V_packed||_2 because V is symmetric and has 0 diagonal
    norm_V = torch.sqrt(torch.tensor(2.0, device=V_t.device)) * torch.norm(V_packed, p=2, dim=-1) # [B, T-1, M-1]
    
    return V_packed, norm_V
