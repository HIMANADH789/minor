import torch
from src.swrst_v4.transport.weighted_field import pack_upper_triangular

def compute_fisher_field(F_num: torch.Tensor, Q_t: torch.Tensor, tau: float = 1e-3):
    """
    F_num: [B, T-1, M, K, K] (P_t - Q_t)
    Q_t: [B, T-1, K, K]
    
    Returns:
        F_t: [B, T-1, M, K, K]
        F_t_packed: [B, T-1, M, K(K+1)//2]
    """
    # Q_t is independent of M, expand it
    Q_t_exp = Q_t.unsqueeze(2)
    
    # Correction 3: F_t = (P - Q) / sqrt(Q + tau)
    # This is Fisher-normalized deviation
    F_t = F_num / torch.sqrt(Q_t_exp + tau)
    
    # H11 - Triangular pack before kernel
    F_t_packed = pack_upper_triangular(F_t)
    
    return F_t, F_t_packed

