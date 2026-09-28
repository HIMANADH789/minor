import torch
from src.swrst_v4.transport.weighted_field import pack_upper_triangular

def compute_canonical_field(P_t: torch.Tensor, Q_t: torch.Tensor, rho_t: torch.Tensor, tau: float = 10**-8):
    """
    P_t: [B, T-1, M, K, K]
    Q_t: [B, T-1, K, K]
    rho_t: [B, T-1, M, K, K]
    
    Returns:
        Xi_can_packed: [B, T-1, M, K(K+1)/2] (H4 packed)
    """
    # H5: Precompute (Q_t + tau)^(-1/2) once
    inv_sqrt_Q = 1.0 / torch.sqrt(Q_t + tau) # [B, T-1, K, K]
    inv_sqrt_Q_expanded = inv_sqrt_Q.unsqueeze(2) # [B, T-1, 1, K, K]
    
    # H6: Precompute (rho_t + tau)^(-1/2)
    inv_sqrt_rho = 1.0 / torch.sqrt(rho_t + tau) # [B, T-1, M, K, K]
    
    # Correction 1: Factorized canonical field, H3 Fused
    # \Xi_can = (\rho / sqrt(\rho + \tau)) \odot ((P - Q) / sqrt(Q + \tau))
    term1 = rho_t * inv_sqrt_rho # [B, T-1, M, K, K]
    
    Q_t_expanded = Q_t.unsqueeze(2)
    term2 = (P_t - Q_t_expanded) * inv_sqrt_Q_expanded # [B, T-1, M, K, K]
    
    Xi_can = term1 * term2 # [B, T-1, M, K, K]
    
    # H4: Upper triangle packing
    Xi_can_packed = pack_upper_triangular(Xi_can)
    
    return Xi_can_packed
