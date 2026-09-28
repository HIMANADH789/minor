import torch
from src.swrst_v4.transport.weighted_field import pack_upper_triangular

def compute_unified_state(Xi_t: torch.Tensor, P_t: torch.Tensor, Q_t: torch.Tensor, K_t: torch.Tensor, h_t: torch.Tensor, tau: float = 1e-3):
    """
    Xi_t: [B, T-1, M, K, K]
    P_t: [B, T-1, M, K, K]
    Q_t: [B, T-1, K, K]
    K_t: [B, T-1, M]
    h_t: [B, T-1, M]
    
    Returns:
        Z_t: [B, T-1, M, 165]  (assuming K=12, M=7)
    """
    B, T_minus_1, M, K_dim, _ = Xi_t.shape
    Q_t_exp = Q_t.unsqueeze(2) # [B, T-1, 1, K, K]
    
    # 1. Fisher state (Correction A)
    F_num = P_t - Q_t_exp
    F_t = F_num / torch.sqrt(Q_t_exp + tau)
    F_t_packed = pack_upper_triangular(F_t) # [B, T-1, M, 78]
    
    # 2. Fisher residual (Boundary replicate)
    dF_t = torch.zeros_like(F_t)
    if T_minus_1 > 1:
        dF_t[:, 1:] = F_t[:, 1:] - F_t[:, :-1]
    dF_t_packed = pack_upper_triangular(dF_t) # [B, T-1, M, 78]
    
    # 3. Order persistence (U1)
    R_t = torch.zeros((B, T_minus_1, M), device=Xi_t.device, dtype=Xi_t.dtype)
    if T_minus_1 > 1:
        diff_Xi = Xi_t[:, 1:] - Xi_t[:, :-1]
        R_t[:, 1:] = torch.norm(diff_Xi, p='fro', dim=(-1, -2))
    R_t = R_t.unsqueeze(-1) # [B, T-1, M, 1]
    
    # 4. Cross-order interaction (Correction B)
    Xi_flat = Xi_t.reshape(B, T_minus_1, M, -1)
    C_mat = torch.matmul(Xi_flat, Xi_flat.transpose(-1, -2)) # [B, T-1, M, M]
    
    idx = ~torch.eye(M, dtype=torch.bool, device=C_mat.device)
    C_t = C_mat[..., idx].reshape(B, T_minus_1, M, M - 1) # [B, T-1, M, 6]
    
    # 5. Resolution curvature
    K_t_exp = K_t.unsqueeze(-1) # [B, T-1, M, 1]
    
    # 6. Entropy profile
    h_t_exp = h_t.unsqueeze(-1) # [B, T-1, M, 1]
    
    # Concatenate all
    Z_t = torch.cat([F_t_packed, dF_t_packed, R_t, C_t, K_t_exp, h_t_exp], dim=-1)
    
    return Z_t
