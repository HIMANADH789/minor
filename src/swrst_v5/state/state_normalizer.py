import torch

def normalize_block(block: torch.Tensor, eps: float = 1e-8):
    """L2 normalize a block across its feature dimension. Skip if it's a scalar."""
    if block.size(-1) == 1:
        return block # DO NOT normalize scalars, they would just become 1.0!
    return block / (torch.norm(block, p=2, dim=-1, keepdim=True) + eps)

def compute_state_normalizer(Z_bar_t: torch.Tensor, Z_t: torch.Tensor):
    """
    Z_bar_t: [B, T-1, 165] (Continuous barycenter)
    Z_t: [B, T-1, M, 165] (Raw states, used to compute covariance)
    
    Returns:
        Z_hat_t: [B, T-1, 330]
    """
    # 1. Normalize Barycenter blocks
    # Format: [F_t (78), dF_t (78), R_t (1), C_t (6), K_t (1), h_t (1)]
    F_bar = Z_bar_t[..., 0:78]
    dF_bar = Z_bar_t[..., 78:156]
    R_bar = Z_bar_t[..., 156:157]
    C_bar = Z_bar_t[..., 157:163]
    K_bar = Z_bar_t[..., 163:164]
    h_bar = Z_bar_t[..., 164:165]
    
    F_norm = normalize_block(F_bar)
    dF_norm = normalize_block(dF_bar)
    R_norm = normalize_block(R_bar)
    C_norm = normalize_block(C_bar)
    K_norm = normalize_block(K_bar)
    h_norm = normalize_block(h_bar)
    
    Z_bar_norm = torch.cat([F_norm, dF_norm, R_norm, C_norm, K_norm, h_norm], dim=-1) # [B, T-1, 165]
    
    # 2. State Covariance (U2)
    Sigma_t = torch.var(Z_t, dim=2) # [B, T-1, 165]
    
    # Normalize Sigma_t blockwise identically
    F_var = Sigma_t[..., 0:78]
    dF_var = Sigma_t[..., 78:156]
    R_var = Sigma_t[..., 156:157]
    C_var = Sigma_t[..., 157:163]
    K_var = Sigma_t[..., 163:164]
    h_var = Sigma_t[..., 164:165]
    
    F_var_norm = normalize_block(F_var)
    dF_var_norm = normalize_block(dF_var)
    R_var_norm = normalize_block(R_var)
    C_var_norm = normalize_block(C_var)
    K_var_norm = normalize_block(K_var)
    h_var_norm = normalize_block(h_var)
    
    Sigma_norm = torch.cat([F_var_norm, dF_var_norm, R_var_norm, C_var_norm, K_var_norm, h_var_norm], dim=-1) # [B, T-1, 165]
    
    # 3. Append Diag (U2)
    Z_hat_t = torch.cat([Z_bar_norm, Sigma_norm], dim=-1) # [B, T-1, 330]
    
    return Z_hat_t
