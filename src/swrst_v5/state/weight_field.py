import torch

def compute_weight_field(Z_bar_t: torch.Tensor, Z_hat_t: torch.Tensor):
    """
    Z_bar_t: [B, T-1, 165] (Used to compute the dynamic weighting)
    
    Returns:
        W_t: [B, T-1, 330] (Same dimensionality as Z_hat_t)
    """
    # Slice base components
    F_bar = Z_bar_t[..., 0:78]
    dF_bar = Z_bar_t[..., 78:156]
    R_bar = Z_bar_t[..., 156:157]
    C_bar = Z_bar_t[..., 157:163]
    K_bar = Z_bar_t[..., 163:164]
    h_bar = Z_bar_t[..., 164:165]
    
    # Weight per feature component (Correction D)
    W_F = torch.abs(F_bar)
    W_dF = 0.20 * torch.abs(dF_bar)
    W_R = 0.10 * torch.abs(R_bar) # Standard uniform prior for missing weight
    W_C = 0.30 * torch.abs(C_bar)
    W_K = 0.20 * torch.abs(K_bar)
    W_h = 0.30 * torch.abs(h_bar)
    
    W_base = torch.cat([W_F, W_dF, W_R, W_C, W_K, W_h], dim=-1) # [B, T-1, 165]
    
    # Mirror the weights for the Covariance block (U2)
    W_cov = W_base.clone()
    
    W_t = torch.cat([W_base, W_cov], dim=-1) # [B, T-1, 330]
    
    # Normalize the weight field
    W_t_norm = W_t / (torch.norm(W_t, p=2, dim=-1, keepdim=True) + 1e-8)
    
    return W_t_norm
