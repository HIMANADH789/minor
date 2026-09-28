import torch

def compute_adaptive_fusion(
    K_Xi: torch.Tensor,
    K_p: torch.Tensor,
    K_V: torch.Tensor,
    K_A: torch.Tensor,
    K_F: torch.Tensor,
    K_eig: torch.Tensor,
    p_1: torch.Tensor,
    p_2: torch.Tensor,
    M: int
):
    """
    Fuses the 6 kernels adaptively in the log-domain.
    p_1: [N1, T-1, M-1]
    p_2: [N2, T-1, M-1]
    
    Returns:
        K_fused: [N1, N2]
    """
    # Compute gamma for p_1 and p_2
    m_idx = torch.arange(1, M, device=p_1.device, dtype=torch.float32) / M # [M-1]
    
    p_mean_1 = torch.mean(p_1, dim=1) # [N1, M-1]
    gamma_1 = torch.sum(p_mean_1 * m_idx.unsqueeze(0), dim=1, keepdim=True) # [N1, 1]
    
    p_mean_2 = torch.mean(p_2, dim=1) # [N2, M-1]
    gamma_2 = torch.sum(p_mean_2 * m_idx.unsqueeze(0), dim=1, keepdim=True) # [N2, 1]
    
    # Cross pairwise mean of gamma
    gamma_xy = 0.5 * (gamma_1 + gamma_2.T) # [N1, N2]
    
    w1 = 0.35
    w2 = 0.25 * (1.0 - gamma_xy)
    w3 = 0.20 * gamma_xy
    w4 = 0.20
    
    # For the new U2 and U3 kernels, we assign weights based on their importance
    wF = 0.35 # Fisher is strongest
    w_eig = 0.20 # Anisotropy
    
    # Log domain fusion
    log_K = (
        w1 * torch.log(K_Xi + 1e-12) +
        w2 * torch.log(K_p + 1e-12) +
        w3 * torch.log(K_V + 1e-12) +
        w4 * torch.log(K_A + 1e-12) +
        wF * torch.log(K_F + 1e-12) +
        w_eig * torch.log(K_eig + 1e-12)
    )
    
    # Upgrade U4: Stability-preserving fusion floor
    log_K = torch.clamp(log_K, min=-25.0)
    
    K_fused = torch.exp(log_K)
    
    return K_fused
