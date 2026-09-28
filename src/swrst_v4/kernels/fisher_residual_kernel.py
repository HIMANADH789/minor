import torch

def compute_fisher_residual(F_star: torch.Tensor):
    """
    F_star: [B, T-1, K, K] (routed Fisher field)
    
    Returns:
        dF_flat: [B, (T-2) * K^2]
    """
    B, T_minus_1, K_dim, _ = F_star.shape
    
    # Delta F_t = F_t - F_{t-1}
    # For t=1 to T-2
    dF_t = F_star[:, 1:] - F_star[:, :-1] # [B, T-2, K, K]
    
    # Flatten
    dF_flat = dF_t.reshape(B, -1) # [B, (T-2) * K^2]
    
    return dF_flat

def compute_fisher_residual_kernel(dF_x1: torch.Tensor, dF_x2: torch.Tensor, chunk_size: int = 256):
    """
    dF_x: [B, (T-2) * K^2]
    """
    N1 = dF_x1.size(0)
    N2 = dF_x2.size(0)
    
    K_dF = torch.zeros((N1, N2), device=dF_x1.device, dtype=torch.float32)
    
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        f1 = dF_x1[i:end_i]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            f2 = dF_x2[j:end_j]
            
            dist2 = torch.sum(f1**2, dim=-1, keepdim=True) + torch.sum(f2**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(f1, f2.T)
            dist2 = torch.clamp(dist2, min=0.0)
            
            K_dF[i:end_i, j:end_j] = dist2

    sigma2 = torch.median(K_dF).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_dF = torch.exp(-K_dF / sigma2)
    return K_dF
