import torch

def compute_path_aggregation(K_t: torch.Tensor, S_x: torch.Tensor, tau: float = 1e-8):
    """
    C4: Causal temporal weights with geometric path aggregation.
    
    K_t: [Nx, Ny, T]
    S_x: [Nx, T, M, d]
    
    Returns:
        K: [Nx, Ny]
    """
    Nx, Ny, T = K_t.shape
    
    # Compute w_t = ||S_t - S_{t-1}||
    # S_x is [Nx, T, M, d]
    diff = S_x[:, 1:, :, :] - S_x[:, :-1, :, :] # [Nx, T-1, M, d]
    norm_diff = torch.norm(diff.reshape(Nx, T-1, -1), dim=-1) # [Nx, T-1]
    
    # Normalize over t=1...T-1
    sum_diff = torch.sum(norm_diff, dim=1, keepdim=True) + tau
    w_t_nonzero = norm_diff / sum_diff # [Nx, T-1]
    
    # Pad w_0 = 0
    w_0 = torch.zeros((Nx, 1), device=S_x.device, dtype=torch.float32)
    w_t = torch.cat([w_0, w_t_nonzero], dim=1) # [Nx, T]
    
    # Final: log K = sum_t w_t log K_t
    w_t_exp = w_t.unsqueeze(1) # [Nx, 1, T]
    
    # Ensure K_t > 0 for log
    log_K_t = torch.log(torch.clamp(K_t, min=1e-12)) # [Nx, Ny, T]
    
    logK = torch.sum(w_t_exp * log_K_t, dim=2) # [Nx, Ny]
    
    K = torch.exp(logK) # [Nx, Ny]
    
    return K
