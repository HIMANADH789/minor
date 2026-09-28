import torch

def compute_resolution_ot(C: torch.Tensor, p_x: torch.Tensor, p_y: torch.Tensor, epsilon: float = 0.1, num_iters: int = 10):
    """
    Solves OT across resolutions to find correspondence Gamma_t.
    
    C: [Nx, Ny, T, M, M]
    p_x: [Nx, M]
    p_y: [Ny, M]
    
    Returns:
        Gamma: [Nx, Ny, T, M, M]
    """
    Nx, Ny, T, M, _ = C.shape
    
    # H5: Flatten batch to [B, M, M] where B = Nx * Ny * T
    C_flat = C.reshape(-1, M, M) # [B, M, M]
    
    # Expand p_x and p_y to match flattened batch
    p_x_exp = p_x.view(Nx, 1, 1, M).expand(Nx, Ny, T, M).reshape(-1, M) # [B, M]
    p_y_exp = p_y.view(1, Ny, 1, M).expand(Nx, Ny, T, M).reshape(-1, M) # [B, M]
    
    # Log domain preparations
    log_a = torch.log(p_x_exp + 1e-8) # [B, M]
    log_b = torch.log(p_y_exp + 1e-8) # [B, M]
    
    K_log = -C_flat / epsilon # [B, M, M]
    
    u = torch.zeros_like(log_a) # [B, M]
    v = torch.zeros_like(log_b) # [B, M]
    
    # H7: Log-domain Sinkhorn
    for _ in range(num_iters):
        # Update u: u = log_a - logsumexp(K_log + v.unsqueeze(1), dim=2)
        # K_log: [B, M, M] (from, to)
        # v: [B, 1, M]
        val = K_log + v.unsqueeze(1)
        u = log_a - torch.logsumexp(val, dim=2)
        
        # Update v: v = log_b - logsumexp(K_log + u.unsqueeze(2), dim=1)
        val = K_log + u.unsqueeze(2)
        v = log_b - torch.logsumexp(val, dim=1)
        
    # Gamma = exp(u + K_log + v)
    log_Gamma = u.unsqueeze(2) + K_log + v.unsqueeze(1)
    Gamma_flat = torch.exp(log_Gamma)
    
    # Reshape back
    Gamma = Gamma_flat.view(Nx, Ny, T, M, M)
    
    return Gamma
