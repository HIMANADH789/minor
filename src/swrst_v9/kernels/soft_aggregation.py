import torch

def compute_soft_aggregation(logK_t: torch.Tensor, alpha: float = 3.0):
    """
    logK_t: [Nx, Ny, T]
    Returns:
        K_xy: [Nx, Ny]
    """
    T = logK_t.shape[2]
    
    # C3: Time-averaged soft aggregation
    # K(x,y) = exp( 1/alpha * log( 1/T sum_t exp(alpha * logK_t) ) )
    
    # Let v_t = alpha * logK_t
    v_t = alpha * logK_t
    
    # logsumexp over t
    log_sum = torch.logsumexp(v_t, dim=2) # [Nx, Ny]
    
    # Average factor: log(1/T) = -log(T)
    log_mean = log_sum - torch.log(torch.tensor(float(T), device=logK_t.device))
    
    # Scale back by 1/alpha
    logK_xy = log_mean / alpha
    
    # Final exp
    K_xy = torch.exp(logK_xy)
    
    return K_xy
