import torch
import math

def compute_final_kernel(MMD2_t: torch.Tensor, w_t: torch.Tensor, sigma: float):
    """
    K(x,y) = exp( - 1/sigma^2 * sum_t w_t * MMD2_t )
    
    MMD2_t: [Nx, Ny, T]
    w_t: [Nx, T]
    sigma: float
    """
    # sum_t w_t * MMD2_t
    weighted_mmd2 = MMD2_t * w_t.unsqueeze(1) # [Nx, Ny, T]
    
    dist2 = torch.sum(weighted_mmd2, dim=2) # [Nx, Ny]
    
    K = torch.exp(- (1.0 / (sigma**2)) * dist2)
    return K, dist2

def compute_global_median_heuristic(dist2_train: torch.Tensor):
    """
    H8: Median heuristic cached.
    dist2_train: [N_train, N_train]
    Returns:
        sigma: float
    """
    # Exclude diagonal zeros
    N = dist2_train.shape[0]
    row, col = torch.triu_indices(N, N, offset=1)
    
    vals = dist2_train[row, col]
    if vals.numel() == 0:
        return 1.0
        
    median_dist = torch.median(vals).item()
    sigma = max(math.sqrt(median_dist), 1e-3)
    return sigma
