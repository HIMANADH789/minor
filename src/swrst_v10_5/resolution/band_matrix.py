import torch

def precompute_resolution_band_matrix(eps_sorted: torch.Tensor):
    """
    B_mm' = exp(- 1/2 * (log eps_m - log eps_m')^2 )
    eps_sorted: [M]
    Returns:
        B: [M, M]
    """
    u = torch.log(eps_sorted)
    dist = (u.unsqueeze(1) - u.unsqueeze(0)) ** 2
    B = torch.exp(-0.5 * dist)
    return B
