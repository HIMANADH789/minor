import torch
from src.swrst.transport.deterministic_ot import compute_cost_matrix

def get_resolution_grid():
    """
    Returns the explicitly defined multi-resolution spectrum (epsilon grid).
    M=7 log-spaced values spanning raw to smooth transport.
    """
    return [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

def precompute_em(K, device):
    """
    Precomputes E_m = exp(-C / eps_m) for all m in the resolution grid.
    Caches the tensor E_m of shape [M, K, K] on the specified device.
    """
    epsilons = get_resolution_grid()
    M = len(epsilons)
    
    C_numpy = compute_cost_matrix(K)
    C = torch.tensor(C_numpy, dtype=torch.float32, device=device)
    
    eps_tensor = torch.tensor(epsilons, dtype=torch.float32, device=device)
    
    # C is [K, K], eps_tensor is [M]
    # We want E_m to be [M, K, K]
    C_expanded = C.unsqueeze(0).expand(M, K, K)
    eps_expanded = eps_tensor.view(M, 1, 1)
    
    E_m = torch.exp(-C_expanded / eps_expanded)
    
    # Return both C and E_m
    return C, E_m

