import torch

def compute_energy_trajectory(Xi_t: torch.Tensor):
    """
    Xi_t: [B, T-1, M, K, K]
    
    Returns:
        e_t: [B, T-1, M] (Frobenius norm of Xi_t per order)
    """
    e_t = torch.norm(Xi_t, p='fro', dim=(-1, -2))
    return e_t

