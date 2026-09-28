import torch

def compute_resolution_persistence(Xi_packed: torch.Tensor):
    """
    Xi_packed: [B, T-1, M, K(K-1)/2]
    
    Returns:
        R_t: [B, T-1, M-1] (Upgrade U1: Raw magnitude continuity)
    """
    diff_Xi = Xi_packed[:, :, 1:] - Xi_packed[:, :, :-1] # [B, T-1, M-1, K(K-1)/2]
    
    # H7-style Frobenius norm on packed symmetric matrix with 0 diagonal
    R_t = torch.sqrt(torch.tensor(2.0, device=Xi_packed.device)) * torch.norm(diff_Xi, p=2, dim=-1) # [B, T-1, M-1]
    
    return R_t
