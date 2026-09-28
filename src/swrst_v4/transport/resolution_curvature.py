import torch

def compute_resolution_curvature(Xi_t: torch.Tensor):
    """
    Xi_t: [B, T-1, M, K, K]
    
    Returns:
        K_t: [B, T-1, M] (Resolution curvature)
    """
    B, T_minus_1, M, K_dim, _ = Xi_t.shape
    
    K_t = torch.zeros((B, T_minus_1, M), dtype=Xi_t.dtype, device=Xi_t.device)
    
    if M >= 1:
        # m=0 -> Xi_t(0) - 2*Xi_t(0) + Xi_t(0) = 0
        K_t[:, :, 0] = 0.0
        
    if M >= 2:
        # m=1 -> Xi_t(1) - 2*Xi_t(0) + Xi_t(0) = Xi_t(1) - Xi_t(0)
        diff_1 = Xi_t[:, :, 1] - Xi_t[:, :, 0]
        K_t[:, :, 1] = torch.norm(diff_1, p='fro', dim=(-1, -2))
        
    if M >= 3:
        # m >= 2
        diff_m = Xi_t[:, :, 2:] - 2 * Xi_t[:, :, 1:-1] + Xi_t[:, :, :-2]
        K_t[:, :, 2:] = torch.norm(diff_m, p='fro', dim=(-1, -2))
        
    return K_t

