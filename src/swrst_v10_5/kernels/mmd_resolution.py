import torch

def compute_mmd_resolution_self(k_self: torch.Tensor, w_sorted: torch.Tensor):
    """
    Computes XX or YY (Self MMD term).
    k_self: [N, M, M] (Self kernel evaluated across resolution pairs)
    w_sorted: [M]
    Returns:
        self_mmd: [N]
    """
    # XX = sum_{m,m'} w_m * w_m' * k(Xi^m, Xi^m')
    W = w_sorted.unsqueeze(1) * w_sorted.unsqueeze(0) # [M, M]
    
    # k_self has shape [N, M, M]
    weighted_k = k_self * W.unsqueeze(0)
    
    self_mmd = torch.sum(weighted_k, dim=(1, 2)) # [N]
    return self_mmd

def compute_mmd_resolution_cross(k_cross: torch.Tensor, w_sorted: torch.Tensor, B: torch.Tensor):
    """
    Computes XY (Cross MMD term).
    k_cross: [Nx, Ny, M, M]
    w_sorted: [M]
    B: [M, M] (Band Matrix)
    Returns:
        cross_mmd: [Nx, Ny]
    """
    W = w_sorted.unsqueeze(1) * w_sorted.unsqueeze(0) # [M, M]
    WB = W * B # [M, M]
    
    weighted_k = k_cross * WB.view(1, 1, W.size(0), W.size(1))
    
    cross_mmd = torch.sum(weighted_k, dim=(2, 3)) # [Nx, Ny]
    return cross_mmd

def compute_mmd_squared(XX: torch.Tensor, YY: torch.Tensor, XY: torch.Tensor):
    """
    XX: [Nx]
    YY: [Ny]
    XY: [Nx, Ny]
    Returns:
        MMD2: [Nx, Ny]
    """
    # MMD2 = XX - 2*XY + YY
    mmd2 = XX.unsqueeze(1) - 2.0 * XY + YY.unsqueeze(0)
    
    # Numerical stability
    return torch.clamp(mmd2, min=0.0)
