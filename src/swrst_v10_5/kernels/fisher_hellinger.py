import torch

def compute_fisher_normalization_cache(Q_t: torch.Tensor, tau: float = 1e-8):
    """
    H5: Shared Fisher normalization cache.
    Q_t: [B, T, K, K]
    Returns:
        fisher_norm: [B, T, K, K]
    """
    return 1.0 / torch.sqrt(Q_t + tau)

def compute_fisher_hellinger_kernel(
    Xi_x: torch.Tensor, Xi_y: torch.Tensor,
    norm_x: torch.Tensor, norm_y: torch.Tensor,
    sigma_H: float = 1.0
):
    """
    Xi_x: [Nx, M, d] (where d is the packed dimension)
    Xi_y: [Ny, M, d]
    norm_x: [Nx, d]
    norm_y: [Ny, d]
    """
    # H5: Normalize Xi_tilde = Xi / sqrt(Q + tau)
    # We apply the packed norm cache
    Xi_tilde_x = Xi_x * norm_x.unsqueeze(1) # [Nx, M, d]
    Xi_tilde_y = Xi_y * norm_y.unsqueeze(1) # [Ny, M, d]
    
    # k_H = exp( - (1/sigma^2) * sum_d (sqrt(|Xi|) - sqrt(|Xi'|))^2 )
    sqrt_abs_x = torch.sqrt(torch.abs(Xi_tilde_x) + 1e-8) # [Nx, M, d]
    sqrt_abs_y = torch.sqrt(torch.abs(Xi_tilde_y) + 1e-8) # [Ny, M, d]
    
    dist = torch.cdist(sqrt_abs_x.view(Xi_x.size(0), -1), sqrt_abs_y.view(Xi_y.size(0), -1), p=2.0) ** 2
    # But wait, the kernel is computed over M, M'? No, MMD handles M and M' combinations!
    # MMD needs k(Xi_x^m, Xi_y^m') for all m, m'.
    # This means the kernel is evaluated for each (m, m') pair.
    
    Nx, M, d = Xi_tilde_x.shape
    Ny = Xi_tilde_y.shape[0]
    
    x_flat = sqrt_abs_x.reshape(Nx * M, d)
    y_flat = sqrt_abs_y.reshape(Ny * M, d)
    
    d2 = torch.cdist(x_flat, y_flat, p=2.0) ** 2 # [Nx*M, Ny*M]
    d2 = d2.view(Nx, M, Ny, M).permute(0, 2, 1, 3) # [Nx, Ny, M, M]
    
    k_H = torch.exp(- (1.0 / (sigma_H**2)) * d2)
    return k_H
