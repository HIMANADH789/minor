import torch

def compute_profile_kernel(
    D_x: torch.Tensor, D_y: torch.Tensor,
    w_sorted: torch.Tensor,
    sigma: float = 1.0
):
    """
    U3: Resolution entropy profile kernel.
    D_x: [Nx, T, M, d] (where d is packed dim)
    D_y: [Ny, T, M, d]
    w_sorted: [M]
    """
    # p_x(m) = w_m * sum_t ||D_t(m)||
    norm_D_x = torch.norm(D_x, dim=-1) # [Nx, T, M]
    norm_D_y = torch.norm(D_y, dim=-1) # [Ny, T, M]
    
    p_x = w_sorted.unsqueeze(0) * torch.sum(norm_D_x, dim=1) # [Nx, M]
    p_y = w_sorted.unsqueeze(0) * torch.sum(norm_D_y, dim=1) # [Ny, M]
    
    # We want K_p(x, y, m, m') = exp( - (p_x(m) - p_y(m'))^2 / sigma^2 )
    p_x_exp = p_x.unsqueeze(2).unsqueeze(3) # [Nx, M, 1, 1]
    p_y_exp = p_y.unsqueeze(0).unsqueeze(3) # [1, Ny, M, 1]
    
    # Actually it's [Nx, Ny, M, M]
    # We can do this efficiently
    p_x_mat = p_x.view(D_x.size(0), 1, -1, 1) # [Nx, 1, M, 1]
    p_y_mat = p_y.view(1, D_y.size(0), 1, -1) # [1, Ny, 1, M]
    
    d2 = (p_x_mat - p_y_mat) ** 2 # [Nx, Ny, M, M]
    
    k = torch.exp(- (1.0 / (sigma**2)) * d2)
    return k
