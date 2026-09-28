import torch

def compute_signed_hellinger_kernel(
    D_x: torch.Tensor, D_y: torch.Tensor,
    norm_x: torch.Tensor, norm_y: torch.Tensor,
    sigma: float = 1.0
):
    """
    C3: Signed Hellinger. phi(x) = sgn(x) * sqrt(|x|)
    D_x: [Nx, M, d]
    D_y: [Ny, M, d]
    norm_x: [Nx, d]
    norm_y: [Ny, d]
    """
    D_tilde_x = D_x * norm_x.unsqueeze(1)
    D_tilde_y = D_y * norm_y.unsqueeze(1)
    
    phi_x = torch.sign(D_tilde_x) * torch.sqrt(torch.abs(D_tilde_x) + 1e-8)
    phi_y = torch.sign(D_tilde_y) * torch.sqrt(torch.abs(D_tilde_y) + 1e-8)
    
    Nx, M, d = phi_x.shape
    Ny = phi_y.shape[0]
    
    x_flat = phi_x.reshape(Nx*M, d)
    y_flat = phi_y.reshape(Ny*M, d)
    
    d2 = torch.cdist(x_flat, y_flat, p=2.0) ** 2
    d2 = d2.view(Nx, M, Ny, M).permute(0, 2, 1, 3) # [Nx, Ny, M, M]
    
    k = torch.exp(- (1.0 / (sigma**2)) * d2)
    return k
