import torch

def compute_spectral_persistence_kernel(
    Xi_x: torch.Tensor, Xi_y: torch.Tensor,
    norm_x: torch.Tensor, norm_y: torch.Tensor,
    sigma: float = 1.0, tau: float = 1e-8
):
    """
    C4: Fisher-normalized Xi -> G = Xi_tilde @ Xi_tilde^T
    U4: Multi-band spectral persistence (mean, variance, trend of eigenvalues)
    
    Xi_x: [Nx, M, K, K] (Note: We need the full un-packed matrix here to do G)
    """
    # H5: Fisher normalized Xi
    # Wait, the input to this function must be the full unpacked Xi!
    # Because packing destroys the non-symmetric or full matrix structure needed for Xi Xi^T.
    # We will compute it.
    Xi_tilde_x = Xi_x * norm_x.unsqueeze(1)
    Xi_tilde_y = Xi_y * norm_y.unsqueeze(1)
    
    G_x = torch.matmul(Xi_tilde_x, Xi_tilde_x.transpose(-1, -2))
    G_y = torch.matmul(Xi_tilde_y, Xi_tilde_y.transpose(-1, -2))
    
    # H8: Batch eigvals
    # G is symmetric positive semi-definite
    eig_x = torch.linalg.eigvalsh(G_x) # [Nx, M, K]
    eig_y = torch.linalg.eigvalsh(G_y) # [Ny, M, K]
    
    def extract_stats(eig):
        mean_eig = eig.mean(dim=-1, keepdim=True)
        var_eig = eig.var(dim=-1, unbiased=False, keepdim=True)
        trend_eig = (eig[..., -1:] - eig[..., 0:1]) # max - min
        return torch.cat([mean_eig, var_eig, trend_eig], dim=-1) # [N, M, 3]
        
    stats_x = extract_stats(eig_x) # [Nx, M, 3]
    stats_y = extract_stats(eig_y) # [Ny, M, 3]
    
    Nx, M, _ = stats_x.shape
    Ny = stats_y.shape[0]
    
    x_flat = stats_x.reshape(Nx*M, 3)
    y_flat = stats_y.reshape(Ny*M, 3)
    
    d2 = torch.cdist(x_flat, y_flat, p=2.0) ** 2
    d2 = d2.view(Nx, M, Ny, M).permute(0, 2, 1, 3) # [Nx, Ny, M, M]
    
    k = torch.exp(- (1.0 / (sigma**2)) * d2)
    return k
