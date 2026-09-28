import torch
import torch.nn.functional as F

def compute_sigma2_m(S_train: torch.Tensor, chunk_size: int = 128):
    """
    S_train: [N, T, M, d]
    Returns:
        sigma2_m: [M]
    """
    N, T, M, d = S_train.shape
    
    num_samples = min(N, 1000)
    indices = torch.randperm(N)[:num_samples]
    S_sample = S_train[indices] # [S, T, M, d]
    
    sigma2_m = torch.zeros(M, device=S_train.device, dtype=torch.float32)
    
    for m in range(M):
        Sm = S_sample[:, :, m, :].reshape(num_samples, -1) # [S, T*d]
        dist = torch.cdist(Sm, Sm, p=2.0)**2
        sigma2_m[m] = torch.median(dist)
        if sigma2_m[m] < 1e-8:
            sigma2_m[m] = 1.0
            
    return sigma2_m

def precompute_S_mn(sigma2_m: torch.Tensor):
    """
    H6: Precompute sigma_m * sigma_n matrix once.
    sigma2_m: [M]
    Returns:
        S_mn: [M, M]
    """
    sigma_m = torch.sqrt(sigma2_m)
    return sigma_m.unsqueeze(1) * sigma_m.unsqueeze(0)

def compute_correspondence_kernel_chunk(
    S_x: torch.Tensor, S_y: torch.Tensor, 
    Gamma_t: torch.Tensor, 
    S_mn: torch.Tensor, 
    sigma2_xy: torch.Tensor
):
    """
    S_x: [Nx, T, M, d]
    S_y: [Ny, T, M, d]
    Gamma_t: [Nx, Ny, T, M, M]
    S_mn: [M, M]
    sigma2_xy: [Nx, Ny]
    
    Returns:
        logK_t: [Nx, Ny, T]
    """
    Nx, T, M, d = S_x.shape
    Ny = S_y.shape[0]
    
    # Pre-square norms to compute pairwise distances
    S1_sq = torch.sum(S_x**2, dim=-1) # [Nx, T, M]
    S2_sq = torch.sum(S_y**2, dim=-1) # [Ny, T, M]
    
    # H4 style: bmm for pairwise dot products
    S_x_flat = S_x.permute(1, 0, 2, 3).reshape(T, Nx*M, d) # [T, Nx*M, d]
    S_y_flat = S_y.permute(1, 0, 2, 3).reshape(T, Ny*M, d) # [T, Ny*M, d]
    
    dot_flat = torch.bmm(S_x_flat, S_y_flat.transpose(1, 2)) # [T, Nx*M, Ny*M]
    dot = dot_flat.view(T, Nx, M, Ny, M).permute(1, 3, 0, 2, 4) # [Nx, Ny, T, M, M]
    
    # d^2 = ||S_x(m) - S_y(n)||^2
    d2 = S1_sq.unsqueeze(1).unsqueeze(4) + S2_sq.unsqueeze(0).unsqueeze(3) - 2.0 * dot # [Nx, Ny, T, M, M]
    d2 = torch.clamp(d2, min=0.0)
    
    # val = d^2 / (sigma_m^2 * sigma_n^2 * sigma_xy^2)
    # sigma2_xy: [Nx, Ny, 1, 1, 1]
    # S_mn: [1, 1, 1, M, M]
    denom = S_mn.view(1, 1, 1, M, M) * sigma2_xy.view(Nx, Ny, 1, 1, 1) + 1e-8
    val = d2 / denom
    
    # H9: Log-domain kernel accumulation
    # log(Gamma_t * exp(-val)) = log(Gamma_t) - val
    log_Gamma = torch.log(Gamma_t + 1e-12)
    log_terms = log_Gamma - val # [Nx, Ny, T, M, M]
    
    # logsumexp over m and n
    log_terms_flat = log_terms.reshape(Nx, Ny, T, M*M)
    logK_t = torch.logsumexp(log_terms_flat, dim=-1) # [Nx, Ny, T]
    
    return logK_t
