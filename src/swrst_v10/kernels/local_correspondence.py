import torch
import math

def compute_sigma_m(S_train: torch.Tensor, chunk_size: int = 128):
    """
    H8: Reuse sigma_m cached.
    S_train: [N, T, M, d]
    Returns:
        sigma_m: [M]
    """
    N, T, M, d = S_train.shape
    
    num_samples = min(N, 1000)
    indices = torch.randperm(N)[:num_samples]
    S_sample = S_train[indices] # [S, T, M, d]
    
    sigma_m = torch.zeros(M, device=S_train.device, dtype=torch.float32)
    
    for m in range(M):
        Sm = S_sample[:, :, m, :].reshape(num_samples, -1) # [S, T*d]
        dist = torch.cdist(Sm, Sm, p=2.0) # Linear distance
        sigma_m[m] = torch.median(dist)
        if sigma_m[m] < 1e-8:
            sigma_m[m] = 1.0
            
    return sigma_m

def precompute_S_mn(sigma_m: torch.Tensor):
    """
    sigma_m: [M]
    Returns:
        S_mn: [M, M]
    """
    return sigma_m.unsqueeze(1) * sigma_m.unsqueeze(0)

def compute_sigma_xy(p_x: torch.Tensor, p_y: torch.Tensor):
    """
    C3: Dimensionless sigma_xy = 1 + 0.3 ||p_x - p_y||_1
    p_x: [Nx, M]
    p_y: [Ny, M]
    Returns:
        sigma_xy: [Nx, Ny]
    """
    dist_p = torch.cdist(p_x, p_y, p=1.0) # [Nx, Ny]
    sigma_xy = 1.0 + 0.30 * dist_p
    return sigma_xy

def compute_profile_confidence(p: torch.Tensor):
    """
    U4: Profile entropy confidence
    p: [N, M]
    """
    tau = 1e-8
    M = p.shape[1]
    H_p = -torch.sum(p * torch.log(p + tau), dim=1) # [N]
    c = 1.0 - H_p / math.log(M)
    return torch.clamp(c, min=0.0, max=1.0)

def compute_local_correspondence_kernel(
    S_x: torch.Tensor, S_y: torch.Tensor, 
    Gamma_t: torch.Tensor, 
    S_mn: torch.Tensor, 
    sigma_xy: torch.Tensor,
    c_x: torch.Tensor, c_y: torch.Tensor
):
    """
    S_x: [Nx, T, M, d]
    S_y: [Ny, T, M, d]
    Gamma_t: [Nx, Ny, T, M, M]
    S_mn: [M, M]
    sigma_xy: [Nx, Ny]
    c_x: [Nx]
    c_y: [Ny]
    
    Returns:
        K_t: [Nx, Ny, T]
    """
    Nx, T, M, d = S_x.shape
    Ny = S_y.shape[0]
    
    S1_sq = torch.sum(S_x**2, dim=-1) # [Nx, T, M]
    S2_sq = torch.sum(S_y**2, dim=-1) # [Ny, T, M]
    
    S_x_flat = S_x.permute(1, 0, 2, 3).reshape(T, Nx*M, d) # [T, Nx*M, d]
    S_y_flat = S_y.permute(1, 0, 2, 3).reshape(T, Ny*M, d) # [T, Ny*M, d]
    
    dot_flat = torch.bmm(S_x_flat, S_y_flat.transpose(1, 2)) # [T, Nx*M, Ny*M]
    dot = dot_flat.view(T, Nx, M, Ny, M).permute(1, 3, 0, 2, 4) # [Nx, Ny, T, M, M]
    
    d2 = S1_sq.unsqueeze(1).unsqueeze(4) + S2_sq.unsqueeze(0).unsqueeze(3) - 2.0 * dot # [Nx, Ny, T, M, M]
    d2 = torch.clamp(d2, min=0.0)
    
    # C3: denominator = sigma_m * sigma_n * sigma_xy
    denom = S_mn.view(1, 1, 1, M, M) * sigma_xy.view(Nx, Ny, 1, 1, 1) + 1e-8
    val = d2 / denom
    
    # H6: Log-domain kernel accumulation not used since we do not logsumexp anymore
    # The kernel here is K_t = sum_m,n Gamma * exp(-val)
    kernel_terms = Gamma_t * torch.exp(-val)
    K_t = torch.sum(kernel_terms, dim=(3, 4)) # [Nx, Ny, T]
    
    # U4: Confidence interpolation
    # K_t <- c_x c_y K_t + (1 - c_x c_y) K_bar_t
    c_xy = c_x.unsqueeze(1) * c_y.unsqueeze(0) # [Nx, Ny]
    c_xy = c_xy.unsqueeze(-1) # [Nx, Ny, 1]
    
    K_bar_t = torch.mean(K_t, dim=2, keepdim=True) # [Nx, Ny, 1]
    
    K_t_final = c_xy * K_t + (1.0 - c_xy) * K_bar_t
    
    return K_t_final
