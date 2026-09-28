import torch

def compute_locality_score_chunk(Gamma_t: torch.Tensor):
    """
    Gamma_t: [Nx, Ny, T, M, M]
    Returns:
        L_xy_t: [Nx, Ny, T]
    """
    Nx, Ny, T, M, _ = Gamma_t.shape
    
    m_idx = torch.arange(M, device=Gamma_t.device).unsqueeze(1)
    n_idx = torch.arange(M, device=Gamma_t.device).unsqueeze(0)
    dist = torch.abs(m_idx - n_idx).float() # [M, M]
    
    # sum_{m,n} Gamma * |m-n|
    spread = Gamma_t * dist.view(1, 1, 1, M, M)
    L_xy_t = torch.sum(spread, dim=(3, 4)) # [Nx, Ny, T]
    
    return L_xy_t
