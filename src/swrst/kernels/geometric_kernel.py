import torch

def compute_geometric_kernel(log_G1: torch.Tensor, log_G2: torch.Tensor):
    """
    log_G: [N, T-1, K, K]
    """
    N1 = log_G1.size(0)
    N2 = log_G2.size(0)
    
    v1 = log_G1.view(N1, -1)
    v2 = log_G2.view(N2, -1)
    
    dist2 = torch.sum(v1**2, dim=-1, keepdim=True) + torch.sum(v2**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1, v2.T)
    dist2 = torch.clamp(dist2, min=0.0)
    
    sigma2 = torch.median(dist2).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_geom = torch.exp(-dist2 / sigma2)
    return K_geom
