import torch

def compute_surprise_kernel(N1: torch.Tensor, N2: torch.Tensor):
    """
    N: [B, T-1, K, K]
    """
    N1_sz = N1.size(0)
    N2_sz = N2.size(0)
    
    v1 = N1.view(N1_sz, -1)
    v2 = N2.view(N2_sz, -1)
    
    dist2 = torch.sum(v1**2, dim=-1, keepdim=True) + torch.sum(v2**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1, v2.T)
    dist2 = torch.clamp(dist2, min=0.0)
    
    sigma2 = torch.median(dist2).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_surp = torch.exp(-dist2 / sigma2)
    return K_surp
