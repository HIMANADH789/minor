import torch

def compute_spectral_kernel(Phi1: torch.Tensor, Phi2: torch.Tensor):
    """
    Phi: [N, 32]
    """
    N1 = Phi1.size(0)
    N2 = Phi2.size(0)
    
    v1 = Phi1
    v2 = Phi2
    
    dist2 = torch.sum(v1**2, dim=-1, keepdim=True) + torch.sum(v2**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1, v2.T)
    dist2 = torch.clamp(dist2, min=0.0)
    
    sigma2 = torch.median(dist2).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_spec = torch.exp(-dist2 / sigma2)
    return K_spec
