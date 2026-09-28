import torch
from torch.amp import autocast

def compute_energy_trajectory_kernel(e_t1: torch.Tensor, e_t2: torch.Tensor, chunk_size: int = 256):
    """
    e_t: [N, T-1, M]
    """
    N1 = e_t1.size(0)
    N2 = e_t2.size(0)
    
    M = e_t1.size(2)
    K_E = torch.zeros((N1, N2), device=e_t1.device, dtype=torch.float32)
    
    for m in range(M):
        v1 = e_t1[:, :, m].contiguous().view(N1, -1)
        v2 = e_t2[:, :, m].contiguous().view(N2, -1)
        
        # Normalize to prevent magnitude outliers
        v1 = v1 / (torch.norm(v1, p=2, dim=-1, keepdim=True) + 1e-8)
        v2 = v2 / (torch.norm(v2, p=2, dim=-1, keepdim=True) + 1e-8)
        
        with autocast('cuda', dtype=torch.bfloat16):
            v1_bf = v1.to(torch.bfloat16)
            v2_bf = v2.to(torch.bfloat16)
            
            dist2 = torch.sum(v1_bf**2, dim=-1, keepdim=True) + torch.sum(v2_bf**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1_bf, v2_bf.T)
            dist2 = torch.clamp(dist2, min=0.0).float()
            
        sigma2 = torch.mean(dist2).item()
        if sigma2 < 1e-8:
            sigma2 = 1.0
            
        K_E += torch.exp(-dist2 / sigma2)
        
    K_E = K_E / M
    return K_E


