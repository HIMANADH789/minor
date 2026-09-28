import torch
from torch.amp import autocast

def compute_resolution_entropy_kernel(h1: torch.Tensor, h2: torch.Tensor, chunk_size: int = 256):
    """
    h: [N, T-1, M]
    """
    N1 = h1.size(0)
    N2 = h2.size(0)
    
    # Flatten T and M
    v1 = h1.view(N1, -1)
    v2 = h2.view(N2, -1)
    
    K_H = torch.zeros((N1, N2), device=h1.device, dtype=torch.float32)
    
    with autocast('cuda', dtype=torch.bfloat16):
        v1_bf = v1.to(torch.bfloat16)
        v2_bf = v2.to(torch.bfloat16)
        
        # H12 - Chunk pairwise kernels
        for i in range(0, N1, chunk_size):
            end_i = min(i + chunk_size, N1)
            v1_c = v1_bf[i:end_i]
            
            for j in range(0, N2, chunk_size):
                end_j = min(j + chunk_size, N2)
                v2_c = v2_bf[j:end_j]
                
                # dist2 = ||v1 - v2||^2
                dist2 = torch.sum(v1_c**2, dim=-1, keepdim=True) + torch.sum(v2_c**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1_c, v2_c.T)
                dist2 = torch.clamp(dist2, min=0.0)
                
                K_H[i:end_i, j:end_j] = dist2.float()

    # Heuristic bandwidth (median)
    sigma2 = torch.median(K_H).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_H = torch.exp(-K_H / sigma2)
    return K_H

