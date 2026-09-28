import torch
from torch.amp import autocast

def compute_spectral_persistence_kernel(delta_lam1: torch.Tensor, delta_lam2: torch.Tensor, chunk_size: int = 256):
    """
    delta_lam: [N, T-2, M, K_top]
    """
    N1 = delta_lam1.size(0)
    N2 = delta_lam2.size(0)
    
    # Flatten T, M, K_top
    v1 = delta_lam1.view(N1, -1)
    v2 = delta_lam2.view(N2, -1)
    
    K_P = torch.zeros((N1, N2), device=delta_lam1.device, dtype=torch.float32)
    
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
                
                K_P[i:end_i, j:end_j] = dist2.float()

    sigma2 = torch.median(K_P).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_P = torch.exp(-K_P / sigma2)
    return K_P

