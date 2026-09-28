import torch

def compute_fisher_kernel(F_t1: torch.Tensor, F_t2: torch.Tensor, chunk_size: int = 256):
    """
    F_t: [N, T-1, M, K(K+1)//2]
    """
    N1 = F_t1.size(0)
    N2 = F_t2.size(0)
    
    # Flatten T and M
    v1 = F_t1.view(N1, -1)
    v2 = F_t2.view(N2, -1)
    
    K_F = torch.zeros((N1, N2), device=F_t1.device, dtype=torch.float32)
    
    # FP32 only for Fisher
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        v1_c = v1[i:end_i]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            v2_c = v2[j:end_j]
            
            dist2 = torch.sum(v1_c**2, dim=-1, keepdim=True) + torch.sum(v2_c**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1_c, v2_c.T)
            dist2 = torch.clamp(dist2, min=0.0)
            
            K_F[i:end_i, j:end_j] = dist2

    sigma2 = torch.median(K_F).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_F = torch.exp(-K_F / sigma2)
    return K_F

