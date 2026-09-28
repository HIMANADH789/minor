import torch
from torch.amp import autocast

def compute_spectral_kernel(Phi1: torch.Tensor, Phi2: torch.Tensor, chunk_size: int = 256):
    N1 = Phi1.size(0)
    N2 = Phi2.size(0)
    M = Phi1.size(1)
    K_S = torch.zeros((N1, N2), device=Phi1.device, dtype=torch.float32)
    
    for m in range(M):
        v1 = Phi1[:, m, :].contiguous().view(N1, -1)
        v2 = Phi2[:, m, :].contiguous().view(N2, -1)
        
        # Normalize
        v1 = v1 / (torch.norm(v1, p=2, dim=-1, keepdim=True) + 1e-8)
        v2 = v2 / (torch.norm(v2, p=2, dim=-1, keepdim=True) + 1e-8)
        
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
                    
                    dist2 = torch.sum(v1_c**2, dim=-1, keepdim=True) + torch.sum(v2_c**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1_c, v2_c.T)
                    dist2 = torch.clamp(dist2, min=0.0)
                    
                    K_S[i:end_i, j:end_j] += torch.exp(-dist2.float() / max(torch.mean(dist2.float()).item(), 1e-8))

    K_S = K_S / M
    return K_S
