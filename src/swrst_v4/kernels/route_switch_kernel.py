import torch

def compute_route_switch_entropy(idx: torch.Tensor):
    """
    idx: [B, T-1] (m_t^* sequence)
    
    Returns:
        R_x: [B, 1]
    """
    # Count how often router changes order: 1[m_t^* != m_{t-1}^*]
    diff = idx[:, 1:] != idx[:, :-1]
    R_x = torch.sum(diff, dim=1, keepdim=True).float() # [B, 1]
    
    return R_x

def compute_route_switch_kernel(R_x1: torch.Tensor, R_x2: torch.Tensor, chunk_size: int = 256):
    """
    R_x: [B, 1]
    """
    N1 = R_x1.size(0)
    N2 = R_x2.size(0)
    
    K_R = torch.zeros((N1, N2), device=R_x1.device, dtype=torch.float32)
    
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        r1 = R_x1[i:end_i]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            r2 = R_x2[j:end_j]
            
            dist2 = (r1 - r2.T)**2
            
            K_R[i:end_i, j:end_j] = dist2

    sigma2 = torch.median(K_R).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_R = torch.exp(-K_R / sigma2)
    return K_R
