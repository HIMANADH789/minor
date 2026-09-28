import torch

def compute_order_variance(Xi_t: torch.Tensor):
    """
    Xi_t: [B, T-1, M, K, K]
    
    Returns:
        V_x: [B, M]
    """
    # Compute Frobenius norm over K, K
    norm_Xi = torch.norm(Xi_t, p='fro', dim=(-1, -2)) # [B, T-1, M]
    
    # Compute variance over time (T-1)
    # Using unbiased variance
    V_x = torch.var(norm_Xi, dim=1, unbiased=True) # [B, M]
    
    # If T-1 <= 1, variance could be NaN. Just in case, clamp to 0.
    V_x = torch.nan_to_num(V_x, nan=0.0)
    
    return V_x

def compute_order_variance_kernel(V_x1: torch.Tensor, V_x2: torch.Tensor, chunk_size: int = 256):
    """
    V_x: [B, M]
    """
    N1 = V_x1.size(0)
    N2 = V_x2.size(0)
    
    K_V = torch.zeros((N1, N2), device=V_x1.device, dtype=torch.float32)
    
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        v1_c = V_x1[i:end_i]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            v2_c = V_x2[j:end_j]
            
            # Squared Euclidean distance
            dist2 = torch.sum(v1_c**2, dim=-1, keepdim=True) + torch.sum(v2_c**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(v1_c, v2_c.T)
            dist2 = torch.clamp(dist2, min=0.0)
            
            K_V[i:end_i, j:end_j] = dist2

    sigma2 = torch.median(K_V).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_V = torch.exp(-K_V / sigma2)
    return K_V
