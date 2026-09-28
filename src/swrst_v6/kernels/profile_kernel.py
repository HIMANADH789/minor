import torch

def compute_profile_kernel(p_1: torch.Tensor, R_1: torch.Tensor, p_2: torch.Tensor, R_2: torch.Tensor, chunk_size: int = 512):
    """
    p: [B, T-1, M-1]
    R: [B, T-1, M-1] (Upgrade U1)
    
    Returns:
        K_p: [N1, N2]
    """
    N1 = p_1.size(0)
    N2 = p_2.size(0)
    
    # Hellinger mapping for p
    sqrt_p_1 = torch.sqrt(torch.abs(p_1) + 1e-8)
    sqrt_p_2 = torch.sqrt(torch.abs(p_2) + 1e-8)
    
    # Concatenate p and R for the profile kernel (Upgrade U1)
    feat_1 = torch.cat([sqrt_p_1, R_1], dim=-1) # [N1, T-1, 2(M-1)]
    feat_2 = torch.cat([sqrt_p_2, R_2], dim=-1) # [N2, T-1, 2(M-1)]
    
    # Flatten across time and features (H14)
    A_full = feat_1.reshape(N1, -1) # [N1, D]
    B_full = feat_2.reshape(N2, -1) # [N2, D]
    
    D_mat = torch.zeros((N1, N2), device=p_1.device, dtype=torch.float32)
    
    # H4: Chunk Pairwise Kernels
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        A = A_full[i:end_i]
        
        A2_sum = torch.sum(A**2, dim=-1, keepdim=True) # [chunk1, 1]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            B = B_full[j:end_j]
            
            B2_sum = torch.sum(B**2, dim=-1).unsqueeze(0) # [1, chunk2]
            
            # dist2 = ||A||^2 + ||B||^2 - 2 A @ B.T
            dist2 = A2_sum + B2_sum - 2.0 * torch.matmul(A, B.T)
            dist2 = torch.clamp(dist2, min=0.0)
            
            D_mat[i:end_i, j:end_j] = dist2
            
    sigma2_median = torch.median(D_mat).item()
    if sigma2_median < 1e-8:
        sigma2_median = 1.0
        
    K_p = torch.exp(-D_mat / sigma2_median)
    
    return K_p
