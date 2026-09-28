import torch

def compute_transport_kernel(Xi_tilde_1: torch.Tensor, Xi_tilde_2: torch.Tensor, chunk_size: int = 512):
    """
    Xi_tilde: [B, T-1, K(K-1)/2]
    
    Returns:
        K_Xi: [N1, N2]
    """
    N1 = Xi_tilde_1.size(0)
    N2 = Xi_tilde_2.size(0)
    
    # Hellinger mapping
    A_full = torch.sqrt(torch.abs(Xi_tilde_1) + 1e-8)
    B_full = torch.sqrt(torch.abs(Xi_tilde_2) + 1e-8)
    
    # Flatten across time and features (H14 - Chunk over T and N)
    A_full = A_full.reshape(N1, -1) # [N1, D]
    B_full = B_full.reshape(N2, -1) # [N2, D]
    
    D_mat = torch.zeros((N1, N2), device=Xi_tilde_1.device, dtype=torch.float32)
    
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
        
    K_Xi = torch.exp(-D_mat / sigma2_median)
    
    return K_Xi
