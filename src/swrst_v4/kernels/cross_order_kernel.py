import torch

def compute_cross_order_interactions(Xi_t: torch.Tensor):
    """
    Xi_t: [B, T-1, M, K, K]
    
    Returns:
        C_t_flat: [B, (T-1) * 21]  (since M=7, 7*6/2 = 21 pairs)
    """
    B, T_minus_1, M, K_dim, _ = Xi_t.shape
    
    # Flatten K, K
    Xi_flat = Xi_t.reshape(B, T_minus_1, M, K_dim * K_dim) # [B, T-1, M, K^2]
    
    # Compute inner product for all m < n
    # We can do this with matmul
    # [B, T-1, M, K^2] x [B, T-1, K^2, M] -> [B, T-1, M, M]
    C_matrix = torch.matmul(Xi_flat, Xi_flat.transpose(-1, -2))
    
    # Extract upper triangular elements (m < n)
    # triu_indices excludes diagonal if offset=1
    row_idx, col_idx = torch.triu_indices(M, M, offset=1)
    
    # Extract pairs
    C_pairs = C_matrix[:, :, row_idx, col_idx] # [B, T-1, 21]
    
    # Flatten over time
    C_t_flat = C_pairs.reshape(B, -1) # [B, (T-1) * 21]
    
    return C_t_flat

def compute_cross_order_kernel(C_x1: torch.Tensor, C_x2: torch.Tensor, chunk_size: int = 256):
    """
    C_x: [B, (T-1) * 21]
    """
    N1 = C_x1.size(0)
    N2 = C_x2.size(0)
    
    K_C = torch.zeros((N1, N2), device=C_x1.device, dtype=torch.float32)
    
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        c1 = C_x1[i:end_i]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            c2 = C_x2[j:end_j]
            
            dist2 = torch.sum(c1**2, dim=-1, keepdim=True) + torch.sum(c2**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(c1, c2.T)
            dist2 = torch.clamp(dist2, min=0.0)
            
            K_C[i:end_i, j:end_j] = dist2

    sigma2 = torch.median(K_C).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_C = torch.exp(-K_C / sigma2)
    return K_C
