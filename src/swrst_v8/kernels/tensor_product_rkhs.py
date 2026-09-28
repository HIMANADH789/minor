import torch

def compute_sigma2_m(Z_train: torch.Tensor, chunk_size: int = 128):
    """
    Z_train: [N, T-1, M, D]
    Returns:
        sigma2_m: [M]
    """
    N, T, M, D = Z_train.shape
    
    # We will sample to compute median to avoid memory explosion
    num_samples = min(N, 1000)
    indices = torch.randperm(N)[:num_samples]
    Z_sample = Z_train[indices] # [S, T, M, D]
    
    sigma2_m = torch.zeros(M, device=Z_train.device, dtype=torch.float32)
    
    for m in range(M):
        Zm = Z_sample[:, :, m, :].reshape(num_samples, -1) # [S, T*D]
        # Pairwise distance
        dist = torch.cdist(Zm, Zm, p=2.0)**2 # [S, S]
        sigma2_m[m] = torch.median(dist)
        if sigma2_m[m] < 1e-8:
            sigma2_m[m] = 1.0
            
    return sigma2_m

def compute_tensor_product_kernel(
    Z_1: torch.Tensor, Z_2: torch.Tensor,
    p_1: torch.Tensor, p_2: torch.Tensor,
    sigma2_m: torch.Tensor,
    tau: float = 1e-8,
    chunk_size: int = 128
):
    """
    Z_1: [N1, T-1, M, D]
    Z_2: [N2, T-1, M, D]
    p_1: [N1, M]
    p_2: [N2, M]
    
    Returns:
        K_mat: [N1, N2]
    """
    N1 = Z_1.size(0)
    N2 = Z_2.size(0)
    T = Z_1.size(1)
    M = Z_1.size(2)
    D = Z_1.size(3)
    
    logK_mat = torch.zeros((N1, N2), device=Z_1.device, dtype=torch.float32)
    
    # Pre-square norms to compute pairwise distances
    Z1_sq = torch.sum(Z_1**2, dim=(1, 3)) # [N1, M]
    Z2_sq = torch.sum(Z_2**2, dim=(1, 3)) # [N2, M]
    
    # H8: Shared sigma_xy cache
    dist_p = torch.cdist(p_1, p_2, p=1.0) # [N1, N2]
    sigma0_2 = torch.median(dist_p)
    if sigma0_2 < 1e-8:
        sigma0_2 = 1.0
        
    sigma2_xy = sigma0_2 * torch.exp(0.5 * dist_p) # [N1, N2]
    
    # H6: Chunk tensor kernel over N1, N2
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        
        Z1_i = Z_1[i:end_i] # [chunk1, T, M, D]
        Z1_sq_i = Z1_sq[i:end_i] # [chunk1, M]
        p1_i = p_1[i:end_i] # [chunk1, M]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            
            Z2_j = Z_2[j:end_j] # [chunk2, T, M, D]
            Z2_sq_j = Z2_sq[j:end_j] # [chunk2, M]
            p2_j = p_2[j:end_j] # [chunk2, M]
            
            sigma2_xy_ij = sigma2_xy[i:end_i, j:end_j] # [chunk1, chunk2]
            
            # logK += local_logK for each m
            logK_chunk = torch.zeros((end_i - i, end_j - j), device=Z_1.device, dtype=torch.float32)
            
            for m in range(M):
                # distance for m
                Z1_m = Z1_i[:, :, m, :].reshape(end_i - i, -1) # [chunk1, T*D]
                Z2_m = Z2_j[:, :, m, :].reshape(end_j - j, -1) # [chunk2, T*D]
                
                # Using pre-squared norms
                dot_m = torch.matmul(Z1_m, Z2_m.T) # [chunk1, chunk2]
                d2_m = Z1_sq_i[:, m].unsqueeze(1) + Z2_sq_j[:, m].unsqueeze(0) - 2.0 * dot_m
                d2_m = torch.clamp(d2_m, min=0.0)
                
                # Local val
                val = d2_m / (sigma2_m[m] * sigma2_xy_ij + tau)
                
                # Exact log to avoid underflow and PSD-destroying distance clipping
                log_k_m = -val
                
                # U4: Localized tensor product weighted by p(m)
                # p_m combination: symmetric weight (p1 * p2)^0.5
                p_m_ij = torch.sqrt(p1_i[:, m].unsqueeze(1) * p2_j[:, m].unsqueeze(0)) # [chunk1, chunk2]
                
                logK_chunk += p_m_ij * log_k_m
                
            logK_mat[i:end_i, j:end_j] = logK_chunk
            
    K_mat = torch.exp(logK_mat)
    
    return K_mat
