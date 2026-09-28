import torch

def compute_unified_metric_and_kernel(Z_hat_1: torch.Tensor, W_1: torch.Tensor, h_1: torch.Tensor,
                                     Z_hat_2: torch.Tensor, W_2: torch.Tensor, h_2: torch.Tensor,
                                     chunk_size: int = 512):
    """
    Z_hat: [B, T-1, 330]
    W: [B, T-1, 330]
    h: [B, T-1, M] (Raw entropy to compute temporal variance per sample)
    
    Returns:
        K_unified: [N1, N2]
    """
    N1 = Z_hat_1.size(0)
    N2 = Z_hat_2.size(0)
    
    # Use Z_hat directly. The Fisher components are already in the tangent space.
    A_full = Z_hat_1.reshape(N1, -1) # [N1, D]
    B_full = Z_hat_2.reshape(N2, -1) # [N2, D]
    X_full = W_1.reshape(N1, -1) # [N1, D]
    Y_full = W_2.reshape(N2, -1) # [N2, D]
    
    D_mat = torch.zeros((N1, N2), device=Z_hat_1.device, dtype=torch.float32)
    
    # Chunked Pairwise Metric (H4)
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        A = A_full[i:end_i]
        X = X_full[i:end_i]
        
        A2 = A ** 2
        XA = X * A
        XA2_sum = torch.sum(X * A2, dim=-1, keepdim=True) # [chunk1, 1]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            B = B_full[j:end_j]
            Y = Y_full[j:end_j]
            
            B2 = B ** 2
            YB = Y * B
            YB2_sum = torch.sum(Y * B2, dim=-1).unsqueeze(0) # [1, chunk2]
            
            # dist2 = 0.5 * ( sum(X*A^2) + X @ B^2.T - 2 * (X*A) @ B.T + A^2 @ Y.T + sum(Y*B^2) - 2 * A @ (Y*B).T )
            term1 = XA2_sum
            term2 = torch.matmul(X, B2.T)
            term3 = -2.0 * torch.matmul(XA, B.T)
            term4 = torch.matmul(A2, Y.T)
            term5 = YB2_sum
            term6 = -2.0 * torch.matmul(A, YB.T)
            
            dist2 = 0.5 * (term1 + term2 + term3 + term4 + term5 + term6)
            dist2 = torch.clamp(dist2, min=0.0)
            
            D_mat[i:end_i, j:end_j] = dist2
            
    # Variance-aware bandwidth (U3)
    # Compute per-sample temporal variance of entropy
    h_bar_1 = torch.mean(h_1, dim=2) # [N1, T-1]
    h_bar_2 = torch.mean(h_2, dim=2) # [N2, T-1]
    
    v_1 = torch.var(h_bar_1, dim=1) # [N1]
    v_2 = torch.var(h_bar_2, dim=1) # [N2]
    
    sigma2_median = torch.median(D_mat).item()
    if sigma2_median < 1e-8:
        sigma2_median = 1.0
        
    sigma2_1 = sigma2_median * (1.0 + v_1) # [N1]
    sigma2_2 = sigma2_median * (1.0 + v_2) # [N2]
    
    sigma_1 = torch.sqrt(sigma2_1).unsqueeze(1) # [N1, 1]
    sigma_2 = torch.sqrt(sigma2_2).unsqueeze(0) # [1, N2]
    
    # Adaptive kernel
    K_unified = torch.exp(-D_mat / (sigma_1 * sigma_2))
    
    return K_unified
