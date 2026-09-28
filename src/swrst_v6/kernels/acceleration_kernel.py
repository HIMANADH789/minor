import torch

def compute_kinematic_kernel(V_bar_1: torch.Tensor, V_bar_2: torch.Tensor, chunk_size: int = 512):
    """
    Computes Cosine-RBF kernel for velocity or acceleration.
    V_bar: [B, T-1, K(K-1)/2]
    
    Returns:
        K_V: [N1, N2]
    """
    N1 = V_bar_1.size(0)
    N2 = V_bar_2.size(0)
    
    # Flatten across time and features (H14)
    A_full = V_bar_1.reshape(N1, -1) # [N1, D]
    B_full = V_bar_2.reshape(N2, -1) # [N2, D]
    
    # Pre-normalize to simplify dot product to cosine similarity
    A_norm = A_full / (torch.norm(A_full, p=2, dim=-1, keepdim=True) + 1e-8)
    B_norm = B_full / (torch.norm(B_full, p=2, dim=-1, keepdim=True) + 1e-8)
    
    K_mat = torch.zeros((N1, N2), device=V_bar_1.device, dtype=torch.float32)
    
    # H4: Chunk Pairwise Kernels
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        A = A_norm[i:end_i]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            B = B_norm[j:end_j]
            
            # Cosine similarity is just dot product of normalized vectors
            cosine_sim = torch.matmul(A, B.T)
            
            K_mat[i:end_i, j:end_j] = cosine_sim
            
    # Estimate sigma from variance of cosine similarity
    sigma2_median = torch.var(K_mat).item()
    if sigma2_median < 1e-8:
        sigma2_median = 1.0
        
    K_V = torch.exp(K_mat / sigma2_median)
    
    return K_V
