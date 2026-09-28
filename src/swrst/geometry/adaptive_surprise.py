import torch

def compute_adaptive_surprise(Xi_t: torch.Tensor, r: int = 3):
    """
    Xi_t: [B, T-1, K, K]
    Returns N_t: [B, T-1, K, K]
    """
    B, T_minus_1, K, _ = Xi_t.shape
    
    # H10: SVD on K^2 x T only
    Xi_mat = Xi_t.view(B, T_minus_1, K*K).transpose(1, 2) # [B, K^2, T-1]
    
    # FP32 required for stable SVD
    U, S, Vh = torch.linalg.svd(Xi_mat, full_matrices=False)
    
    Ur = U[:, :, :r] # [B, K^2, r]
    
    # Projection: U_r U_r^T Xi_mat
    proj = torch.matmul(Ur, torch.matmul(Ur.transpose(1, 2), Xi_mat)) # [B, K^2, T-1]
    
    Xi_hat = proj.transpose(1, 2).view(B, T_minus_1, K, K)
    N_t = Xi_t - Xi_hat
    
    return N_t
