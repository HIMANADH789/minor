import torch

def pack_upper_triangular(X: torch.Tensor):
    """
    X: [..., K, K]
    Returns upper triangular elements [..., K(K+1)/2]
    """
    K = X.size(-1)
    row_idx, col_idx = torch.triu_indices(K, K)
    return X[..., row_idx, col_idx]

def compute_weighted_field(rho_t: torch.Tensor, Omega_t: torch.Tensor):
    """
    rho_t: [B, T-1, M, K, K]
    Omega_t: [B, T-1, M, K, K]
    
    Returns:
        Xi_t: [B, T-1, M, K, K] (full)
        Xi_t_packed: [B, T-1, M, K(K+1)//2]
    """
    Xi_t = rho_t * Omega_t
    Xi_t_packed = pack_upper_triangular(Xi_t)
    return Xi_t, Xi_t_packed

