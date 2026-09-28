import torch

def compute_log_geometric_field(Xi_t: torch.Tensor, tau: float = 1e-5):
    """
    Xi_t: [B, T-1, M, K, K]
    
    Returns:
        log_G_t: [B, T-1, M, K, K]
        log_G_packed: [B, T-1, M, K(K+1)//2]
    """
    B, T_minus_1, M, K, _ = Xi_t.shape
    
    # G_t = Xi * Xi^T
    G_t = torch.matmul(Xi_t, Xi_t.transpose(-1, -2))
    G_t = G_t + tau * torch.eye(K, device=Xi_t.device, dtype=Xi_t.dtype).view(1, 1, 1, K, K)
    
    # Flatten M dimension for eigh
    G_flat = G_t.view(B * T_minus_1 * M, K, K)
    
    # Eigendecomposition
    L, Q = torch.linalg.eigh(G_flat)
    
    log_L = torch.log(torch.clamp(L, min=1e-8))
    log_G_flat = torch.matmul(Q, log_L.unsqueeze(-1) * Q.transpose(-1, -2))
    
    log_G_t = log_G_flat.view(B, T_minus_1, M, K, K)
    
    # Pack upper triangle
    row_idx, col_idx = torch.triu_indices(K, K)
    log_G_packed = log_G_t[..., row_idx, col_idx]
    
    return log_G_t, log_G_packed

