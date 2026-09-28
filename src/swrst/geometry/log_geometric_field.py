import torch

def compute_log_geometric_field(Xi_t: torch.Tensor, tau: float = 1e-5):
    """
    Xi_t: [B, T-1, K, K]
    """
    B, T_minus_1, K, _ = Xi_t.shape
    
    G_t = torch.matmul(Xi_t, Xi_t.transpose(-1, -2))
    G_t = G_t + tau * torch.eye(K, device=Xi_t.device, dtype=Xi_t.dtype).view(1, 1, K, K)
    
    # Eigendecomposition
    # H9: batched eigh
    L, Q = torch.linalg.eigh(G_t)
    
    log_L = torch.log(torch.clamp(L, min=1e-8))
    log_G_t = torch.matmul(Q, log_L.unsqueeze(-1) * Q.transpose(-1, -2))
    
    return log_G_t
