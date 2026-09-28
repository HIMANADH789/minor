import torch

def compute_path_signature(Xi_t: torch.Tensor):
    """
    Xi_t: [B, T-1, K, K]
    Returns S1, S2
    """
    B, T_minus_1, K, _ = Xi_t.shape
    
    S1 = torch.sum(Xi_t, dim=1)
    
    S2 = torch.zeros((B, K, K), device=Xi_t.device, dtype=Xi_t.dtype)
    
    # H8: Optimization use causal band |s - t| <= 4, upper triangle only
    # S2 = sum_{s < t} [Xi_s, Xi_t]
    for s in range(T_minus_1):
        # Limit t to s + 1 to s + 4 inclusive
        for t in range(s + 1, min(T_minus_1, s + 5)):
            Xi_s = Xi_t[:, s]
            Xi_t_curr = Xi_t[:, t]
            # Commutator
            comm = torch.matmul(Xi_s, Xi_t_curr) - torch.matmul(Xi_t_curr, Xi_s)
            S2 += comm
            
    return S1, S2
