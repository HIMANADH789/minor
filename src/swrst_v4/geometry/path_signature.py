import torch

def compute_path_signature(Xi_bar_t: torch.Tensor):
    """
    Xi_bar_t: [B, T-1, K, K]
    Returns S1, S2 using batched bmm and causal band |s-t| <= 4.
    """
    B, T_minus_1, K, _ = Xi_bar_t.shape
    
    S1 = torch.sum(Xi_bar_t, dim=1)
    
    S2 = torch.zeros((B, K, K), device=Xi_bar_t.device, dtype=Xi_bar_t.dtype)
    
    # H6 - Batched bmm for path signatures over |s-t| <= 4
    for k in range(1, 5):
        if T_minus_1 <= k:
            break
            
        Xi_s = Xi_bar_t[:, :-k].reshape(-1, K, K)     # [B * (T-1-k), K, K]
        Xi_t_curr = Xi_bar_t[:, k:].reshape(-1, K, K) # [B * (T-1-k), K, K]
        
        # [Xi_s, Xi_t] = Xi_s @ Xi_t - Xi_t @ Xi_s
        term1 = torch.bmm(Xi_s, Xi_t_curr)
        term2 = torch.bmm(Xi_t_curr, Xi_s)
        comm = term1 - term2
        
        # Reshape back to [B, T-1-k, K, K] and sum over time
        comm_b = comm.view(B, T_minus_1 - k, K, K)
        S2 += torch.sum(comm_b, dim=1)
        
    return S1, S2

