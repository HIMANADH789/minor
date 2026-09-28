import torch

def compute_cross_commutator(Xi_t: torch.Tensor):
    """
    Xi_t: [B, T-1, M, K, K]
    
    Returns:
        S_cross: [B, K, K] (sum over time and over scale pairs m < n of commutators)
    """
    B, T_minus_1, M, K, _ = Xi_t.shape
    
    S_cross = torch.zeros((B, K, K), dtype=Xi_t.dtype, device=Xi_t.device)
    
    # We can aggregate over time first to save some memory on intermediates, 
    # but [A_t, B_t] is not [sum A_t, sum B_t].
    # So we must compute commutator per time step, then sum over time.
    
    # Let's flatten time and batch to use bmm
    Xi_flat = Xi_t.view(B * T_minus_1, M, K, K)
    
    S_cross_flat = torch.zeros((B * T_minus_1, K, K), dtype=Xi_t.dtype, device=Xi_t.device)
    
    for m in range(M):
        for n in range(m + 1, M):
            X_m = Xi_flat[:, m]
            X_n = Xi_flat[:, n]
            
            comm = torch.bmm(X_m, X_n) - torch.bmm(X_n, X_m)
            S_cross_flat += comm
            
    # Reshape and sum over time
    S_cross_time = S_cross_flat.view(B, T_minus_1, K, K)
    S_cross = torch.sum(S_cross_time, dim=1)
    
    return S_cross

