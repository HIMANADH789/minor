import torch

def compute_multi_transport(
    mu_t: torch.Tensor, mu_t_plus_1: torch.Tensor, 
    Q_t: torch.Tensor, rho_t_adaptive: torch.Tensor,
    C: torch.Tensor, eps_sorted: torch.Tensor,
    tau: float = 1e-8, num_iters: int = 5
):
    """
    H1: Batch all branches together in a single Sinkhorn.
    H2: Cache exp(-C/eps_m) exactly.
    
    mu_t, mu_t_plus_1: [B, T, K]
    Q_t: [B, T, K, K]
    rho_t_adaptive: [B, T, K, K] (Computed from adaptive epsilon only)
    C: [K, K]
    eps_sorted: [M]
    
    Returns:
        Xi: [B, T, M, K, K]
    """
    B, T, K_bins = mu_t.shape
    M = eps_sorted.shape[0]
    
    a = mu_t.reshape(B*T, K_bins) # [B*T, K]
    b = mu_t_plus_1.reshape(B*T, K_bins) # [B*T, K]
    
    # Check for empty margins
    a_sum = a.sum(dim=1, keepdim=True)
    b_sum = b.sum(dim=1, keepdim=True)
    
    a_safe = a / (a_sum + tau)
    b_safe = b / (b_sum + tau)
    
    # H2: Cache exp(-C/eps_m) for all M
    # eps_sorted is [M]
    # C is [K, K]
    K_matrices = torch.exp(-C.view(1, K_bins, K_bins) / eps_sorted.view(M, 1, 1)) # [M, K, K]
    
    # H1: Batch all branches
    # We expand a and b to [B*T*M, K]
    a_exp = a_safe.unsqueeze(1).expand(B*T, M, K_bins).reshape(B*T*M, K_bins)
    b_exp = b_safe.unsqueeze(1).expand(B*T, M, K_bins).reshape(B*T*M, K_bins)
    
    K_exp = K_matrices.unsqueeze(0).expand(B*T, M, K_bins, K_bins).reshape(B*T*M, K_bins, K_bins)
    
    u = torch.ones_like(a_exp) / K_bins
    v = torch.ones_like(b_exp) / K_bins
    
    for _ in range(num_iters):
        v = b_exp / (torch.bmm(K_exp.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + tau)
        u = a_exp / (torch.bmm(K_exp, v.unsqueeze(2)).squeeze(2) + tau)
        
    P_flat = u.unsqueeze(2) * K_exp * v.unsqueeze(1) # [B*T*M, K, K]
    P = P_flat.view(B, T, M, K_bins, K_bins)
    
    # Phase III: Xi_t^{(m)} = rho_t \odot (P_t^{(m)} - Q_t)
    # rho_t_adaptive is [B, T, K, K]. We expand to M
    rho_exp = rho_t_adaptive.unsqueeze(2) # [B, T, 1, K, K]
    Q_exp = Q_t.unsqueeze(2) # [B, T, 1, K, K]
    
    Xi = rho_exp * (P - Q_exp) # [B, T, M, K, K]
    
    return P, Xi
