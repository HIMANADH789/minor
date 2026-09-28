import torch

def compute_multi_density(Q_t: torch.Tensor, E_m: torch.Tensor):
    """
    Q_t: [B, T-1, K, K]
    E_m: [M, K, K]
    
    Returns:
        rho_t: [B, T-1, M, K, K]
    """
    # D_t_m: [B, T-1, M, K, K]
    D_t_m = Q_t.unsqueeze(2) * E_m.unsqueeze(0).unsqueeze(0)
    
    # Normalize to get D_tilde for statistics
    # M_t_m is the unnormalized mass
    M_t_m = torch.sum(D_t_m, dim=(-1, -2), keepdim=True) + 1e-8
    D_tilde = D_t_m / M_t_m
    
    # Statistics per order
    # H_t: Entropy
    H_t = -torch.sum(D_tilde * torch.log(D_tilde + 1e-8), dim=(-1, -2))
    # S_t: Sparsity
    S_t = torch.sum(D_tilde**2, dim=(-1, -2))
    # G_t: Peak
    # max over last two dims
    G_t = torch.max(D_tilde.view(*D_tilde.shape[:-2], -1), dim=-1)[0]
    # M_t: Mass (scalar per grid point)
    M_t = M_t_m.squeeze(-1).squeeze(-1)
    
    # Fixed arbitrary sensible scales as per original code logic
    lambda1, lambda2, lambda3, lambda4 = 1.0, 1.0, 1.0, 1.0
    
    alpha_raw = lambda1 * H_t + lambda2 * S_t - lambda3 * G_t + lambda4 * M_t
    alpha_t = 1.0 + 2.0 * torch.sigmoid(alpha_raw)
    
    alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
    
    # Self-density using D_tilde to prevent massive underflow
    num = D_tilde ** alpha_t
    den = torch.sum(num, dim=(-1, -2), keepdim=True) + 1e-8
    rho_t = num / den
    
    return rho_t

