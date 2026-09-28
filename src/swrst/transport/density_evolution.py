import torch

def _compute_density_evolution(mu_t: torch.Tensor, mu_t_plus_1: torch.Tensor, C: torch.Tensor):
    """
    mu_t: [B, T-1, K]
    mu_t_plus_1: [B, T-1, K]
    """
    # Independence coupling
    Q_t = mu_t.unsqueeze(-1) * mu_t_plus_1.unsqueeze(-2) # [B, T-1, K, K]
    
    # Structural potential
    D_t = Q_t * torch.exp(-C)
    D_t_sum = torch.sum(D_t, dim=(-1, -2), keepdim=True) + 1e-8
    D_tilde_t = D_t / D_t_sum
    
    # Statistics
    H_t = -torch.sum(D_tilde_t * torch.log(D_tilde_t + 1e-8), dim=(-1, -2))
    S_t = torch.sum(D_tilde_t**2, dim=(-1, -2))
    G_t = torch.max(D_tilde_t.view(*D_tilde_t.shape[:-2], -1), dim=-1)[0]
    
    # Adaptive sharpness
    # lambda1, lambda2, lambda3 fixed to arbitrary sensible scales as in prior iterations
    lambda1, lambda2, lambda3 = 1.0, 1.0, 1.0
    alpha_raw = lambda1 * H_t + lambda2 * S_t - lambda3 * G_t
    alpha_t = 1.0 + torch.sigmoid(alpha_raw)
    alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
    
    # Self-density
    num = D_tilde_t ** alpha_t
    den = torch.sum(num, dim=(-1, -2), keepdim=True) + 1e-8
    rho_t = num / den
    
    return Q_t, D_t, H_t, S_t, G_t, alpha_t.squeeze(), rho_t

compute_density_evolution = _compute_density_evolution
