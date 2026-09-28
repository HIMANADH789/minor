import torch

def compute_density_constrained_sinkhorn(rho_t: torch.Tensor, mu_t: torch.Tensor, mu_t_plus_1: torch.Tensor, C: torch.Tensor, eps_min: float = 0.01, eps_max: float = 0.1, tau: float = 1.0):
    """
    rho_t: [B, T-1, K, K]
    """
    eps_t = eps_min + (eps_max - eps_min) * (1.0 - rho_t) # [B, T-1, K, K]
    
    L = -C.unsqueeze(0).unsqueeze(0) / (eps_t + 1e-8) + tau * torch.log(rho_t + 1e-8)
    
    # Log-domain Sinkhorn
    u = torch.zeros_like(mu_t)
    v = torch.zeros_like(mu_t_plus_1)
    
    log_mu = torch.log(mu_t + 1e-8)
    log_nu = torch.log(mu_t_plus_1 + 1e-8)
    
    for _ in range(20):
        # u update
        arg1 = L + v.unsqueeze(-2)
        u = log_mu - torch.logsumexp(arg1, dim=-1)
        
        # v update
        arg2 = L.transpose(-1, -2) + u.unsqueeze(-2)
        v = log_nu - torch.logsumexp(arg2, dim=-1)
        
    P_t = torch.exp(u.unsqueeze(-1) + L + v.unsqueeze(-2))
    return P_t
