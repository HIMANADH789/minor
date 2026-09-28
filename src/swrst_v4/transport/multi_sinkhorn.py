import torch

def compute_multi_sinkhorn(mu_t: torch.Tensor, mu_t_plus_1: torch.Tensor, Q_t: torch.Tensor, rho_t: torch.Tensor, C: torch.Tensor, epsilons: list, u_buf=None, v_buf=None, tau: float = 1.0):
    """
    Computes batched parallel Sinkhorn for all M resolutions in strict FP32.
    mu_t: [B, T-1, K]
    mu_t_plus_1: [B, T-1, K]
    Q_t: [B, T-1, K, K]
    C: [K, K]
    epsilons: list of M floats
    
    Returns:
        P_t: [B, T-1, M, K, K]
        h_t: [B, T-1, M] (entropy profile)
        F_num: [B, T-1, M, K, K] (Fisher numerator P_t - Q_t)
    """
    B, T_minus_1, K = mu_t.shape
    M = len(epsilons)
    
    # We flatten B, T, M into a single batch dimension for Sinkhorn
    # mu_t_expanded: [B, T-1, M, K]
    mu_t_exp = mu_t.unsqueeze(2).expand(B, T_minus_1, M, K).reshape(B * T_minus_1 * M, K)
    mu_t_plus_1_exp = mu_t_plus_1.unsqueeze(2).expand(B, T_minus_1, M, K).reshape(B * T_minus_1 * M, K)
    
    # Correct L computation in log domain to prevent exp underflow
    eps_tensor = torch.tensor(epsilons, dtype=torch.float32, device=C.device).view(M, 1, 1)
    
    # L_base = -C / eps_m
    L_base = -C.unsqueeze(0) / eps_tensor # [M, K, K]
    L_exp = L_base.unsqueeze(0).unsqueeze(0).expand(B, T_minus_1, M, K, K)
    
    # Add density constraint
    L_constrained = L_exp + tau * torch.log(rho_t + 1e-8)
    L_exp = L_constrained.reshape(B * T_minus_1 * M, K, K)
    
    if u_buf is None:
        u = torch.zeros_like(mu_t_exp)
    else:
        u = u_buf.reshape(B * T_minus_1 * M, K)
        u.zero_()
        
    if v_buf is None:
        v = torch.zeros_like(mu_t_plus_1_exp)
    else:
        v = v_buf.reshape(B * T_minus_1 * M, K)
        v.zero_()
        
    log_mu = torch.log(mu_t_exp + 1e-12)
    log_nu = torch.log(mu_t_plus_1_exp + 1e-12)
    
    # Transposed L for v update
    L_exp_t = L_exp.transpose(-1, -2)
    
    for _ in range(20):
        # u update
        arg1 = L_exp + v.unsqueeze(-2)
        u = log_mu - torch.logsumexp(arg1, dim=-1)
        
        # v update
        arg2 = L_exp_t + u.unsqueeze(-2)
        v = log_nu - torch.logsumexp(arg2, dim=-1)
        
    log_P = u.unsqueeze(-1) + L_exp + v.unsqueeze(-2)
    P_flat = torch.exp(log_P)
    
    # H5 - Fused entropy + Fisher numerator
    # Entropy h_t = -sum(P * log P)
    h_flat = -torch.sum(P_flat * log_P, dim=(-1, -2))
    
    P_t = P_flat.view(B, T_minus_1, M, K, K)
    h_t = h_flat.view(B, T_minus_1, M)
    
    # Fisher numerator: P_t - Q_t. Q_t is [B, T-1, K, K]
    F_num = P_t - Q_t.unsqueeze(2)
    
    return P_t, h_t, F_num

