import torch

def compute_log_resolution(epsilons: torch.Tensor):
    """
    epsilons: [M] (e.g. 7 resolutions)
    
    Returns:
        u_m: [M]
        w_hat_m: [M-1] (Normalized log-grid widths for quadrature, Correction A)
        inv_du_m: [M-1] (Precomputed inverse for derivatives, H13)
    """
    u_m = torch.log(epsilons)
    
    du_m = u_m[1:] - u_m[:-1]
    
    # H13 - Precompute (Delta u)^-1
    inv_du_m = 1.0 / (du_m + 1e-8)
    
    # Correction A - Normalized log-grid widths
    w_hat_m = du_m / torch.sum(du_m)
    
    return u_m, w_hat_m, inv_du_m
