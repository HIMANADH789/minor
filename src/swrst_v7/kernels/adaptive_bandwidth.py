import torch

def compute_adaptive_kernel(
    d2_mat: torch.Tensor,
    KL_1: torch.Tensor,
    KL_2: torch.Tensor,
    beta: float = 0.50,
    sigma0_2: float = None
):
    """
    d2_mat: [N1, N2]
    KL_1: [N1]
    KL_2: [N2]
    
    Returns:
        K_mat: [N1, N2]
        sigma0_2: The computed or used median sigma0^2
    """
    if sigma0_2 is None:
        sigma0_2 = torch.median(d2_mat).item()
        if sigma0_2 < 1e-8:
            sigma0_2 = 1.0
            
    # sigma_x^2 = sigma_0^2 * exp(beta * KL_x)
    sigma2_1 = sigma0_2 * torch.exp(beta * KL_1) # [N1]
    sigma2_2 = sigma0_2 * torch.exp(beta * KL_2) # [N2]
    
    # sigma_xy = sqrt(sigma_x^2 * sigma_y^2)
    # sigma_xy = sigma0_2 * exp(0.5 * beta * (KL_x + KL_y))
    sigma_xy = torch.sqrt(sigma2_1.unsqueeze(1) * sigma2_2.unsqueeze(0)) # [N1, N2]
    
    K_mat = torch.exp(-d2_mat / sigma_xy)
    
    return K_mat, sigma0_2
