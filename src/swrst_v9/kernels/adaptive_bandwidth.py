import torch

def compute_adaptive_bandwidth(p_train: torch.Tensor, p_test: torch.Tensor = None):
    """
    p_train: [N_train, M]
    p_test: [N_test, M] or None
    
    Returns:
        sigma2_xy: [N_test, N_train] or [N_train, N_train]
    """
    if p_test is None:
        p_test = p_train
        
    N1 = p_test.size(0)
    N2 = p_train.size(0)
    
    dist_p = torch.cdist(p_test, p_train, p=1.0) # [N1, N2]
    
    # sigma_0^2 is median of train-train distance
    if p_test is p_train:
        sigma0_2 = torch.median(dist_p)
    else:
        dist_p_train = torch.cdist(p_train, p_train, p=1.0)
        sigma0_2 = torch.median(dist_p_train)
        
    if sigma0_2 < 1e-8:
        sigma0_2 = 1.0
        
    sigma2_xy = sigma0_2 * torch.exp(0.5 * dist_p) # [N1, N2]
    
    return sigma2_xy
