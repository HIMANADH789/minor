import torch

def _compute_trajectory_kernel(Xi_t1: torch.Tensor, Xi_t2: torch.Tensor):
    """
    Xi_t: [N, T-1, K, K]
    """
    N1 = Xi_t1.size(0)
    N2 = Xi_t2.size(0)
    
    e1 = torch.norm(Xi_t1, p='fro', dim=(-1, -2)) # [N1, T-1]
    e2 = torch.norm(Xi_t2, p='fro', dim=(-1, -2)) # [N2, T-1]
    
    e1_hat = e1 / (torch.norm(e1, p=2, dim=-1, keepdim=True) + 1e-8)
    e2_hat = e2 / (torch.norm(e2, p=2, dim=-1, keepdim=True) + 1e-8)
    
    dist2 = torch.sum(e1_hat**2, dim=-1, keepdim=True) + torch.sum(e2_hat**2, dim=-1).unsqueeze(0) - 2 * torch.matmul(e1_hat, e2_hat.T)
    dist2 = torch.clamp(dist2, min=0.0)
    
    # Median heuristic
    sigma2 = torch.median(dist2).item()
    if sigma2 < 1e-8:
        sigma2 = 1.0
        
    K_traj = torch.exp(-dist2 / sigma2)
    return K_traj

compute_trajectory_kernel = _compute_trajectory_kernel
