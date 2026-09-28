import torch

def get_sorted_extra_grid(C_median: float, device: torch.device):
    """
    C1: Adaptive epsilon sorted into the grid.
    Build {eps_adaptive, 0.001, 0.005, 0.02, 0.1, 0.5}, sort ascending.
    """
    eps_adaptive = 0.10 * C_median
    eps_fixed = torch.tensor([0.001, 0.005, 0.02, 0.1, 0.5], dtype=torch.float32, device=device)
    
    eps_all = torch.cat([torch.tensor([eps_adaptive], dtype=torch.float32, device=device), eps_fixed])
    
    # H1: Sort in-place
    eps_sorted, sort_idx = torch.sort(eps_all, dim=-1)
    
    # Find where the adaptive branch ended up
    adaptive_idx = (sort_idx == 0).nonzero(as_tuple=True)[0].item()
    
    return eps_sorted, sort_idx, adaptive_idx
