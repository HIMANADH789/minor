import torch

def get_sorted_branch_weights(sort_idx: torch.Tensor):
    """
    w_0 = 0.35 (adaptive branch)
    w_m = 0.65 / 5 (fixed branches)
    
    Weights mapped according to the sorting of epsilons.
    """
    device = sort_idx.device
    w_unsorted = torch.tensor([0.35, 0.13, 0.13, 0.13, 0.13, 0.13], dtype=torch.float32, device=device)
    
    # H1: Weights mapped to sorted epsilons
    w_sorted = w_unsorted.gather(0, sort_idx)
    return w_sorted
