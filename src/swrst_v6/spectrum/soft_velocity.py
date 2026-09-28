import torch

def compute_soft_velocity(V_packed: torch.Tensor, p_t: torch.Tensor):
    """
    V_packed: [B, T-1, M-1, K(K-1)/2]
    p_t: [B, T-1, M-1]
    
    Returns:
        V_bar: [B, T-1, K(K-1)/2]
    """
    p_expanded = p_t.unsqueeze(-1) # [B, T-1, M-1, 1]
    V_bar = torch.sum(p_expanded * V_packed, dim=2) # [B, T-1, K(K-1)/2]
    return V_bar
