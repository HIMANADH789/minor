import torch

def compute_soft_acceleration(A_padded: torch.Tensor, p_t: torch.Tensor):
    """
    A_padded: [B, T-1, M-1, K(K-1)/2]
    p_t: [B, T-1, M-1]
    
    Returns:
        A_bar: [B, T-1, K(K-1)/2]
    """
    p_expanded = p_t.unsqueeze(-1) # [B, T-1, M-1, 1]
    A_bar = torch.sum(p_expanded * A_padded, dim=2) # [B, T-1, K(K-1)/2]
    return A_bar
