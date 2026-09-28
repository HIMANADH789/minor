import torch

def compute_structural_deviation(P_t: torch.Tensor, Q_t: torch.Tensor):
    """
    P_t: [B, T-1, M, K, K]
    Q_t: [B, T-1, K, K]
    """
    return P_t - Q_t.unsqueeze(2)

