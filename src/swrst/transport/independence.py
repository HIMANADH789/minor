import torch
from src.swrst.transport.deterministic_ot import compute_cost_matrix

def compute_independence_cost(mu_t, K_bins):
    """
    Computes a cost matrix.
    We return the broadcasted spatial cost matrix.
    """
    C_np = compute_cost_matrix(K_bins)
    C = torch.tensor(C_np, dtype=torch.float32, device=mu_t.device)
    B, T, _ = mu_t.shape
    return C.unsqueeze(0).unsqueeze(0).expand(B, T, K_bins, K_bins)
