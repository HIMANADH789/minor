import torch

def compute_structural_energy(P_t: torch.Tensor, Q_t: torch.Tensor, rho_t: torch.Tensor):
    """
    P_t: [B, T-1, K, K]
    Q_t: [B, T-1, K, K]
    rho_t: [B, T-1, K, K]
    """
    Omega_t = P_t - Q_t
    Xi_t = rho_t * Omega_t
    return Omega_t, Xi_t
