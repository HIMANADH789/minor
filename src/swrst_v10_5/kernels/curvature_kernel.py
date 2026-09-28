import torch
from src.swrst_v10_5.kernels.derivative_hellinger import compute_signed_hellinger_kernel

def compute_curvature_kernel(
    A_x: torch.Tensor, A_y: torch.Tensor,
    norm_x: torch.Tensor, norm_y: torch.Tensor,
    sigma: float = 1.0
):
    """
    U2: Curvature kernel. Same signed Hellinger on A_eps.
    """
    return compute_signed_hellinger_kernel(A_x, A_y, norm_x, norm_y, sigma)
