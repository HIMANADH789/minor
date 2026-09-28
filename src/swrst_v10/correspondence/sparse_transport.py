import torch

def compute_sparse_sharpening(Gamma: torch.Tensor, p: float = 1.5):
    """
    H7: Sparse sharpening in-place
    Gamma: [Nx, Ny, T, M, M]
    """
    Gamma.pow_(p)
    
    # Normalize over the target dimension n
    # Gamma_sum = sum_n Gamma(m,n)
    sum_G = Gamma.sum(dim=-1, keepdim=True) + 1e-12
    Gamma.div_(sum_G)
    
    return Gamma
