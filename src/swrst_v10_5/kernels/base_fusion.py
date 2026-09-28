import torch

def compute_base_fusion(
    k_H: torch.Tensor, k_D: torch.Tensor, k_A: torch.Tensor, 
    k_P: torch.Tensor, k_S: torch.Tensor
):
    """
    H4: In-place log fusion. Geometric mean.
    k_H, k_D, k_A, k_P, k_S: [Nx, Ny, M, M]
    Returns:
        k: [Nx, Ny, M, M]
    """
    # logk = 1/5 (logk_H + logk_D + logk_A + logk_P + logk_S)
    # We must ensure no exactly zero values to avoid log(0)
    tau = 1e-12
    
    logk = torch.log(k_H + tau)
    logk.add_(torch.log(k_D + tau))
    logk.add_(torch.log(k_A + tau))
    logk.add_(torch.log(k_P + tau))
    logk.add_(torch.log(k_S + tau))
    
    logk.div_(5.0)
    
    k = torch.exp(logk)
    return k
