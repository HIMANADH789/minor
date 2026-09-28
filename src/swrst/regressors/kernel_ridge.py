import torch

def solve_kernel_ridge(K_train: torch.Tensor, Y_train: torch.Tensor, rho_train: torch.Tensor, lambda0: float = 0.1):
    """
    K_train: [N, N]
    Y_train: [N, C] one-hot
    rho_train: [N, T-1, K, K]
    """
    N = K_train.size(0)
    
    # Density-Adaptive KRR
    # lambda(x) = lambda0 * (1 + Var(rho_x))^-1
    rho_var = torch.var(rho_train.view(N, -1), dim=-1)
    lam_x = lambda0 / (1.0 + rho_var)
    
    Lambda_mat = torch.diag(lam_x)
    
    # Solve alpha = (K + Lambda)^-1 Y
    # Use torch.linalg.solve which is more numerically stable for near-singular matrices than cholesky
    alpha = torch.linalg.solve(K_train + Lambda_mat, Y_train)
    
    return alpha
