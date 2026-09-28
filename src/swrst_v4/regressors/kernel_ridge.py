import torch

def solve_kernel_ridge(K_train: torch.Tensor, Y_train: torch.Tensor, h_train: torch.Tensor, lambda0: float = 0.1, beta: float = 0.5):
    """
    K_train: [N, N]
    Y_train: [N, C] one-hot
    h_train: [N, T-1, M] (entropy profile)
    """
    N, C_classes = Y_train.shape
    
    # Class-conditional lambda
    class_counts = torch.sum(Y_train, dim=0) # [C]
    n_max = torch.max(class_counts)
    
    # Avoid division by zero for empty classes (if any)
    class_counts_safe = torch.clamp(class_counts, min=1.0)
    lambda_c = lambda0 * ((class_counts_safe / n_max) ** beta) # [C]
    
    # Map lambda_c to each sample
    lambda_c_x = torch.matmul(Y_train, lambda_c) # [N]
    
    # Sample variance scaling
    h_var_m = torch.var(h_train, dim=2) # [N, T-1]
    h_var_x = torch.mean(h_var_m, dim=1) # [N]
    
    lam_x = lambda_c_x * (1.0 + h_var_x) # [N]
    
    print(f"DEBUG KRR: Mean h_var_x = {torch.mean(h_var_x).item():.4f}, Mean lam_x = {torch.mean(lam_x).item():.4f}")
    
    Lambda_mat = torch.diag(lam_x)
    
    # Solve alpha = (K + Lambda)^-1 Y
    alpha = torch.linalg.solve(K_train + Lambda_mat, Y_train)
    
    return alpha

