import torch

def compute_fisher_dominant_fusion(K_F: torch.Tensor, K_H: torch.Tensor, K_P: torch.Tensor, K_E: torch.Tensor, K_V: torch.Tensor, K_C: torch.Tensor, K_R: torch.Tensor, K_dF: torch.Tensor):
    """
    Implements Fisher-dominant log-space fusion with fixed weights.
    All kernels must be of the same shape [N1, N2].
    
    Weights (Sum = 1.0):
        K_F: 0.35 (Fisher)
        K_H: 0.15 (Hellinger)
        K_P: 0.10 (Spectral Persistence)
        K_E: 0.10 (Energy Trajectory)
        K_V: 0.10 (Order Variance)
        K_C: 0.10 (Cross-Order)
        K_R: 0.05 (Route-Switch)
        K_dF: 0.05 (Fisher Residual)
    """
    
    # Safe log-space fusion
    # Shape of all K is [N1, N2]. We add 1e-8 before log to prevent -inf
    log_K = (0.35 * torch.log(K_F + 1e-8) +
             0.15 * torch.log(K_H + 1e-8) +
             0.10 * torch.log(K_P + 1e-8) +
             0.10 * torch.log(K_E + 1e-8) +
             0.10 * torch.log(K_V + 1e-8) +
             0.10 * torch.log(K_C + 1e-8) +
             0.05 * torch.log(K_R + 1e-8) +
             0.05 * torch.log(K_dF + 1e-8))
             
    # Exponentiate and clamp in one operation
    K_fused = torch.clamp(torch.exp(log_K), min=1e-8, max=1.0)
    
    return K_fused
