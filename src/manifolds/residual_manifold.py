import torch
import torch.nn as nn

class ResidualManifold(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
        
    def forward(self, log_S_bar_seq, log_S_bar_x):
        """
        log_S_bar_seq: Log of the local barycenters, shape (B, T, K, K)
        log_S_bar_x: Log of the global barycenter, shape (B, K, K)
        
        Returns:
        R_hat_t: Normalized topological residual, shape (B, T, K, K)
        """
        # Log-residual geometry
        R_t = log_S_bar_seq - log_S_bar_x.unsqueeze(1)
        
        # Intrinsic normalization (PATCH I)
        norm_seq = torch.norm(log_S_bar_seq, p='fro', dim=(-2, -1), keepdim=True) ** 2
        norm_x = torch.norm(log_S_bar_x, p='fro', dim=(-2, -1), keepdim=True).unsqueeze(1) ** 2
        
        R_hat_t = R_t / (norm_seq + norm_x + self.tau)
        
        return R_hat_t
