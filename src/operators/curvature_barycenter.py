import torch
import torch.nn as nn
import torch.nn.functional as F

class CurvatureWeightedBarycenter(nn.Module):
    def __init__(self, beta=4.0):
        super().__init__()
        self.beta = beta

    def forward(self, P, Lam, U):
        B, T, N, _ = P.shape
        
        # We construct log_S directly from the shared eigbank Lam, U
        # log_S = U * log(Lam) * U^T
        log_Lam = torch.log(Lam)
        log_S = torch.matmul(U * log_Lam.unsqueeze(-2), U.transpose(-2, -1))
        
        log_S_pad = torch.cat([log_S[:, :1], log_S[:, :1], log_S], dim=1)
        
        log_S_t = log_S_pad[:, 2:]
        log_S_tm1 = log_S_pad[:, 1:-1]
        log_S_tm2 = log_S_pad[:, :-2]
        
        d_t = torch.sqrt(torch.sum((log_S_t - log_S_tm1)**2, dim=(-2, -1)) + 1e-8)
        d_tm1 = torch.sqrt(torch.sum((log_S_tm1 - log_S_tm2)**2, dim=(-2, -1)) + 1e-8)
        
        kappa = d_t - d_tm1
        
        # Patch 10: Second-order Operator Curvature K_t = ||P_t - 2P_{t-1} + P_{t-2}||_F
        P_pad = torch.cat([P[:, :1], P[:, :1], P], dim=1)
        P_t = P_pad[:, 2:]
        P_tm1 = P_pad[:, 1:-1]
        P_tm2 = P_pad[:, :-2]
        K_t_scalar = torch.sqrt(torch.sum((P_t - 2 * P_tm1 + P_tm2)**2, dim=(-2, -1)) + 1e-8)
        
        omega = F.softmax(self.beta * kappa, dim=-1)
        rho = torch.sum(omega.view(B, T, 1, 1) * P, dim=1)
        
        return rho, kappa, K_t_scalar
