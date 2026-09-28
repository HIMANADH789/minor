import torch
import torch.nn as nn

class ResidualGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, P, rho, Lam, U):
        B, T, N, _ = P.shape
        
        rho_expanded = rho.unsqueeze(1)
        R_bank = P - rho_expanded
        
        # We no longer compute eigh(Sigma). We just use Lam, U from shared eigbank.
        # But we still return R_bank, and other features that path_rep expects.
        
        # For legacy compatibility with path_rep, we compute some basic norms
        F_t = torch.sqrt(torch.sum(R_bank**2, dim=(-2, -1)) + 1e-8)
        
        Q_t = torch.mean(R_bank, dim=-1)
        B_t = torch.mean(R_bank, dim=-2)
        
        Q_t_norm = torch.sqrt(torch.sum(Q_t**2, dim=-1) + 1e-8)
        B_t_norm = torch.sqrt(torch.sum(B_t**2, dim=-1) + 1e-8)
        
        R_pad = torch.cat([R_bank[:, :1], R_bank], dim=1)
        delta_t = torch.sqrt(torch.sum((R_pad[:, 1:] - R_pad[:, :-1])**2, dim=(-2, -1)) + 1e-8)
        
        C_t = torch.sqrt(torch.sum(R_bank**2, dim=(-2, -1)) + 1e-8)
        C_pad = torch.cat([C_t[:, :1], C_t], dim=1)
        Delta_C_t = torch.abs(C_pad[:, 1:] - C_pad[:, :-1])
        
        return R_bank, Lam, U, F_t, Q_t_norm, B_t_norm, delta_t, C_t, Delta_C_t
