import torch
import torch.nn as nn

class MultiOrderModularTransport(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, P, mu, U, H_t_scalar):
        B, T, N = mu.shape
        
        # Patch 4: Adaptive PTCO order
        # p_t = 1 + floor(6 H_t), bounded [1, 8]
        p_t = 1.0 + torch.floor(6.0 * H_t_scalar)
        p_t = torch.clamp(p_t, min=1.0, max=8.0)
        p_t = p_t.view(B, T, 1) # (B, T, 1)
        
        # Patch 3: Replace hard inverse clamp with smooth soft inverse
        tau = 0.02
        mu_safe = torch.clamp(mu, min=1e-8)
        mu_p = torch.pow(mu_safe, p_t)
        
        # Soft inverse: \tilde{\mu}^{-p} = exp(-p * log(\mu + \tau))
        mu_inv_p = torch.exp(-p_t * torch.log(mu_safe + tau))
        
        Sigma_p = torch.matmul(U * mu_p.unsqueeze(-2), U.transpose(-2, -1))
        Sigma_inv_p = torch.matmul(U * mu_inv_p.unsqueeze(-2), U.transpose(-2, -1))
        
        M_t = torch.matmul(Sigma_p, torch.matmul(P, Sigma_inv_p)) - torch.matmul(Sigma_inv_p, torch.matmul(P, Sigma_p))
        
        # Return as (B, T, 1, N, N) for compatibility with persistence_energy
        M_t = M_t.unsqueeze(2)
        return M_t
