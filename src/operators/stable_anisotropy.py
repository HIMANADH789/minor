import torch
import torch.nn as nn

class StableAnisotropy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, mu):
        """
        mu: (B, T, N) eigenvalues sorted in ascending order.
        """
        # Since eigh returns ascending, mu[:, :, -1] is max, mu[:, :, 0] is min
        mu_max = mu[:, :, -1]
        mu_min = mu[:, :, 0]
        
        a_t = torch.log(mu_max + 1e-8) - torch.log(mu_min + 1e-8) # (B, T)
        a_tilde = torch.tanh(a_t / 8.0) # (B, T)
        
        return a_tilde
