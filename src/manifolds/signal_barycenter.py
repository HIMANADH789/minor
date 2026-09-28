from src.utils.safe_eigh import safe_eigh
import torch
import torch.nn as nn

class SignalBarycenter(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, log_S_bar_seq, eps_seq):
        """
        log_S_bar_seq: (B, T, K_dim, K_dim)
        eps_seq: (B, T, 1)
        """
        w = 1.0 / (eps_seq + 1e-8)
        w = w / w.sum(dim=1, keepdim=True)
        w = w.view(log_S_bar_seq.size(0), log_S_bar_seq.size(1), 1, 1) # (B, T, 1, 1)
        
        # Log-Euclidean Barycenter
        L_x = (w * log_S_bar_seq).sum(dim=1) # (B, K_dim, K_dim)
        
        # Matrix Exp of Barycenter with straight-through estimator
        evals_x, evecs_x = safe_eigh(L_x.detach().float())
        exp_evals_x = torch.exp(evals_x).unsqueeze(-1)
        S_bar_x_detach = torch.matmul(evecs_x * exp_evals_x.transpose(-2, -1), evecs_x.transpose(-2, -1)).to(log_S_bar_seq.dtype)
        
        S_bar_x = S_bar_x_detach + (L_x - L_x.detach())
        log_S_bar_x = L_x
        
        return S_bar_x, log_S_bar_x, torch.exp(evals_x).to(log_S_bar_seq.dtype), evecs_x.to(log_S_bar_seq.dtype)
