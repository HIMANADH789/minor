import torch
import torch.nn as nn

class SpectralGapResolutionState(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, evals, delta_lag1=None, delta_lag2=None):
        """
        evals: Eigenvalues of S_t, shape (B, K). Assumed sorted ascending (default for eigh).
        delta_lag1: Delta_t from previous timestep.
        delta_lag2: Delta_t from two timesteps ago.
        
        Returns:
        delta_t: Current spectral gap (B, 1)
        d_delta_t: First derivative (B, 1)
        d2_delta_t: Second derivative (B, 1)
        """
        # evals are ascending: last is largest (lambda_1 in user notation), second to last is lambda_2
        lambda_1 = evals[..., -1].unsqueeze(-1)
        lambda_2 = evals[..., -2].unsqueeze(-1)
        
        # Spectral gap for SPD matrix
        delta_t = 1.0 - (lambda_2 / (lambda_1 + 1e-8))
        
        if delta_lag1 is None:
            delta_lag1 = delta_t.clone()
        if delta_lag2 is None:
            delta_lag2 = delta_lag1.clone()
            
        d_delta_t = delta_t - delta_lag1
        d2_delta_t = d_delta_t - (delta_lag1 - delta_lag2)
        
        return delta_t, d_delta_t, d2_delta_t
