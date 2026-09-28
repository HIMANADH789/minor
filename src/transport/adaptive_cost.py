import torch
import torch.nn as nn

class AdaptiveCostGeometry(nn.Module):
    def __init__(self, eps_0=0.10):
        super().__init__()
        self.eps_0 = eps_0
        
    def forward(self, x, eps_t):
        """
        x: Feature window of shape (B, K, D)
        eps_t: Dynamic temperature for the window, shape (B, 1)
        
        Returns:
        C: Cost matrix of shape (B, K, K)
        """
        # Compute L1 distance first: |w_i - w_j|
        # x_i: (B, K, 1, D)
        # x_j: (B, 1, K, D)
        diff = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)) # (B, K, K, D)
        
        # Sum over feature dimension to get total L1 distance
        dist_l1 = diff.sum(dim=-1) # (B, K, K)
        
        # Scale dist_l1 by sqrt(D) to prevent fp16 overflow when powered
        D_feat = diff.size(-1)
        dist_l1 = dist_l1 / (torch.sqrt(torch.tensor(D_feat, dtype=torch.float32, device=dist_l1.device)) + 1e-8)
        
        # Power scaling: 2 / (1 + eps_t / eps_0)
        if eps_t.dim() == 4:
            # eps_t is (B, 3, 1, 1), make dist_l1 broadcastable -> (B, 1, K, K)
            dist_l1 = dist_l1.unsqueeze(1)
            eps_t_view = eps_t
        else:
            # eps_t is (B, 1) or similar
            eps_t_view = eps_t.view(-1, 1, 1)
            
        power = 2.0 / (1.0 + eps_t_view / self.eps_0)
        
        # Apply power
        C = torch.pow(dist_l1 + 1e-8, power)
        
        return C
