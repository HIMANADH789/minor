import torch
import torch.nn as nn

class EntropicResolutionFlow(nn.Module):
    def __init__(self, eps_0=0.10, eps_min=0.025, alpha=2.0, rho=0.7, xi=0.5):
        super().__init__()
        self.eps_0 = eps_0
        self.eps_min = eps_min
        self.alpha = alpha
        self.rho = rho
        self.xi = xi
        
    def forward(self, delta_t, surprise_t, delta_mem_t_minus_1=None):
        """
        delta_t: Spectral gap at time t, shape (B, 1)
        surprise_t: Surprise metric at time t (e.g., |K_t|), shape (B, 1)
        delta_mem_t_minus_1: Memory state from previous timestep, shape (B, 1)
        
        Returns:
            eps_t: Dynamic temperature for current timestep, shape (B, 1)
            delta_mem_t: Updated memory state, shape (B, 1)
        """
        B = delta_t.size(0)
        device = delta_t.device
        
        if delta_mem_t_minus_1 is None:
            # Initialize memory with current delta_t if no history
            delta_mem_t_minus_1 = delta_t.clone()
            
        # Surprise-weighted memory
        delta_tilde_t = delta_t + self.xi * torch.abs(surprise_t)
        
        # Exponential moving average memory
        delta_mem_t = self.rho * delta_mem_t_minus_1 + (1 - self.rho) * delta_tilde_t
        
        # Dynamic temperature calculation using previous memory state (as per spec: Delta_memory_t-1)
        # Wait, the spec says: eps_t = eps_0 * exp(-alpha * (1 - delta_mem_{t-1})) + eps_min
        eps_t = self.eps_0 * torch.exp(-self.alpha * (1.0 - delta_mem_t_minus_1)) + self.eps_min
        
        return eps_t, delta_mem_t
