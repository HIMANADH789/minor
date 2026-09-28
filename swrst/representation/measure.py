import torch
from ..config import SWRSTConfig

class IncrementalMeasureBuilder:
    def __init__(self, config: SWRSTConfig):
        self.config = config
        
    def build_master_coordinates(self, shared_stft: torch.Tensor):
        """
        Builds the global geometry map for the entire batch up to T_max.
        shared_stft: (B, F, T_max)
        Returns: 
          geometry: (B, F, T_max, D)
          mass: (B, F, T_max)
        """
        B, F, T_max = shared_stft.shape
        device = shared_stft.device
        
        f_idx = torch.arange(F, device=device).view(1, F, 1).expand(B, F, T_max)
        t_idx = torch.arange(T_max, device=device).view(1, 1, T_max).expand(B, F, T_max)
        
        mass = shared_stft + 1e-8
        
        channels = []
        if self.config.use_frequency:
            channels.append(f_idx.float() / (F - 1 + 1e-8))
            
        if self.config.use_time:
            channels.append(t_idx.float() / (T_max - 1 + 1e-8))
            
        if self.config.use_energy:
            # Normalize energy per sequence up to T_max
            max_e = shared_stft.amax(dim=(1,2), keepdim=True) + 1e-8
            e_norm = shared_stft / max_e
            channels.append(e_norm)
            
        if self.config.use_gradient:
            grad = torch.zeros_like(shared_stft)
            grad[:, :, 1:] = shared_stft[:, :, 1:] - shared_stft[:, :, :-1]
            max_g = grad.abs().amax(dim=(1,2), keepdim=True) + 1e-8
            grad_norm = grad / max_g
            channels.append(grad_norm)
            
        geometry = torch.stack(channels, dim=-1) # (B, F, T_max, D)
        
        return geometry, mass
