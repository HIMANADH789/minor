import torch
import torch.nn as nn

class ResolutionCurvature(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, delta_t, delta_t_minus_1, delta_t_minus_2):
        """
        Computes resolution acceleration.
        More powerful than standard acceleration, captures topological shock.
        """
        K_t = delta_t - 2.0 * delta_t_minus_1 + delta_t_minus_2
        return K_t
