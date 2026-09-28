import torch
import torch.nn.functional as F

def compute_curvature_field(Xi_t: torch.Tensor):
    """
    Xi_t: [B, T-1, K, K]
    Returns K_t: [B, T-1]
    """
    B, T_minus_1, K, _ = Xi_t.shape
    
    Xi_flat = Xi_t.view(B, T_minus_1, K*K).transpose(1, 2)
    Xi_pad = F.pad(Xi_flat, (1, 1), mode='replicate')
    
    kernel = torch.tensor([1.0, -2.0, 1.0], dtype=Xi_flat.dtype, device=Xi_flat.device)
    kernel = kernel.view(1, 1, 3).expand(K*K, 1, 3)
    
    delta_Xi = F.conv1d(Xi_pad, kernel, groups=K*K)
    delta_Xi = delta_Xi.transpose(1, 2).view(B, T_minus_1, K, K)
    
    K_t = torch.norm(delta_Xi, p='fro', dim=(-1, -2))
    return K_t
