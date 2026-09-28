import torch

def compute_spectral_entropy(z: torch.Tensor, tau: float = 1e-8):
    """
    z: [B, T-1, M, r]
    
    Returns:
        H_spec: [B, M]
    """
    B, T, M, r = z.shape
    
    # We want singular values of z as a (T-1) x r matrix for each (B, M)
    # Permute to [B, M, T-1, r]
    z_perm = z.permute(0, 2, 1, 3)
    
    # SVD values
    s = torch.linalg.svdvals(z_perm) # [B, M, min(T-1, r)]
    
    # Normalize s to sum to 1 to compute entropy
    s_sum = torch.sum(s, dim=-1, keepdim=True) + tau
    s_norm = s / s_sum
    
    # Compute entropy H_spec = -sum(s * log(s))
    H_spec = -torch.sum(s_norm * torch.log(s_norm + tau), dim=-1) # [B, M]
    
    return H_spec
