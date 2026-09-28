import torch

def unpack_upper_triangular(packed_tensor: torch.Tensor, K: int):
    """
    packed_tensor: [..., K(K+1)/2]
    """
    shape = packed_tensor.shape[:-1]
    matrix = torch.zeros(*shape, K, K, device=packed_tensor.device, dtype=packed_tensor.dtype)
    
    idx = 0
    for i in range(K):
        for j in range(i, K):
            matrix[..., i, j] = packed_tensor[..., idx]
            if i != j:
                matrix[..., j, i] = packed_tensor[..., idx]
            idx += 1
            
    return matrix

def compute_spectral_residual(Xi_can_packed: torch.Tensor, K: int):
    """
    Xi_can_packed: [B, T-1, M, K(K+1)/2]
    
    Returns:
        R_t: [B, T-1, M] (Upgrade U1)
    """
    Xi_can = unpack_upper_triangular(Xi_can_packed, K) # [B, T-1, M, K, K]
    
    cov = torch.matmul(Xi_can, Xi_can.transpose(-1, -2)) # [B, T-1, M, K, K]
    
    eigvals = torch.linalg.eigvalsh(cov) # [B, T-1, M, K]
    
    # Extract top-4
    S_t = eigvals[..., -4:] # [B, T-1, M, 4]
    
    # R_t = Var(S_t) across the 4 spectral bands to measure anisotropy instability
    R_t = torch.var(S_t, dim=-1) # [B, T-1, M]
    
    return R_t
