import torch

def unpack_upper_triangular(packed_tensor: torch.Tensor, K: int):
    """
    Unpacks an upper triangular packed symmetric matrix.
    packed_tensor: [..., K(K+1)/2]
    
    Returns:
        matrix: [..., K, K]
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

def compute_resolution_anisotropy(Xi_tilde_packed: torch.Tensor, K: int):
    """
    Xi_tilde_packed: [B, T-1, K(K+1)/2]
    
    Returns:
        lambdas: [B, T-1, 4] (Top-4 eigenvalues, Upgrade U2)
    """
    # Unpack to dense
    Xi_tilde = unpack_upper_triangular(Xi_tilde_packed, K) # [B, T-1, K, K]
    
    # Compute eigenvalues of Xi * Xi^T
    # Note: Xi is symmetric, so Xi * Xi^T = Xi^2. 
    # The eigenvalues of Xi^2 are the squared eigenvalues of Xi.
    # Alternatively, just use torch.linalg.eigvalsh on Xi and square them, or directly on Xi @ Xi^T.
    cov = torch.matmul(Xi_tilde, Xi_tilde.transpose(-1, -2)) # [B, T-1, K, K]
    
    # eigvalsh returns eigenvalues in ascending order
    eigvals = torch.linalg.eigvalsh(cov) # [B, T-1, K]
    
    # Extract top-4
    top4 = eigvals[..., -4:] # [B, T-1, 4]
    
    # Reverse to descending order for consistency
    top4 = torch.flip(top4, dims=[-1])
    
    return top4
