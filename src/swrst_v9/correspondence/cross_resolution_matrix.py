import torch

def compute_cost_matrix_chunk(S_x: torch.Tensor, S_y: torch.Tensor, tau: float = 1e-8):
    """
    Computes cross-resolution alignment cost C_t(m,n) for a chunk.
    
    S_x: [Nx, T, M, d]
    S_y: [Ny, T, M, d]
    
    Returns:
        C: [Nx, Ny, T, M, M]
    """
    Nx, T, M, d = S_x.shape
    Ny = S_y.shape[0]
    
    # Normalize S to compute cosine similarity A_t
    norm_x = torch.norm(S_x, dim=-1, keepdim=True) + tau
    norm_y = torch.norm(S_y, dim=-1, keepdim=True) + tau
    
    S_x_norm = S_x / norm_x # [Nx, T, M, d]
    S_y_norm = S_y / norm_y # [Ny, T, M, d]
    
    # Compute A_t(m, n) = <S_x(m), S_y(n)>
    # Reshape for bmm: [Nx, T, M, d] -> [Nx*T, M, d]
    # Wait, we need Nx x Ny. Best is einsum.
    # A_t = torch.einsum('xtmd,ytnd->xytmn', S_x_norm, S_y_norm)
    
    # More memory-efficient bmm:
    S_x_flat = S_x_norm.permute(1, 0, 2, 3).reshape(T, Nx*M, d) # [T, Nx*M, d]
    S_y_flat = S_y_norm.permute(1, 0, 2, 3).reshape(T, Ny*M, d) # [T, Ny*M, d]
    
    # [T, Nx*M, Ny*M]
    A_flat = torch.bmm(S_x_flat, S_y_flat.transpose(1, 2))
    
    # Reshape back to [T, Nx, M, Ny, M] -> [Nx, Ny, T, M, M]
    A = A_flat.view(T, Nx, M, Ny, M).permute(1, 3, 0, 2, 4)
    
    # U1: Cross-time correspondence coupling
    # B_t = A_t + 0.15 A_{t-1}
    B = torch.zeros_like(A)
    B[:, :, 0, :, :] = A[:, :, 0, :, :]
    B[:, :, 1:, :, :] = A[:, :, 1:, :, :] + 0.15 * A[:, :, :-1, :, :]
    
    # Normalize B_t to [-1, 1] range approximately (max is 1.15)
    B_norm = B / 1.15
    
    # H4: Fused dot-product + cost (C_t = 1 - B_t)
    C = 1.0 - B_norm
    
    return C
