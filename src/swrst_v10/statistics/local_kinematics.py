import torch

def compute_local_kinematics(z: torch.Tensor, tau: float = 1e-8):
    """
    z: [B, T, M, r]
    
    Returns:
        S: [B, T, M, 5r + 4]
    """
    B, T, M, r = z.shape
    
    # H8: Reuse state tensors via shared views where possible
    # We will allocate them here and concatenate once.
    D = torch.zeros_like(z)
    A = torch.zeros_like(z)
    J = torch.zeros_like(z)
    R = torch.zeros_like(z)
    
    # D_t(m) = (z(m+1) - z(m-1)) / 2
    D[:, :, 1:-1, :] = (z[:, :, 2:, :] - z[:, :, :-2, :]) / 2.0
    D[:, :, 0, :] = (-3.0 * z[:, :, 0, :] + 4.0 * z[:, :, 1, :] - z[:, :, 2, :]) / 2.0
    D[:, :, -1, :] = (3.0 * z[:, :, -1, :] - 4.0 * z[:, :, -2, :] + z[:, :, -3, :]) / 2.0
    
    # A_t(m) = z(m+1) - 2z(m) + z(m-1)
    A[:, :, 1:-1, :] = z[:, :, 2:, :] - 2.0 * z[:, :, 1:-1, :] + z[:, :, :-2, :]
    A[:, :, 0, :] = z[:, :, 0, :] - 2.0 * z[:, :, 1, :] + z[:, :, 2, :]
    A[:, :, -1, :] = z[:, :, -1, :] - 2.0 * z[:, :, -2, :] + z[:, :, -3, :]
    
    # C2: J_t(m) = (A_t(m+1) - A_t(m-1)) / 2 (centered)
    J[:, :, 1:-1, :] = (A[:, :, 2:, :] - A[:, :, :-2, :]) / 2.0
    J[:, :, 0, :] = (-3.0 * A[:, :, 0, :] + 4.0 * A[:, :, 1, :] - A[:, :, 2, :]) / 2.0
    J[:, :, -1, :] = (3.0 * A[:, :, -1, :] - 4.0 * A[:, :, -2, :] + A[:, :, -3, :]) / 2.0
    
    # U1: Transport persistence channel R_t(m) = z_t(m+1) - z_t(m)
    R[:, :, :-1, :] = z[:, :, 1:, :] - z[:, :, :-1, :]
    R[:, :, -1, :] = R[:, :, -2, :]
    
    # U2: Cross-resolution topology eigs
    # G_t(m, n) = <z_t(m), z_t(n)>
    G = torch.matmul(z, z.transpose(-1, -2)) # [B, T, M, M]
    tr_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) + tau
    G_norm = G / tr_G
    
    # H9: Eigs using symmetric matrix solver directly on G_norm
    eigvals = torch.linalg.eigvalsh(G_norm) # [B, T, M]
    G_eig = eigvals[..., -4:] # [B, T, 4]
    G_eig_expanded = G_eig.unsqueeze(2).expand(-1, -1, M, -1) # [B, T, M, 4]
    
    # Cat once
    S = torch.cat([z, D, A, J, R, G_eig_expanded], dim=-1) # [B, T, M, 5r + 4]
    
    return S
