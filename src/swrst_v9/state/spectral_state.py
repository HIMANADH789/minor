import torch
import torch.nn.functional as F

def compute_spectral_state(z: torch.Tensor, tau: float = 1e-8):
    """
    z: [B, T, M, r]
    
    Returns:
        S: [B, T, M, 4r + 5]
    """
    B, T, M, r = z.shape
    
    # H8: Reuse kinematic buffers
    D = torch.zeros_like(z)
    A = torch.zeros_like(z)
    J = torch.zeros_like(z)
    
    # C1: Second-order one-sided stencils
    # First derivative
    D[:, :, 1:-1, :] = (z[:, :, 2:, :] - z[:, :, :-2, :]) / 2.0
    D[:, :, 0, :] = (-3.0 * z[:, :, 0, :] + 4.0 * z[:, :, 1, :] - z[:, :, 2, :]) / 2.0
    D[:, :, -1, :] = (3.0 * z[:, :, -1, :] - 4.0 * z[:, :, -2, :] + z[:, :, -3, :]) / 2.0
    
    # Second derivative
    A[:, :, 1:-1, :] = z[:, :, 2:, :] - 2.0 * z[:, :, 1:-1, :] + z[:, :, :-2, :]
    A[:, :, 0, :] = z[:, :, 0, :] - 2.0 * z[:, :, 1, :] + z[:, :, 2, :]
    A[:, :, -1, :] = z[:, :, -1, :] - 2.0 * z[:, :, -2, :] + z[:, :, -3, :]
    
    # Third derivative (Jerk) - Centered + Forward/Backward (No replication)
    J[:, :, 1:-1, :] = (A[:, :, 2:, :] - A[:, :, :-2, :]) / 2.0
    J[:, :, 0, :] = A[:, :, 1, :] - A[:, :, 0, :]
    J[:, :, -1, :] = A[:, :, -1, :] - A[:, :, -2, :]
    
    # U3: Resolution entropy correction
    # p_t(m) = ||z_t(m)||^2 / sum_t ||z_t(m)||^2
    z_sq = torch.sum(z**2, dim=-1) # [B, T, M]
    z_sq_sum = torch.sum(z_sq, dim=1, keepdim=True) + tau # [B, 1, M]
    p_t = z_sq / z_sq_sum # [B, T, M]
    
    # H_m = - sum_t p_t * log(p_t)
    H_m = -torch.sum(p_t * torch.log(p_t + tau), dim=1) # [B, M]
    H_m_expanded = H_m.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, -1) # [B, T, M, 1]
    
    # U4: Cross-resolution Gram eigs
    # G_t(m, n) = <z_t(m), z_t(n)>
    G = torch.matmul(z, z.transpose(-1, -2)) # [B, T, M, M]
    tr_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) + tau
    G_norm = G / tr_G
    
    eigvals = torch.linalg.eigvalsh(G_norm) # [B, T, M]
    G_eig = eigvals[..., -4:] # [B, T, 4]
    G_eig_expanded = G_eig.unsqueeze(2).expand(-1, -1, M, -1) # [B, T, M, 4]
    
    # Construct S_t(m) = [z, D, A, J, H_m, G_eig]
    S = torch.cat([z, D, A, J, H_m_expanded, G_eig_expanded], dim=-1) # [B, T, M, 4r + 5]
    
    return S
