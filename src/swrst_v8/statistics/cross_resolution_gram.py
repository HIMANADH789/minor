import torch

def compute_cross_resolution_gram(z: torch.Tensor, tau: float = 1e-8):
    """
    z: [B, T-1, M, r]
    
    Returns:
        eigs_top4: [B, T-1, 4]
    """
    # G_t(m, n) = <z_t(m), z_t(n)>
    G = torch.matmul(z, z.transpose(-1, -2)) # [B, T-1, M, M]
    
    # C5: Normalize by tr(G_t) + tau
    tr_G = torch.diagonal(G, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) + tau # [B, T-1, 1, 1]
    
    G_norm = G / tr_G
    
    # Extract eigenvalues
    eigvals = torch.linalg.eigvalsh(G_norm) # [B, T-1, M]
    
    # Top-4
    eigs_top4 = eigvals[..., -4:] # [B, T-1, 4]
    
    return eigs_top4
