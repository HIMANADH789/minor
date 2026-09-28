import torch
import torch.nn.functional as F

def compute_profile_energy_and_state(z: torch.Tensor, D: torch.Tensor, A: torch.Tensor, J: torch.Tensor, H_spec: torch.Tensor, G_eigs: torch.Tensor):
    """
    z, D, A, J: [B, T-1, M, r]
    H_spec: [B, M]
    G_eigs: [B, T-1, 4]
    
    Returns:
        Z_full: [B, T-1, M, 4r + r + 1 + 4] = [B, T-1, M, 5r + 5]
        p_m: [B, M]
    """
    B, T, M, r = z.shape
    
    # U3: Resolution covariance profile
    # Sigma_m = Cov_t(z_t(m)) -> Use diagonal only
    Sigma = torch.var(z, dim=1, unbiased=False) # [B, M, r]
    Sigma_expanded = Sigma.unsqueeze(1).expand(-1, T, -1, -1) # [B, T-1, M, r]
    
    H_spec_expanded = H_spec.unsqueeze(1).unsqueeze(-1).expand(-1, T, -1, -1) # [B, T-1, M, 1]
    
    G_eigs_expanded = G_eigs.unsqueeze(2).expand(-1, -1, M, -1) # [B, T-1, M, 4]
    
    # Assemble full state
    Z_full = torch.cat([z, D, A, J, Sigma_expanded, H_spec_expanded, G_eigs_expanded], dim=-1)
    
    # E_m = sum_t ||Z_full||^2
    E_m = torch.sum(Z_full**2, dim=(1, 3)) # [B, M]
    
    # U2: Spectral persistence profile
    # R_m = sum_t ||z(m+1) - z(m)||^2
    # Note: Using original z for persistence
    diff_z = z[:, :, 1:, :] - z[:, :, :-1, :] # [B, T-1, M-1, r]
    R_m = torch.zeros((B, M), device=z.device, dtype=torch.float32)
    R_m[:, :-1] = torch.sum(diff_z**2, dim=(1, 3)) # [B, M-1]
    R_m[:, -1] = R_m[:, -2] # Replicate last
    
    # p(m) = Softmax(E_m + 0.25 * R_m + H_spec(m))
    logits = E_m + 0.25 * R_m + H_spec
    p_m = F.softmax(logits, dim=-1) # [B, M]
    
    return Z_full, p_m
