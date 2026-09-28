import torch
import torch.nn.functional as F

def compute_discriminative_profile(z: torch.Tensor):
    """
    z: [B, T, M, r]
    
    Returns:
        p_tilde: [B, M] (Persistence-aware OT marginals)
        p: [B, M] (Raw discriminative profile)
    """
    B, T, M, r = z.shape
    
    # E_m = sum_t ||z_t(m)||^2
    E_m = torch.sum(z**2, dim=(1, 3)) # [B, M]
    
    # R_m = sum_t ||z_t(m+1) - z_t(m)||^2
    diff_z = z[:, :, 1:, :] - z[:, :, :-1, :] # [B, T, M-1, r]
    R_m = torch.zeros((B, M), device=z.device, dtype=torch.float32)
    R_m[:, :-1] = torch.sum(diff_z**2, dim=(1, 3)) # [B, M-1]
    R_m[:, -1] = R_m[:, -2] # Replicate final for size consistency
    
    # S_m = - sum_j Var_t(z_{t,j}(m))
    var_t = torch.var(z, dim=1, unbiased=False) # [B, M, r]
    S_m = -torch.sum(var_t, dim=-1) # [B, M]
    
    # p(m) = Softmax(E_m + 0.30 R_m + 0.20 S_m)
    logits = E_m + 0.30 * R_m + 0.20 * S_m
    p = F.softmax(logits, dim=-1) # [B, M]
    
    # U2: Resolution persistence prior in OT
    # p_tilde(m) = p(m) * (1 + 0.2 R_m)
    p_tilde_unnorm = p * (1.0 + 0.20 * R_m)
    p_tilde = p_tilde_unnorm / torch.sum(p_tilde_unnorm, dim=-1, keepdim=True)
    
    return p_tilde, p
