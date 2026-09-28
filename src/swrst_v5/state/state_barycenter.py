import torch
import torch.nn.functional as F

def compute_state_barycenter(Z_t: torch.Tensor):
    """
    Z_t: [B, T-1, M, 165]
    Format: [F_t (78), dF_t (78), R_t (1), C_t (6), K_t (1), h_t (1)]
    
    Returns:
        Z_bar_t: [B, T-1, 165]
    """
    # Slice components for score calculation
    dF_t_packed = Z_t[..., 78:156]
    C_t = Z_t[..., 157:163]
    K_t = Z_t[..., 163]
    h_t = Z_t[..., 164]
    
    # Compute norms
    # || \Delta F_t ||_F = sqrt(2) * || dF_packed ||_2 (since diagonal is 0)
    norm_dF = torch.sqrt(torch.tensor(2.0, device=Z_t.device)) * torch.norm(dF_t_packed, p=2, dim=-1) # [B, T-1, M]
    norm_C = torch.norm(C_t, p=2, dim=-1) # [B, T-1, M]
    
    # Compute score (Correction C)
    Phi_t = 0.30 * h_t + 0.25 * K_t + 0.30 * norm_C + 0.15 * norm_dF # [B, T-1, M]
    
    # Continuous softmax
    pi_t = F.softmax(Phi_t, dim=-1) # [B, T-1, M]
    
    # Barycenter
    pi_t_exp = pi_t.unsqueeze(-1) # [B, T-1, M, 1]
    Z_bar_t = torch.sum(pi_t_exp * Z_t, dim=2) # [B, T-1, 165]
    
    return Z_bar_t
