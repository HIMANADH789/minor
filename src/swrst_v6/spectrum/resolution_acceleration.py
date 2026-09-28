import torch

def compute_resolution_acceleration(V_packed: torch.Tensor, u_m: torch.Tensor):
    """
    V_packed: [B, T-1, M-1, K(K-1)/2]
    u_m: [M]
    
    Returns:
        A_padded: [B, T-1, M-1, K(K-1)/2] (Correction B: padded to match V)
        norm_A: [B, T-1, M-1] (H7: Fused Frobenius norm)
    """
    # A_t = (V_t(m+1) - V_t(m)) / (u_{m+2} - u_m)
    diff_V = V_packed[:, :, 1:] - V_packed[:, :, :-1] # [B, T-1, M-2, K(K-1)/2]
    
    # u_{m+2} - u_m for m=0 to M-3
    du_step2 = u_m[2:] - u_m[:-2] # [M-2]
    inv_du_step2 = 1.0 / (du_step2 + 1e-8)
    
    inv_du_expanded = inv_du_step2.view(1, 1, -1, 1) # [B, T-1, M-2, K(K-1)/2]
    
    A_packed = diff_V * inv_du_expanded # [B, T-1, M-2, K(K-1)/2]
    
    # Correction B: Pad acceleration to match V's resolution dimension M-1
    # A_t(0) = A_t(1), A_t(M-2) = A_t(M-3)
    # A_packed has M-2 elements. To satisfy both and get M-1, we use M-3 elements from A_packed.
    A_front = A_packed[:, :, 0:1, :]
    A_middle = A_packed[:, :, :-1, :] # M-3 elements
    A_back = A_middle[:, :, -1:, :]
    
    A_padded = torch.cat([A_front, A_middle, A_back], dim=2) # 1 + (M-3) + 1 = M-1 elements
    
    # H7: Fused Frobenius norm
    norm_A = torch.sqrt(torch.tensor(2.0, device=A_padded.device)) * torch.norm(A_padded, p=2, dim=-1) # [B, T-1, M-1]
    
    return A_padded, norm_A
