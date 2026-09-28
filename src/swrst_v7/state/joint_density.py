import torch
import torch.nn.functional as F

def compute_joint_density(
    Xi_can_packed: torch.Tensor,
    R_t: torch.Tensor,
    u_m: torch.Tensor,
    w_m: torch.Tensor,
    inv_du_m: torch.Tensor,
    H_t: torch.Tensor,
    lam_A: float = 0.35,
    eta: float = 0.15
):
    """
    Xi_can_packed: [B, T-1, M, K(K+1)/2]
    R_t: [B, T-1, M]
    u_m: [M]
    w_m: [M-1]
    inv_du_m: [M-1]
    H_t: [B, T-1, M]
    
    Returns:
        P_t: [B, T-1, M-1] (Joint density)
        KL_x: [B] (Pooled KL divergence, H5)
    """
    # H3: Fused D and A pass
    # D_t,m
    diff_Xi = Xi_can_packed[:, :, 1:] - Xi_can_packed[:, :, :-1] # [B, T-1, M-1, K(K+1)/2]
    inv_du_expanded = inv_du_m.view(1, 1, -1, 1)
    
    D_packed = diff_Xi * inv_du_expanded # [B, T-1, M-1, K(K+1)/2]
    
    # ||D||_F
    norm_D = torch.sqrt(torch.tensor(2.0, device=Xi_can_packed.device)) * torch.norm(D_packed, p=2, dim=-1) # [B, T-1, M-1]
    
    # A_t,m
    diff_D = D_packed[:, :, 1:] - D_packed[:, :, :-1] # [B, T-1, M-2, K(K+1)/2]
    du_step2 = u_m[2:] - u_m[:-2] # [M-2]
    inv_du_step2 = 1.0 / (du_step2 + 1e-8)
    inv_du_step2_expanded = inv_du_step2.view(1, 1, -1, 1)
    
    A_packed = diff_D * inv_du_step2_expanded # [B, T-1, M-2, K(K+1)/2]
    
    # Pad A to M-1
    A_front = A_packed[:, :, 0:1, :]
    A_middle = A_packed[:, :, :-1, :] # M-3
    A_back = A_middle[:, :, -1:, :]
    A_padded = torch.cat([A_front, A_middle, A_back], dim=2) # M-1
    
    # ||A||_F
    norm_A = torch.sqrt(torch.tensor(2.0, device=Xi_can_packed.device)) * torch.norm(A_padded, p=2, dim=-1) # [B, T-1, M-1]
    
    # T_t,m
    # ||Xi_can||_F
    norm_Xi = torch.sqrt(torch.tensor(2.0, device=Xi_can_packed.device)) * torch.norm(Xi_can_packed, p=2, dim=-1) # [B, T-1, M]
    mean_norm_Xi_t = torch.mean(norm_Xi, dim=1, keepdim=True) + 1e-8 # [B, 1, M]
    
    # Slice to M-1 to match D and A
    norm_Xi_sliced = norm_Xi[:, :, :-1]
    mean_norm_Xi_t_sliced = mean_norm_Xi_t[:, :, :-1]
    R_t_sliced = R_t[:, :, :-1]
    
    T_tm = (norm_Xi_sliced / mean_norm_Xi_t_sliced) * (1.0 + eta * R_t_sliced) # [B, T-1, M-1]
    
    # G_t,m
    G_tm = (norm_D + lam_A * norm_A) * T_tm # [B, T-1, M-1]
    
    # P_t(m) = Softmax(G_t,m / tau_t)
    tau_t = 1.0 + torch.mean(H_t, dim=2, keepdim=True) # [B, T-1, 1] (Correction 3)
    
    P_t = F.softmax(G_tm / tau_t, dim=2) # [B, T-1, M-1]
    
    # H5: Pooled temporal KL divergence (Correction 5)
    # KL_x = 1/T sum_t sum_m P_t(m) log(P_t(m) / w_m)
    w_m_expanded = w_m.view(1, 1, -1)
    # Clamp to avoid log(0)
    P_t_safe = torch.clamp(P_t, min=1e-8)
    log_ratio = torch.log(P_t_safe / w_m_expanded)
    KL_t = torch.sum(P_t * log_ratio, dim=2) # [B, T-1]
    KL_x = torch.mean(KL_t, dim=1) # [B]
    
    return P_t, KL_x
