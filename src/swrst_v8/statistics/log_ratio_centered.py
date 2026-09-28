import torch
from src.swrst_v4.transport.weighted_field import pack_upper_triangular

# Cache for the double-centering projection matrix
_CENTERING_MAT_CACHE = {}

def get_double_centering_matrix(K: int, device: torch.device):
    if K in _CENTERING_MAT_CACHE:
        return _CENTERING_MAT_CACHE[K].to(device)
        
    D = K * (K + 1) // 2
    A = torch.zeros((D, K, K), dtype=torch.float32)
    
    idx = 0
    for i in range(K):
        for j in range(i, K):
            A[idx, i, j] = 1.0
            A[idx, j, i] = 1.0
            idx += 1
            
    r = A.mean(dim=2, keepdim=True)
    c = A.mean(dim=1, keepdim=True)
    g = A.mean(dim=(1,2), keepdim=True)
    
    A_tilde = A - r - c + g
    
    C_mat = torch.zeros((D, D), dtype=torch.float32)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            C_mat[:, idx] = A_tilde[:, i, j]
            idx += 1
            
    _CENTERING_MAT_CACHE[K] = C_mat
    return C_mat.to(device)

def compute_log_ratio_fisher(P_t: torch.Tensor, Q_t: torch.Tensor, K: int, tau: float = 1e-8):
    """
    P_t: [B, T-1, M, K, K]
    Q_t: [B, T-1, K, K]
    
    Returns:
        L_F_packed: [B, T-1, M, K(K+1)/2]
    """
    # H1: Upper triangular packing BEFORE log-ratio
    P_packed = pack_upper_triangular(P_t) # [B, T-1, M, D]
    Q_packed = pack_upper_triangular(Q_t) # [B, T-1, D]
    Q_packed_exp = Q_packed.unsqueeze(2) # [B, T-1, 1, D]
    
    # H2: Fused centered log-ratio kernel
    # C1: L = log(P + tau) - log(Q + tau) (Natural parameter)
    log_P = torch.log(torch.clamp(P_packed, min=1e-8) + tau)
    log_Q = torch.log(torch.clamp(Q_packed_exp, min=1e-8) + tau)
    
    L_packed = log_P - log_Q # [B, T-1, M, D]
    
    # Apply double centering directly in packed space via projection matrix (H2 optimized)
    C_mat = get_double_centering_matrix(K, P_t.device)
    L_tilde_packed = torch.matmul(L_packed, C_mat) # [B, T-1, M, D]
    
    # C2: Fisher log weighting L_F = sqrt(P + tau) * L_tilde / sqrt(Q + tau)
    sqrt_P = torch.sqrt(torch.clamp(P_packed, min=1e-8) + tau)
    inv_sqrt_Q = 1.0 / torch.sqrt(torch.clamp(Q_packed_exp, min=1e-8) + tau)
    
    L_F_packed = sqrt_P * inv_sqrt_Q * L_tilde_packed # [B, T-1, M, D]
    
    return L_F_packed
