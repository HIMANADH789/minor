import torch

def compute_jap_metric(
    Xi_can_1: torch.Tensor,
    Xi_can_2: torch.Tensor,
    P_1: torch.Tensor,
    P_2: torch.Tensor,
    w_m: torch.Tensor,
    gamma: float = 0.10,
    tau: float = 1e-8,
    chunk_size: int = 128
):
    """
    Xi_can: [N, T-1, M, D]
    P: [N, T-1, M-1]
    w_m: [M-1]
    
    Returns:
        d2: [N1, N2]
    """
    N1 = Xi_can_1.size(0)
    N2 = Xi_can_2.size(0)
    
    # Slice Xi_can to match P_t dimensions (M-1)
    Xi_1 = Xi_can_1[:, :, :-1, :] # [N1, T-1, M-1, D]
    Xi_2 = Xi_can_2[:, :, :-1, :] # [N2, T-1, M-1, D]
    
    # Upgrade U2: Path persistence term (O(N) computation)
    # L_x = sum_t sum_m || Xi_{t,m+1} - Xi_{t,m} ||_F
    diff_Xi_1 = Xi_can_1[:, :, 1:, :] - Xi_can_1[:, :, :-1, :]
    L_1 = torch.sum(torch.norm(diff_Xi_1, p=2, dim=-1), dim=(1, 2)) # [N1]
    
    diff_Xi_2 = Xi_can_2[:, :, 1:, :] - Xi_can_2[:, :, :-1, :]
    L_2 = torch.sum(torch.norm(diff_Xi_2, p=2, dim=-1), dim=(1, 2)) # [N2]
    
    d2_mat = torch.zeros((N1, N2), device=Xi_can_1.device, dtype=torch.float32)
    
    w_m_expanded = w_m.view(1, 1, -1) # [1, 1, M-1]
    
    # Pre-square norms for distance expansion: ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2 X dot Y
    # Actually, because the weights omega_{t,m}^{xy} depend on both x and y,
    # we have to compute omega * ||X||^2 + omega * ||Y||^2 - 2 omega * (X dot Y)
    Xi1_sq = torch.sum(Xi_1**2, dim=-1) # [N1, T-1, M-1]
    Xi2_sq = torch.sum(Xi_2**2, dim=-1) # [N2, T-1, M-1]
    
    # H7: Pairwise JAP chunking
    for i in range(0, N1, chunk_size):
        end_i = min(i + chunk_size, N1)
        
        P_i = P_1[i:end_i] # [chunk1, T-1, M-1]
        Xi1_i = Xi_1[i:end_i] # [chunk1, T-1, M-1, D]
        Xi1_sq_i = Xi1_sq[i:end_i] # [chunk1, T-1, M-1]
        
        for j in range(0, N2, chunk_size):
            end_j = min(j + chunk_size, N2)
            
            P_j = P_2[j:end_j] # [chunk2, T-1, M-1]
            Xi2_j = Xi_2[j:end_j] # [chunk2, T-1, M-1, D]
            Xi2_sq_j = Xi2_sq[j:end_j] # [chunk2, T-1, M-1]
            
            # Correction 4: Symmetric overlap normalization (H8 Vectorized)
            # P_i: [chunk1, 1, T-1, M-1], P_j: [1, chunk2, T-1, M-1]
            P_ij_sqrt = torch.sqrt(P_i.unsqueeze(1) * P_j.unsqueeze(0)) # [chunk1, chunk2, T-1, M-1]
            
            w_P_ij = w_m_expanded.unsqueeze(0) * P_ij_sqrt # [chunk1, chunk2, T-1, M-1]
            
            norm_factor = torch.sum(w_P_ij, dim=-1, keepdim=True) + tau # [chunk1, chunk2, T-1, 1]
            
            omega_ij = w_P_ij / norm_factor # [chunk1, chunk2, T-1, M-1]
            
            # Compute distances
            # Xi1_sq_i: [chunk1, 1, T-1, M-1]
            # Xi2_sq_j: [1, chunk2, T-1, M-1]
            dist2_base = Xi1_sq_i.unsqueeze(1) + Xi2_sq_j.unsqueeze(0) # [chunk1, chunk2, T-1, M-1]
            
            # Dot product: Xi1_i [chunk1, T-1, M-1, D] x Xi2_j [chunk2, T-1, M-1, D]
            # We want [chunk1, chunk2, T-1, M-1]
            # using einsum for efficient batched dot product over D
            dot_ij = torch.einsum('itmd,jtmd->ijtm', Xi1_i, Xi2_j)
            
            dist2_full = dist2_base - 2.0 * dot_ij
            dist2_full = torch.clamp(dist2_full, min=0.0)
            
            # Integrate with omega
            d2_chunk = torch.sum(omega_ij * dist2_full, dim=(2, 3)) # [chunk1, chunk2]
            
            d2_mat[i:end_i, j:end_j] = d2_chunk
            
    # U2: Path persistence term
    L1_exp = L_1.unsqueeze(1)
    L2_exp = L_2.unsqueeze(0)
    L_diff2 = (L1_exp - L2_exp)**2
    
    d2_mat = d2_mat + gamma * L_diff2
    
    return d2_mat
