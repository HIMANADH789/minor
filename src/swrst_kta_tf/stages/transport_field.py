import torch
import torch.nn.functional as F

def exact_ot_1d_batched(mu: torch.Tensor, nu: torch.Tensor, v: torch.Tensor, cost: torch.Tensor) -> tuple:
    # mu, nu: (B, K)
    # v: (K,)
    # cost: (K, K)
    B, K = mu.shape
    
    # H7: CDFs using torch.cumsum
    F_mu = torch.cumsum(mu, dim=-1)
    F_nu = torch.cumsum(nu, dim=-1)
    
    # Shifted CDFs for interval computation: (B, K+1)
    F_mu_pad = F.pad(F_mu, (1,0), value=0.0)
    F_nu_pad = F.pad(F_nu, (1,0), value=0.0)
    
    # P_{ij} = max(0, min(F_mu[i], F_nu[j]) - max(F_mu[i-1], F_nu[j-1]))
    F_mu_i = F_mu_pad[:, 1:].unsqueeze(2).expand(B, K, K)
    F_mu_im1 = F_mu_pad[:, :-1].unsqueeze(2).expand(B, K, K)
    F_nu_j = F_nu_pad[:, 1:].unsqueeze(1).expand(B, K, K)
    F_nu_jm1 = F_nu_pad[:, :-1].unsqueeze(1).expand(B, K, K)
    
    P = torch.clamp(torch.minimum(F_mu_i, F_nu_j) - torch.maximum(F_mu_im1, F_nu_jm1), min=0.0)
    
    W = (P * cost.unsqueeze(0)).sum(dim=(-2,-1))
    return P, W

def transport_field(MU: torch.Tensor, MU_list: list[torch.Tensor], bin_edges: torch.Tensor) -> tuple:
    """
    Stage 3: Exact 1D optimal transport field
    MU: (N, n_pairs, 2, K)
    MU_list: list of MU per scale
    Returns: 
        R: (N, n_pairs, K, K) transport residuals
        W_dist: (N, n_pairs)
        R_by_scale: list of (N, K*K*n_p)
    """
    N, n_pairs, _, K = MU.shape
    
    # Compute bin centers from bin_edges
    v = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # H6: Pre-compute cost matrix C and pin to GPU
    cost = (v.unsqueeze(0) - v.unsqueeze(1))**2
    
    # Flatten MU
    mu_flat = MU[:, :, 0, :].reshape(N * n_pairs, K)
    nu_flat = MU[:, :, 1, :].reshape(N * n_pairs, K)
    
    R_flat = torch.zeros((N * n_pairs, K, K), dtype=torch.float32, device=MU.device)
    W_flat = torch.zeros((N * n_pairs), dtype=torch.float32, device=MU.device)
    
    # H5: Chunked exact OT to stay in L2 cache
    chunk_size = 512
    for i in range(0, N * n_pairs, chunk_size):
        mu_chunk = mu_flat[i:i+chunk_size]
        nu_chunk = nu_flat[i:i+chunk_size]
        
        P_chunk, W_chunk = exact_ot_1d_batched(mu_chunk, nu_chunk, v, cost)
        
        # Independence coupling: Q_{ij} = mu_i * nu_j
        Q_chunk = mu_chunk.unsqueeze(2) * nu_chunk.unsqueeze(1)
        
        # Transport residual: R = P - Q
        R_chunk = P_chunk - Q_chunk
        
        R_flat[i:i+chunk_size] = R_chunk
        W_flat[i:i+chunk_size] = W_chunk
        
    R = R_flat.view(N, n_pairs, K, K)
    W_dist = W_flat.view(N, n_pairs)
    
    R_by_scale = []
    offset = 0
    for mu_s in MU_list:
        n_p = mu_s.shape[1]
        R_by_scale.append(R[:, offset:offset+n_p, :, :].reshape(N, -1))
        offset += n_p
    
    return R, W_dist, R_by_scale
