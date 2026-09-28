import torch
from ..config import SWRSTConfig

def compute_similarity(P_A: torch.Tensor, P_B: torch.Tensor, config: SWRSTConfig) -> torch.Tensor:
    """
    Computes the similarity between two transport observations.
    Similarity = exp(-gamma_F * Fisher) * exp(-gamma_C * Cosine)
    P_A: (M, L_A, D_A) - Transport observations for sequence A
    P_B: (N, L_B, D_B) - Transport observations for sequence B
    
    Since D_A and D_B might differ due to locality, we compute distance on the overlapping
    topological support or just flatten if they are already projected to a common space.
    Actually, to compare transport matrices of different sizes, we usually flatten to max size 
    or compare the marginalized structure. 
    But since SWRST-CE incrementally grows the structure, we can pad to the maximum size of the two.
    """
    M, L_A, D_A = P_A.shape
    N, L_B, D_B = P_B.shape
    
    # Pad to maximum dimension for comparison
    D = max(D_A, D_B)
    
    if D_A < D:
        P_A_pad = torch.zeros((M, L_A, D), device=P_A.device, dtype=P_A.dtype)
        P_A_pad[:, :, :D_A] = P_A
    else:
        P_A_pad = P_A
        
    if D_B < D:
        P_B_pad = torch.zeros((N, L_B, D), device=P_B.device, dtype=P_B.dtype)
        P_B_pad[:, :, :D_B] = P_B
    else:
        P_B_pad = P_B
        
    # Fisher (Hellinger) distance
    # H^2 = 1 - sum(sqrt(P_A * P_B))
    sqrt_pA = torch.sqrt(torch.clamp(P_A_pad, min=0))
    sqrt_pB = torch.sqrt(torch.clamp(P_B_pad, min=0))
    
    norm_pA_sq = sqrt_pA.pow(2).sum(dim=-1) # (M, L_A)
    norm_pB_sq = sqrt_pB.pow(2).sum(dim=-1) # (N, L_B)
    
    dot_sqrt = torch.einsum('mid,njd->minj', sqrt_pA, sqrt_pB)
    
    sq_dist_fisher = norm_pA_sq.unsqueeze(2).unsqueeze(3) + norm_pB_sq.unsqueeze(0).unsqueeze(1) - 2 * dot_sqrt
    sq_dist_fisher = torch.clamp(sq_dist_fisher, min=0.0)
    
    # Cosine distance
    pA_norm = P_A_pad / (P_A_pad.norm(dim=-1, keepdim=True) + 1e-8)
    pB_norm = P_B_pad / (P_B_pad.norm(dim=-1, keepdim=True) + 1e-8)
    
    cos_sim = torch.einsum('mid,njd->minj', pA_norm, pB_norm)
    cos_dist = 1.0 - cos_sim
    cos_dist = torch.clamp(cos_dist, min=0.0)
    
    # Combined Similarity
    sim_F = torch.exp(-config.cck_gamma_fisher * sq_dist_fisher)
    sim_C = torch.exp(-config.cck_gamma_cosine * cos_dist)
    
    return sim_F * sim_C
