import torch
from ..config import SWRSTConfig
from .similarity import compute_similarity

def compute_chunked_cck(
    P_A: torch.Tensor, 
    P_B: torch.Tensor,
    loc_A: torch.Tensor,
    loc_B: torch.Tensor,
    conf_A: torch.Tensor,
    conf_B: torch.Tensor,
    mask_A: torch.Tensor,
    mask_B: torch.Tensor,
    config: SWRSTConfig
) -> torch.Tensor:
    """
    Computes the Continuous Correspondence Kernel between two chunks of transport sequences.
    Kernel = Similarity * Compatibility * Confidence
    """
    M, L = P_A.shape[0], P_A.shape[1]
    N, L_B = P_B.shape[0], P_B.shape[1]
    
    # 1. Similarity
    # Flatten spatial dims to size D
    # P_A is (M, L, D, D), we flatten the last two for similarity
    D_A = P_A.shape[2] * P_A.shape[3]
    D_B = P_B.shape[2] * P_B.shape[3]
    pA = P_A.view(M, L, D_A)
    pB = P_B.view(N, L_B, D_B)
    
    sim = compute_similarity(pA, pB, config) # (M, L, N, L_B)
    
    # 2. Compatibility
    # Gaussian compatibility over normalized locality (lambda / lambda_stop)
    # lambda_stop is the max locality for that sequence, which is the last valid locality
    
    # Find max valid locality per sequence
    loc_A_max = loc_A.max(dim=1, keepdim=True)[0] + 1e-8 # (M, 1)
    loc_B_max = loc_B.max(dim=1, keepdim=True)[0] + 1e-8 # (N, 1)
    
    norm_loc_A = loc_A / loc_A_max # (M, L)
    norm_loc_B = loc_B / loc_B_max # (N, L_B)
    
    # Gaussian distance
    d_loc = (norm_loc_A.unsqueeze(2).unsqueeze(3) - norm_loc_B.unsqueeze(0).unsqueeze(1)) ** 2
    gamma_loc = 5.0 # Fixed or config
    compat = torch.exp(-gamma_loc * d_loc)
    
    # 3. Confidence
    # Combine confidences
    conf = conf_A.unsqueeze(2).unsqueeze(3) * conf_B.unsqueeze(0).unsqueeze(1)
    
    # Final Kernel elements
    w_ij = sim * compat * conf
    
    # Zero out invalid entries
    valid_mask = mask_A.unsqueeze(2).unsqueeze(3) & mask_B.unsqueeze(0).unsqueeze(1)
    w_ij = w_ij * valid_mask.float()
    
    # The final kernel value for (M, N) is the sum over the alignment
    # (Since it's correspondence, we can sum or soft-dtw over the paths. Standard is sum for CCK)
    K = w_ij.sum(dim=(1, 3))
    
    return K
