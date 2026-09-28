import torch
import torch.nn.functional as F

def compute_resolution_profile(norm_V: torch.Tensor, norm_A: torch.Tensor, H_t: torch.Tensor, lam: float = 0.35):
    """
    norm_V: [B, T-1, M-1]
    norm_A: [B, T-1, M-1]
    H_t: [B, T-1, M]
    
    Returns:
        p_t: [B, T-1, M-1] (Intrinsic resolution density)
    """
    g_t = norm_V + lam * norm_A # [B, T-1, M-1]
    
    # Correction C: Order-aware profile temperature
    # We use H_t corresponding to the m-th index
    H_t_sliced = H_t[:, :, :-1] # [B, T-1, M-1]
    tau_t = 1.0 + H_t_sliced
    
    # Softmax with temperature
    logits = g_t / tau_t
    p_t = F.softmax(logits, dim=-1) # [B, T-1, M-1]
    
    return p_t
