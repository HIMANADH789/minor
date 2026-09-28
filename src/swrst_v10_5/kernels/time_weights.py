import torch

def compute_time_weights(Xi: torch.Tensor, w_sorted: torch.Tensor):
    """
    C5: Time weights must use multi-resolution instability.
    w_t = sum_m w_m ||Xi_t(m) - Xi_{t-1}(m)||
    
    Xi: [B, T, M, d] (where d is packed dim, Frobenius norm is Euclidean on packed elements)
    w_sorted: [M]
    Returns:
        w_t: [B, T]
    """
    B, T, M, d = Xi.shape
    tau = 1e-8
    
    # ||Xi_t(m) - Xi_{t-1}(m)||
    diff = Xi[:, 1:, :, :] - Xi[:, :-1, :, :] # [B, T-1, M, d]
    norm_diff = torch.norm(diff, dim=-1) # [B, T-1, M]
    
    # sum_m w_m * ||...||
    w_m_diff = norm_diff * w_sorted.view(1, 1, M)
    w_t_nonzero = torch.sum(w_m_diff, dim=-1) # [B, T-1]
    
    # Normalize
    sum_w = torch.sum(w_t_nonzero, dim=1, keepdim=True) + tau
    w_t_nonzero = w_t_nonzero / sum_w
    
    w_0 = torch.zeros((B, 1), device=Xi.device, dtype=torch.float32)
    w_t = torch.cat([w_0, w_t_nonzero], dim=1) # [B, T]
    
    return w_t
