import torch

def compute_continuous_barycenter(Xi_packed: torch.Tensor, p_t: torch.Tensor, w_hat_m: torch.Tensor):
    """
    Xi_packed: [B, T-1, M, K(K-1)/2]
    p_t: [B, T-1, M-1]
    w_hat_m: [M-1]
    
    Returns:
        Xi_tilde: [B, T-1, K(K-1)/2] (Continuous transport summary)
    """
    # Average of adjacent resolutions
    Xi_mid = 0.5 * (Xi_packed[:, :, :-1] + Xi_packed[:, :, 1:]) # [B, T-1, M-1, K(K-1)/2]
    
    # Weight by intrinsic density and log-grid width
    # p_t: [B, T-1, M-1], w_hat_m: [M-1]
    w_combined = p_t * w_hat_m.view(1, 1, -1) # [B, T-1, M-1]
    
    w_combined_expanded = w_combined.unsqueeze(-1) # [B, T-1, M-1, 1]
    
    # Integrate
    Xi_tilde = torch.sum(w_combined_expanded * Xi_mid, dim=2) # [B, T-1, K(K-1)/2]
    
    return Xi_tilde
