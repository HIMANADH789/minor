import torch

def _compute_adaptive_fusion(K_list: list, tau: float = 1.0):
    num_kernels = len(K_list)
    N1 = K_list[0].size(0)
    
    h_vals = []
    log_K_vals = []
    for k in range(num_kernels):
        K_mat = K_list[k]
        
        K_sum = torch.sum(K_mat, dim=1, keepdim=True) + 1e-8
        K_tilde = K_mat / K_sum
        h = -torch.sum(K_tilde * torch.log(K_tilde + 1e-8), dim=1)
        h_vals.append(h)
        log_K_vals.append(torch.log(K_mat + 1e-8))
        
    h_tensor = torch.stack(h_vals, dim=0) # [num_kernels, N1]
    
    # Debug: print average entropies
    mean_h = torch.mean(h_tensor, dim=1)
    print(f"DEBUG: Mean Entropies: {mean_h.tolist()}")
    
    w_unnorm = torch.exp(-h_tensor / tau)
    w = w_unnorm / (torch.sum(w_unnorm, dim=0, keepdim=True) + 1e-8) # [num_kernels, N1]
    
    # Debug: print average weights
    mean_w = torch.mean(w, dim=1)
    print(f"DEBUG: Adaptive Fusion Mean Weights: {mean_w.tolist()}")
    
    w = w.unsqueeze(-1) # [num_kernels, N1, 1]
    
    log_K_fused = torch.zeros_like(log_K_vals[0])
    for k in range(num_kernels):
        log_K_fused += w[k] * log_K_vals[k]
            
    K_fused = torch.exp(log_K_fused)
    return K_fused

compute_adaptive_fusion = _compute_adaptive_fusion
