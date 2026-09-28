import torch
import os
import numpy as np

def compute_cost_matrix(K=12):
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = (i - j)**2
    cost = cost / cost.max()
    return cost

def compute_deterministic_ot(mu_t: torch.Tensor, mu_t_plus_1: torch.Tensor, C: torch.Tensor, eps: float = 0.001):
    """
    Computes batched Sinkhorn approximation to deterministic OT.
    mu_t: [B, T-1, K]
    mu_t_plus_1: [B, T-1, K]
    C: [K, K]
    eps: 0.001 for near-exact deterministic approximation
    """
    # log-domain Sinkhorn on GPU
    L = -C.unsqueeze(0).unsqueeze(0) / eps
    
    u = torch.zeros_like(mu_t)
    v = torch.zeros_like(mu_t_plus_1)
    
    log_mu = torch.log(mu_t + 1e-12)
    log_nu = torch.log(mu_t_plus_1 + 1e-12)
    
    for _ in range(50):
        # u update
        arg1 = L + v.unsqueeze(-2)
        u = log_mu - torch.logsumexp(arg1, dim=-1)
        
        # v update
        arg2 = L.transpose(-1, -2) + u.unsqueeze(-2)
        v = log_nu - torch.logsumexp(arg2, dim=-1)
        
    P_sinkhorn = torch.exp(u.unsqueeze(-1) + L + v.unsqueeze(-2))
    return P_sinkhorn

def get_cached_ot(X, mu, cache_path=r'C:\temp\ECG_Benchmark\cache\p_det_cache.npy'):
    """
    Using fast GPU Sinkhorn, we don't strictly need CPU caching, 
    but we keep the interface and cache to disk to save memory across runs.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        print("Loading cached OT plans...")
        return torch.tensor(np.load(cache_path), dtype=torch.float32, device=mu.device)
    
    print("Computing batched GPU Sinkhorn OT plans...")
    C = torch.tensor(compute_cost_matrix(mu.size(-1)), dtype=torch.float32, device=mu.device)
    
    # Batch process to prevent VRAM OOM if dataset is huge
    B = mu.size(0)
    batch_size = 256
    P_det_list = []
    
    with torch.no_grad():
        for i in range(0, B, batch_size):
            m1 = mu[i:i+batch_size, :-1, :]
            m2 = mu[i:i+batch_size, 1:, :]
            P_batch = compute_deterministic_ot(m1, m2, C, eps=0.001)
            P_det_list.append(P_batch.cpu())
            
    P_det = torch.cat(P_det_list, dim=0)
    np.save(cache_path, P_det.numpy())
    
    return P_det.to(mu.device)
