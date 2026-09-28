import torch
from typing import Dict, Any, Tuple

def batched_log_domain_sinkhorn(
    u: torch.Tensor, v: torch.Tensor, C: torch.Tensor,
    epsilon: float = 0.01,
    max_iter: int = 120,
    tol: float = 1e-4,
    f_init: torch.Tensor = None,
    g_init: torch.Tensor = None,
    prev_P: torch.Tensor = None,
    incremental_threshold: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Batched Log-domain Sinkhorn for exact optimal transport with warm starts and early exit.
    u: (B, N)
    v: (B, M)
    C: (B, N, M)
    """
    B, N = u.shape
    M = v.shape[1]
    
    metrics = {
        'iterations': [0] * B,
        'marginal_error': [0.0] * B,
        'reused': [False] * B
    }
    
    # Fast path check for each batch item
    active_b = []
    if prev_P is not None and prev_P.shape[1] == N and prev_P.shape[2] == M:
        marg_u_prev = prev_P.sum(dim=-1)
        marg_v_prev = prev_P.sum(dim=-2)
        
        err_u = torch.max(torch.abs(marg_u_prev - u), dim=1).values
        err_v = torch.max(torch.abs(marg_v_prev - v), dim=1).values
        max_err = torch.maximum(err_u, err_v)
        
        for i in range(B):
            if max_err[i] < incremental_threshold:
                metrics['reused'][i] = True
                metrics['marginal_error'][i] = max_err[i].item()
            else:
                active_b.append(i)
    else:
        active_b = list(range(B))
        
    log_u = torch.log(u)
    log_v = torch.log(v)
    
    f = f_init if f_init is not None else torch.zeros_like(u)
    g = g_init if g_init is not None else torch.zeros_like(v)
    
    # Only compute Sinkhorn for active elements
    if active_b:
        active_idx = torch.tensor(active_b, device=u.device)
        C_act = C[active_idx]
        log_u_act = log_u[active_idx]
        log_v_act = log_v[active_idx]
        f_act = f[active_idx]
        g_act = g[active_idx]
        
        # Adaptive max iter: we run up to max_iter, but we check convergence
        for it in range(max_iter):
            # g update
            # g = epsilon * (log_v - logsumexp((f.unsqueeze(-1) - C) / epsilon, dim=1))
            arg_v = (f_act.unsqueeze(-1) - C_act) / epsilon
            g_act = epsilon * (log_v_act - torch.logsumexp(arg_v, dim=1))
            
            # f update
            arg_u = (g_act.unsqueeze(1) - C_act) / epsilon
            f_act = epsilon * (log_u_act - torch.logsumexp(arg_u, dim=2))
            
            if (it + 1) % 10 == 0:
                # Check convergence
                # P = exp((f + g - C)/eps)
                arg_P = (f_act.unsqueeze(-1) + g_act.unsqueeze(1) - C_act) / epsilon
                P_act = torch.exp(arg_P)
                
                marg_u = P_act.sum(dim=-1)
                err = torch.max(torch.abs(marg_u - torch.exp(log_u_act)), dim=1).values
                if err.max() < tol:
                    break
                    
        arg_P = (f_act.unsqueeze(-1) + g_act.unsqueeze(1) - C_act) / epsilon
        P_act = torch.exp(arg_P)
        
        # Write back to full batch tensors
        f[active_idx] = f_act
        g[active_idx] = g_act
        
        if prev_P is None or prev_P.shape[1] != N or prev_P.shape[2] != M:
            P_out = torch.zeros((B, N, M), device=u.device)
            P_out[active_idx] = P_act
        else:
            P_out = prev_P.clone()
            P_out[active_idx] = P_act
            
        for idx in active_b:
            metrics['iterations'][idx] = it + 1
    else:
        P_out = prev_P.clone()
        
    return P_out, f, g, metrics
