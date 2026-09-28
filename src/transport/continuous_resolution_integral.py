from src.utils.safe_eigh import safe_eigh
import torch
import torch.nn as nn

def log_domain_sinkhorn_barycenter(C, eps, n_iters=10):
    """
    Log-domain Sinkhorn for numerical stability.
    C: (B, 3, K, K)
    eps: (B, 3, 1, 1)
    r, c are marginals, assumed uniform (1/K)
    """
    B, num_eps, K, _ = C.shape
    device = C.device
    
    # Uniform marginals in log domain
    log_u = torch.zeros(B, num_eps, K, 1, device=device)
    log_v = torch.zeros(B, num_eps, 1, K, device=device)
    
    log_r = torch.log(torch.ones(B, num_eps, K, 1, device=device) / K)
    log_c = torch.log(torch.ones(B, num_eps, 1, K, device=device) / K)
    
    # Cost scaled by temp
    M = -C / eps # (B, 3, K, K)
    
    for _ in range(n_iters):
        log_u = log_r - torch.logsumexp(M + log_v, dim=-1, keepdim=True)
        log_v = log_c - torch.logsumexp(M + log_u, dim=-2, keepdim=True)
        
    log_P = log_u + M + log_v
    log_P = torch.nan_to_num(log_P, nan=-100.0, posinf=0.0, neginf=-100.0)
    log_P = torch.clamp(log_P, max=0.0)
    return torch.exp(log_P)

class ContinuousResolutionIntegral(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
        
    def forward(self, C_batch, eps_batch, H_t=None):
        """
        C_batch: (B, 3, K, K) or (B*T, 3, K, K)
        eps_batch: (B, 3, 1, 1) or (B*T, 3, 1, 1) -> eps_t/2, eps_t, 2eps_t
        H_t: (B*T, 1) or None
        """
        # 1. Fused Sinkhorn (Batch 3 eps together)
        P_batch = log_domain_sinkhorn_barycenter(C_batch, eps_batch)
        
        # 2. Lift to SPD: S_t^(k) = P_t^(k) P_t^(k)^T + tau * H_t * I
        B, _, K, _ = P_batch.shape
        I = torch.eye(K, device=P_batch.device).view(1, 1, K, K)
        if H_t is not None:
            tau_adaptive = self.tau * H_t.view(B, 1, 1, 1)
        else:
            tau_adaptive = self.tau
            
        # FP32 cast for matmul stability to prevent FP16 overflow!
        P_batch_f32 = P_batch.float()
        I_f32 = I.float()
        tau_adaptive_f32 = tau_adaptive.float() if isinstance(tau_adaptive, torch.Tensor) else tau_adaptive
        
        S_batch_f32 = torch.matmul(P_batch_f32, P_batch_f32.transpose(-2, -1)) + tau_adaptive_f32 * I_f32
        
        alpha = 0.05
        tr_S = S_batch_f32.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        S_batch_f32 = (1 - alpha) * S_batch_f32 + alpha * (tr_S / K) * I_f32
        
        evals, evecs = safe_eigh(S_batch_f32)
        
        # log(S) = U log(Lambda) U^T
        log_evals = torch.log(evals.clamp(min=1e-8)).unsqueeze(-1)
        log_S_batch = torch.matmul(evecs * log_evals.transpose(-2, -1), evecs.transpose(-2, -1)).to(P_batch.dtype)
        
        # 3. Entropy-Adaptive Geodesic Weights
        # w_k = exp(-|delta_k| H_t) / sum_j exp(-|delta_j| H_t)
        # delta = [-0.02, 0.0, 0.02]
        delta_vals = torch.tensor([-0.02, 0.0, 0.02], device=P_batch.device, dtype=P_batch.dtype).view(1, 3)
        abs_delta = torch.abs(delta_vals) # (1, 3)
        
        if H_t is not None:
            # H_t is (B, 1)
            H_t = H_t.view(-1, 1)
            weights_logits = -abs_delta * H_t # (B, 3)
            w = torch.softmax(weights_logits, dim=-1) # (B, 3)
            w = w.view(B, 3, 1, 1)
        else:
            # Fallback if H_t is None
            weights_logits = -abs_delta * 1.0 # (1, 3)
            w = torch.softmax(weights_logits, dim=-1) # (1, 3)
            w = w.view(1, 3, 1, 1)
        
        # 4. Log-Euclidean Barycenter: S_bar_t = exp(sum w_k log S_t^(k))
        L_t = (w * log_S_batch).sum(dim=1) # (B, K, K)
        
        # Fused SPD barycenter diagonal exp with straight-through estimator
        evals_bar, evecs_bar = safe_eigh(L_t.detach().float())
        exp_S_bar_vals = torch.exp(evals_bar).unsqueeze(-1)
        S_bar_detach = torch.matmul(evecs_bar * exp_S_bar_vals.transpose(-2, -1), evecs_bar.transpose(-2, -1)).to(P_batch.dtype)
        
        S_bar = S_bar_detach + (L_t - L_t.detach())
        log_S_bar = L_t
        
        # Return S_bar and central_S properties
        central_evals = evals[:, 1, :]
        central_evecs = evecs[:, 1, :, :]
        central_S = S_batch_f32[:, 1, :, :].to(P_batch.dtype)
        log_S_center = log_S_batch[:, 1, :, :]
        central_P = P_batch_f32[:, 1, :, :].to(P_batch.dtype)
        
        return S_bar, central_S, central_evals, central_evecs, log_S_center, log_S_bar, central_P
