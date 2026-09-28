from src.utils.safe_eigh import safe_eigh
import torch
import torch.nn as nn

class SafeEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        L, V = safe_eigh(A)
        ctx.save_for_backward(L, V)
        return L, V

    @staticmethod
    def backward(ctx, grad_L, grad_V):
        L, V = ctx.saved_tensors
        L_i = L.unsqueeze(-2)
        L_j = L.unsqueeze(-1)
        diff = L_j - L_i
        
        mask = torch.abs(diff) > 1e-5
        F = torch.zeros_like(diff)
        F[mask] = 1.0 / diff[mask]
        
        Vt_dV = torch.matmul(V.transpose(-2, -1), grad_V)
        Vt_dV_sym = 0.5 * (Vt_dV - Vt_dV.transpose(-2, -1))
        
        term2 = F * Vt_dV_sym
        term1 = torch.diag_embed(grad_L)
        
        dA = torch.matmul(V, torch.matmul(term1 + term2, V.transpose(-2, -1)))
        return dA

class AdaptivePTCO(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
        
    def forward(self, S_bar_x, R_hat_seq, central_evals_seq, K_dim, evals_x, evecs_x, delta_seq=None):
        """
        S_bar_x: (B, K_dim, K_dim)
        R_hat_seq: (B, T, K_dim, K_dim)
        central_evals_seq: (B, T, K_dim)
        evals_x: (B, K_dim)
        evecs_x: (B, K_dim, K_dim)
        delta_seq: (B, T)
        """
        B, T, _ = central_evals_seq.shape
        
        # 1. Entropy H_t from central evals (normalized)
        # Normalize evals to be a probability distribution
        eval_probs = central_evals_seq / (central_evals_seq.sum(dim=-1, keepdim=True) + 1e-8)
        H_t = -torch.sum(eval_probs * torch.log(eval_probs.clamp(min=1e-8)), dim=-1) # (B, T)
        H_t = torch.nan_to_num(H_t, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Adaptive power p_t
        log_K = torch.log(torch.tensor(K_dim, dtype=torch.float32, device=H_t.device))
        if delta_seq is not None:
            # Patch D: Entropy-damped power with spectral gap (restored log_K norm)
            p_t = 1.0 + torch.floor(4.0 * (H_t / log_K) * (1.0 - delta_seq))
        else:
            log_K = torch.log(torch.tensor(K_dim, dtype=torch.float32, device=H_t.device))
            p_t = 1.0 + torch.floor(4.0 * H_t / log_K)
            
        p_t = p_t.clamp(min=1, max=5).long() # (B, T)
        
        # 3. Create scale-invariant trace-normalized signal barycenter evals
        trace_Sx = torch.diagonal(S_bar_x, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        evals_x_scaled = evals_x / (trace_Sx.squeeze(-1) + 1e-8)
        
        # 4. Precompute powers of S_tilde_x
        # We need S_tilde_x^{p/2} and S_tilde_x^{-p/2}
        S_pow = {}
        S_inv_pow = {}
        # Smooth floor for spectral clipping before log (PATCH H)
        # Lambda_clip = tau + (Lambda - tau) * sigmoid((Lambda - tau) / tau)
        diff = evals_x_scaled - self.tau
        evals_x_clip = self.tau + diff * torch.sigmoid(diff / self.tau)
        log_evals_x = torch.log(evals_x_clip)
        
        for p in range(1, 6):
            p_float = float(p) / 2.0
            # S^{p/2}
            evals_pow = torch.pow(evals_x_scaled.clamp(min=1e-8), p_float).unsqueeze(-1)
            S_pow[p] = torch.matmul(evecs_x * evals_pow.transpose(-2, -1), evecs_x.transpose(-2, -1))
            
            # S^{-p/2} using Soft Inverse: exp(-p * log(mu + tau))
            evals_inv_pow = torch.exp(-p_float * log_evals_x).unsqueeze(-1)
            S_inv_pow[p] = torch.matmul(evecs_x * evals_inv_pow.transpose(-2, -1), evecs_x.transpose(-2, -1))
            
        # 5. Construct exact commutator M_t
        # S_pow and S_inv_pow have keys 1 to 5
        S_pow_tensor = torch.stack([S_pow[i] for i in range(1, 6)], dim=0) # (5, B, K, K)
        S_inv_pow_tensor = torch.stack([S_inv_pow[i] for i in range(1, 6)], dim=0) # (5, B, K, K)
        
        # Select correctly based on p_t
        p_idx = p_t - 1 # (B, T) indices 0 to 4
        B_idx = torch.arange(B, device=p_t.device).unsqueeze(1) # (B, 1)
        
        S_p_seq = S_pow_tensor[p_idx, B_idx] # (B, T, K, K)
        S_inv_p_seq = S_inv_pow_tensor[p_idx, B_idx] # (B, T, K, K)
        
        # 6. M_t = S^{-p_t/2} R_t S^{-p_t/2}
        M_seq = torch.matmul(S_inv_p_seq, torch.matmul(R_hat_seq.float(), S_inv_p_seq))
        
        M_seq = torch.nan_to_num(M_seq, nan=0.0)
        K = M_seq.size(-1)
        diag_noise = torch.diag(torch.linspace(1e-4, 1e-3, K, device=M_seq.device, dtype=M_seq.dtype))
        M_seq = M_seq + diag_noise.view(1, 1, K, K)
        # Enforce symmetry to prevent NaN gradients in eigh
        M_seq = (M_seq + M_seq.transpose(-2, -1)) * 0.5
        
        # Batched eigh over the entire sequence!
        evals_seq, evecs_seq = safe_eigh(M_seq)
        # Prevent float32 overflow when raising to power p_t (max p_t=5 -> (100)^5 = 1e10, E_t ~ 1e11, phi^2 ~ 1e22 < 3.4e38)
        evals_seq = evals_seq.clamp(min=1e-8, max=100.0)
        
        # M_t^{p_t}
        p_float_seq = p_t.float().unsqueeze(-1) # (B, T, 1)
        M_seq_pow = torch.matmul(evecs_seq * torch.pow(evals_seq, p_float_seq).unsqueeze(-2), evecs_seq.transpose(-2, -1))
        
        # 7. Final transport metric: M_t_seq = S^{p_t/2} M_t^{p_t} S^{p_t/2}
        M_t_seq = torch.matmul(S_p_seq, torch.matmul(M_seq_pow, S_p_seq))
        
        return M_t_seq, H_t, p_t
