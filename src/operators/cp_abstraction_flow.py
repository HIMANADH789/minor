import torch
import torch.nn as nn

class CPAbstractionFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
    def forward(self, C_t_seq, central_evals_seq, evecs_seq, H_t_seq):
        """
        C_t_seq: Cost matrices (B, T, K, K)
        central_evals_seq: Eigenvalues nu_k (B, T, K)
        evecs_seq: Eigenvectors (B, T, K, K)
        H_t_seq: Entropy (B, T)
        """
        B, T, K_dim = central_evals_seq.shape
        device = central_evals_seq.device
        
        # 1. QTCF Tensor
        # Xi_x(s) = sum_{k,l} nu_k^{1-s} nu_l^s |C_{kl}|^2
        # C_{kl} in the eigenbasis: evecs^T C evecs
        # Let's compute C_tilde = |evecs^T C evecs|^2
        # Actually C is already (B, T, K, K), we project it
        # Compute qtcf_feats entirely in fp32 to prevent fp16 overflow!
        C_t_seq_f32 = C_t_seq.float()
        evecs_seq_f32 = evecs_seq.float()
        nu = central_evals_seq.clamp(min=1e-8) # (B, T, K)
        nu_f32 = nu.float()
        
        C_tilde = torch.matmul(evecs_seq_f32.transpose(-2, -1), torch.matmul(C_t_seq_f32, evecs_seq_f32))
        C_tilde_sq = torch.abs(C_tilde) ** 2 # (B, T, K, K)
        
        qtcf_features = []
        for s in self.s_values:
            nu_1_s = torch.pow(nu_f32, 1.0 - s).unsqueeze(-1) # (B, T, K, 1)
            nu_s = torch.pow(nu_f32, s).unsqueeze(-2) # (B, T, 1, K)
            
            # (nu_k^{1-s} nu_l^s) * C_tilde_sq
            term = nu_1_s * nu_s * C_tilde_sq # (B, T, K, K)
            xi_s = term.sum(dim=(-2, -1)) # (B, T)
            # Pool across time to get global 5D feature
            qtcf_features.append(xi_s.mean(dim=1)) # (B,)
            
        qtcf_out = torch.stack(qtcf_features, dim=1).to(C_t_seq.dtype) # (B, 5)
        
        # 2. Spectral Flow
        log_K = torch.log(torch.tensor(K_dim, dtype=torch.float32, device=device))
        alpha_t = log_K / (H_t_seq + 1e-8) # (B, T)
        
        # We need 48D CP output. K_dim = 16. So 3 * 16 = 48.
        # We compute for alpha in {0, 0.5 * alpha_t, alpha_t}
        # nu_1 is the largest eigenvalue (eigh returns ascending, so it's the last one)
        nu_1 = nu[:, :, -1:]
        
        # We need a distribution pi_k. Usually pi_k is just the normalized eigenvalues or the basis projection.
        # Let's use pi_k = normalized nu_k
        pi_k = nu / nu.sum(dim=-1, keepdim=True)
        
        ratio = (nu / nu_1).clamp(min=1e-8) # (B, T, K)
        
        cp_features = []
        for multiplier in [0.0, 0.5, 1.0]:
            a_t = (alpha_t * multiplier).unsqueeze(-1) # (B, T, 1)
            phi_alpha = torch.pow(ratio, a_t) * pi_k # (B, T, K)
            cp_features.append(phi_alpha.mean(dim=1)) # (B, K)
            
        cp_out = torch.cat(cp_features, dim=-1) # (B, 48)
        
        return qtcf_out, cp_out
