from src.utils.safe_eigh import safe_eigh
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductKernel(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable simplex weights for the 4 kernels
        self.theta = nn.Parameter(torch.zeros(4))
        
    def forward(self, res_feats, G_x, M_t_seq, qtcf_feats):
        """
        Compute batch-wise kernels.
        res_feats: (B, 80)
        G_x: (B, K_dim, K_dim) - from prototype classifier
        M_t_seq: (B, T, K_dim, K_dim)
        qtcf_feats: (B, 5)
        
        Returns:
        K_combined: (B, B) kernel matrix
        """
        B = res_feats.size(0)
        
        # 1. Resolution Kernel (RBF)
        dist_res = torch.cdist(res_feats, res_feats, p=2) ** 2
        gamma_res = 1.0 / res_feats.size(1)
        K_delta = torch.exp(-gamma_res * dist_res)
        
        # 2. SPD Kernel (Log-Euclidean)
        # G_x is SPD. We must add random symmetric noise to break the exact
        # degeneracy (rank-1 + Identity has 207 repeated singular values)
        # which causes NaN gradients in SVD/Eigh backward pass (division by s_i^2 - s_j^2).
        G_x = torch.nan_to_num(G_x, nan=0.0)
        K_dim = G_x.size(-1)
        diag_noise = torch.diag(torch.linspace(1e-4, 1e-3, K_dim, device=G_x.device, dtype=G_x.dtype))
        G_x = G_x + diag_noise.unsqueeze(0)
        
        evals_x, evecs_x = safe_eigh(G_x.float())
        log_evals = torch.log(evals_x.clamp(min=1e-8)).unsqueeze(-1)
        log_G_x = torch.matmul(evecs_x * log_evals.transpose(-2, -1), evecs_x.transpose(-2, -1))
        
        log_G_x_flat = log_G_x.view(B, -1)
        dist_S = torch.cdist(log_G_x_flat, log_G_x_flat, p=2) ** 2
        gamma_S = 1.0 / log_G_x_flat.size(1)
        K_S = torch.exp(-gamma_S * dist_S)
        
        # 3. Commutator Kernel (Frobenius)
        M_flat = M_t_seq.view(B, -1)
        dist_M = torch.cdist(M_flat, M_flat, p=2) ** 2
        gamma_M = 1.0 / M_flat.size(1)
        K_M = torch.exp(-gamma_M * dist_M)
        
        # 4. QTCF Kernel (RBF)
        dist_Q = torch.cdist(qtcf_feats, qtcf_feats, p=2) ** 2
        gamma_Q = 1.0 / qtcf_feats.size(1)
        K_Q = torch.exp(-gamma_Q * dist_Q)
        
        # Learnable simplex weights
        w = F.softmax(self.theta, dim=0)
        
        # Combined kernel
        K_combined = w[0] * K_delta + w[1] * K_S + w[2] * K_M + w[3] * K_Q
        
        return K_combined
