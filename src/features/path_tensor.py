import torch
import torch.nn as nn

class PathTensor(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.tau = tau
        
    def forward(self, eps_seq, delta_seq, d_delta_seq, d2_delta_seq, H_seq,
                algebra_feats, Pi_seq, E_seq, p_seq, cp_feats, qtcf_feats,
                evals_x, K_seq):
        """
        eps_seq, delta_seq, d_delta_seq, d2_delta_seq, H_seq: (B, T) - 16 * 5 = 80D
        algebra_feats: (B, 7) - 7D
        Pi_seq: (B, T) - 16D
        E_seq: (B, T) - 16D
        p_seq: (B, T) - 16D
        cp_feats: (B, 48) - 48D
        qtcf_feats: (B, 5) - 5D
        evals_x: (B, K_dim) - 16D
        K_seq: (B, T) - used for surprise stats 4D
        """
        B = eps_seq.size(0)
        
        # 1. Resolution 80D
        res_feats = torch.cat([
            eps_seq.view(B, -1),
            delta_seq.view(B, -1),
            d_delta_seq.view(B, -1),
            d2_delta_seq.view(B, -1),
            H_seq.view(B, -1)
        ], dim=-1) # (B, 80)
        
        # 2. Surprise Stats 4D
        K_seq = K_seq.view(B, -1)
        surprise_mean = K_seq.mean(dim=-1, keepdim=True)
        surprise_std = K_seq.std(dim=-1, keepdim=True)
        surprise_max = K_seq.max(dim=-1, keepdim=True)[0]
        surprise_min = K_seq.min(dim=-1, keepdim=True)[0]
        surprise_feats = torch.cat([surprise_mean, surprise_std, surprise_max, surprise_min], dim=-1) # (B, 4)
        
        # 3. Assemble all (enforce exactly 208D)
        evals_x_16 = evals_x[:, :16].view(B, -1) # Keep top-16 dominant evals
        
        components = [
            res_feats,              # 80
            algebra_feats,          # 7
            Pi_seq.view(B, -1),     # 16
            E_seq.view(B, -1),      # 16
            p_seq.float().view(B, -1), # 16
            cp_feats[:, :48],       # 48 (sliced from 60 to enforce 208D exactly)
            qtcf_feats,             # 5
            evals_x_16,             # 16
            surprise_feats          # 4
        ]
        
        phi = torch.cat(components, dim=-1) # Total: 80+7+16+16+16+48+5+16+4 = 208D
        
        # 4. Normalize to unit hypersphere
        phi_sq_norm = torch.sum(phi ** 2, dim=-1, keepdim=True)
        phi = phi / torch.sqrt(phi_sq_norm + self.tau)
        
        return phi
