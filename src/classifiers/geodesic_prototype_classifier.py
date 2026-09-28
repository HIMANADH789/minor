from src.utils.safe_eigh import safe_eigh
import torch
import torch.nn as nn

class GeodesicPrototypeClassifier(nn.Module):
    def __init__(self, in_features=208, num_classes=5, gamma=0.05):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.gamma_scale = gamma
        
        # Prototypes M_c in persistent FP32 memory
        # Initialized as Identity
        self.M_c = nn.Parameter(torch.eye(in_features, dtype=torch.float32).unsqueeze(0).repeat(num_classes, 1, 1), requires_grad=False)
        
    def forward(self, phi, targets=None):
        """
        phi: (B, 208)
        targets: (B,) optional for EMA update during training
        """
        B, D = phi.shape
        device = phi.device
        
        # 1. G_x = Phi Phi^T + sigma_x I
        phi_uns = phi.unsqueeze(-1) # (B, 208, 1)
        phi_phi_T = torch.matmul(phi_uns, phi_uns.transpose(-2, -1)) # (B, 208, 208)
        
        trace = torch.diagonal(phi_phi_T, dim1=-2, dim2=-1).sum(-1) # (B,)
        sigma_x = trace / D # (B,)
        
        I = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)
        G_x = phi_phi_T + sigma_x.view(B, 1, 1) * I
        
        # Clean NaNs just in case
        G_x = torch.nan_to_num(G_x, nan=0.0)
        
        noise_G = torch.randn_like(G_x) * 1e-4
        noise_G = (noise_G + noise_G.transpose(-2, -1)) * 0.5
        G_x = G_x + noise_G
        
        # Use eigh for robust differentiable spectral gradients
        evals_x, evecs_x = safe_eigh(G_x.float())
        log_evals_x = torch.log(evals_x.clamp(min=1e-8)).unsqueeze(-1)
        log_G_x = torch.matmul(evecs_x * log_evals_x.transpose(-2, -1), evecs_x.transpose(-2, -1)).to(G_x.dtype)
        
        # 2. Log-Euclidean distance to prototypes
        # M_c is (num_classes, D, D)
        # We need log(M_c)
        M_c_clean = torch.nan_to_num(self.M_c.float(), nan=0.0)
        noise_M = torch.randn_like(M_c_clean) * 1e-4
        noise_M = (noise_M + noise_M.transpose(-2, -1)) * 0.5
        M_c_clean = M_c_clean + noise_M
        
        evals_c, evecs_c = safe_eigh(M_c_clean)
        log_evals_c = torch.log(evals_c.clamp(min=1e-8)).unsqueeze(-1)
        log_M_c = torch.matmul(evecs_c * log_evals_c.transpose(-2, -1), evecs_c.transpose(-2, -1)).to(G_x.dtype)
        
        # Broadcast for pairwise dist: G_x is (B, 1, D, D), M_c is (1, C, D, D)
        log_G_x_exp = log_G_x.unsqueeze(1)
        log_M_c_exp = log_M_c.unsqueeze(0)
        
        diff = log_G_x_exp - log_M_c_exp # (B, C, D, D)
        d_LE_sq = (torch.norm(diff, p='fro', dim=(-2, -1)) ** 2) / D # (B, C)
        
        # Logits
        z_c = -self.gamma_scale * d_LE_sq
        
        # 3. Prototype Update (if training and targets provided)
        if self.training and targets is not None:
            # We compute spectral gap of G_x for adaptive gamma
            # lambda_1 is trace + sigma_x, lambda_2 is sigma_x
            lambda_1 = trace + sigma_x
            lambda_2 = sigma_x
            delta_x = 1.0 - (lambda_2 / (lambda_1 + 1e-8)) # (B,)
            gamma_t = 0.04 + 0.08 * (1.0 - delta_x) # (B,)
            
            # Noncommutative geodesic update
            # M_c <- M_c^{1/2} (M_c^{-1/2} G_x M_c^{-1/2})^{gamma_t} M_c^{1/2}
            # We do this for each class c
            with torch.no_grad():
                for c in range(self.num_classes):
                    mask = (targets == c)
                    if mask.sum() > 0:
                        # Average G_x and gamma_t for this class in this batch
                        G_x_c = G_x[mask].mean(dim=0)
                        gamma_c = gamma_t[mask].mean().item()
                        
                        # Stabilize G_x_c (SPD trace normalization)
                        tr_G = torch.trace(G_x_c)
                        tau = 0.02
                        G_x_c = G_x_c / (tr_G + tau) + tau * torch.eye(D, device=device)
                        
                        M_curr = torch.nan_to_num(self.M_c[c].float(), nan=0.0)
                        noise_curr = torch.randn_like(M_curr) * 1e-4
                        noise_curr = (noise_curr + noise_curr.transpose(-2, -1)) * 0.5
                        M_curr = M_curr + noise_curr
                        
                        # M_c^{1/2} and M_c^{-1/2}
                        evals_c, evecs_c = safe_eigh(M_curr)
                        evals_c = evals_c.clamp(min=1e-8)
                        
                        M_half = torch.matmul(evecs_c * torch.sqrt(evals_c).unsqueeze(-2), evecs_c.transpose(-2, -1))
                        M_inv_half = torch.matmul(evecs_c * (1.0 / torch.sqrt(evals_c)).unsqueeze(-2), evecs_c.transpose(-2, -1))
                        
                        # Inner = M_c^{-1/2} G_x M_c^{-1/2}
                        inner = torch.matmul(M_inv_half, torch.matmul(G_x_c.float(), M_inv_half))
                        inner = torch.nan_to_num(inner, nan=0.0)
                        noise_inner = torch.randn_like(inner) * 1e-4
                        noise_inner = (noise_inner + noise_inner.transpose(-2, -1)) * 0.5
                        inner = inner + noise_inner
                        
                        # Inner^{gamma_c}
                        evals_in, evecs_in = safe_eigh(inner)
                        evals_in = evals_in.clamp(min=1e-8)
                        inner_pow = torch.matmul(evecs_in * torch.pow(evals_in, gamma_c).unsqueeze(-2), evecs_in.transpose(-2, -1))
                        
                        # New M_c
                        M_new = torch.matmul(M_half, torch.matmul(inner_pow, M_half))
                        
                        # Apply spectral floor tau to prevent collapse
                        M_new = M_new + 0.015 * torch.eye(D, device=device)
                        
                        self.M_c[c] = M_new.to(self.M_c.dtype)
                        
        return z_c
