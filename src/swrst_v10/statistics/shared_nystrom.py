import torch
import os

class SharedNystrom:
    def __init__(self, r: int, num_classes: int, dim: int, epsilons: torch.Tensor, gamma: float = 0.25):
        self.r = r
        self.num_classes = num_classes
        self.dim = dim
        
        # C_c shape: [num_classes, dim, dim]
        self.C_c = torch.zeros((num_classes, dim, dim), dtype=torch.float32)
        self.n_c = torch.zeros(num_classes, dtype=torch.float32)
        
        self.U_r = None # [dim, r]
        
        # C1: Local adapters with positive power scaling
        # P_m = diag((eps_m / eps_mid)^gamma)
        mid_idx = len(epsilons) // 2
        eps_mid = epsilons[mid_idx]
        
        scale_factors = (epsilons / eps_mid) ** gamma
        # P_m diagonal vectors for broadcasting: [M, r] or just scale after projection
        # P_m is a scalar multiplier per resolution since it applies to all r dims
        # Wait, if P_m is diag(...) for r dims, it just scales the whole vector by that scalar.
        self.P_scale = scale_factors # [M]

    def accumulate(self, L_F: torch.Tensor, y: torch.Tensor):
        """
        L_F: [B, T, M, dim]
        y: [B]
        """
        device = L_F.device
        self.C_c = self.C_c.to(device)
        self.n_c = self.n_c.to(device)
        self.P_scale = self.P_scale.to(device)
        
        B, T, M, dim = L_F.shape
        
        for c in range(self.num_classes):
            mask = (y == c)
            count = mask.sum().item()
            if count == 0:
                continue
            
            self.n_c[c] += count
            
            # H1: Shared covariance streaming
            L_F_c = L_F[mask].reshape(-1, dim) # [count * T * M, dim]
            self.C_c[c] += torch.matmul(L_F_c.T, L_F_c)

    def fit(self, cache_path: str = 'U_r_shared.pt'):
        """
        Computes the class-balanced global covariance and extracts Top-r basis.
        """
        device = self.C_c.device
        
        if os.path.exists(cache_path):
            print(f"Loading cached shared basis from {cache_path}")
            self.U_r = torch.load(cache_path, map_location=device, weights_only=True)
            return

        print("Fitting Shared Hierarchical Spectral Basis...")
        n_c_safe = torch.clamp(self.n_c, min=1.0).view(self.num_classes, 1, 1)
        C_balanced = torch.sum(self.C_c / n_c_safe, dim=0) # [dim, dim]
        
        eigvals, eigvecs = torch.linalg.eigh(C_balanced)
        self.U_r = eigvecs[:, -self.r:] # [dim, r]
            
        torch.save(self.U_r, cache_path)
        print(f"Saved shared basis to {cache_path}")

    def project(self, L_F: torch.Tensor):
        """
        L_F: [B, T, M, dim]
        Returns:
            z: [B, T, M, r]
        """
        if self.U_r is None:
            raise RuntimeError("Nystrom basis not fitted.")
            
        # z_raw = L_F @ U_r
        # [B, T, M, dim] @ [dim, r] -> [B, T, M, r]
        z_raw = torch.matmul(L_F, self.U_r)
        
        # Apply local adapter P_m
        # self.P_scale: [M] -> [1, 1, M, 1]
        z = z_raw * self.P_scale.view(1, 1, -1, 1)
            
        return z
