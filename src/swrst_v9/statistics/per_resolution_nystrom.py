import torch
import os

class PerResolutionNystrom:
    def __init__(self, M: int, r: int, num_classes: int, dim: int):
        self.M = M
        self.r = r
        self.num_classes = num_classes
        self.dim = dim
        
        # C_m[c] shape: [num_classes, M, dim, dim]
        self.C_m = torch.zeros((num_classes, M, dim, dim), dtype=torch.float32)
        self.n_c = torch.zeros(num_classes, dtype=torch.float32)
        
        self.U_r = None # [M, dim, r]

    def accumulate(self, L_F: torch.Tensor, y: torch.Tensor):
        """
        L_F: [B, T, M, dim]
        y: [B]
        """
        device = L_F.device
        self.C_m = self.C_m.to(device)
        self.n_c = self.n_c.to(device)
        
        B, T, M, dim = L_F.shape
        
        for c in range(self.num_classes):
            mask = (y == c)
            count = mask.sum().item()
            if count == 0:
                continue
            
            self.n_c[c] += count
            
            L_F_c = L_F[mask] # [count, T, M, dim]
            
            for m in range(M):
                # H2: Stream covariance online per resolution
                L_F_cm = L_F_c[:, :, m, :].reshape(-1, dim) # [count*T, dim]
                # C += z^T z
                self.C_m[c, m] += torch.matmul(L_F_cm.T, L_F_cm)

    def fit(self, cache_path: str = 'U_r_per_resolution.pt'):
        """
        Computes the class-balanced covariance and extracts Top-r basis for each m.
        """
        device = self.C_m.device
        
        if os.path.exists(cache_path):
            print(f"Loading cached per-resolution basis from {cache_path}")
            self.U_r = torch.load(cache_path, map_location=device)
            return

        print("Fitting Per-Resolution Nyström Basis...")
        # Class balancing
        # C^{(m)} = sum_c (1/n_c) C_{m, c}
        n_c_safe = torch.clamp(self.n_c, min=1.0).view(self.num_classes, 1, 1, 1)
        C_balanced = torch.sum(self.C_m / n_c_safe, dim=0) # [M, dim, dim]
        
        self.U_r = torch.zeros((self.M, self.dim, self.r), device=device, dtype=torch.float32)
        
        for m in range(self.M):
            eigvals, eigvecs = torch.linalg.eigh(C_balanced[m])
            # Top-r
            self.U_r[m] = eigvecs[:, -self.r:]
            
        # H1: Cache per-resolution basis
        torch.save(self.U_r, cache_path)
        print(f"Saved per-resolution basis to {cache_path}")

    def project(self, L_F: torch.Tensor):
        """
        L_F: [B, T, M, dim]
        Returns:
            z: [B, T, M, r]
        """
        if self.U_r is None:
            raise RuntimeError("Nystrom basis not fitted.")
            
        B, T, M, dim = L_F.shape
        z = torch.zeros((B, T, M, self.r), device=L_F.device, dtype=torch.float32)
        
        for m in range(M):
            L_F_m = L_F[:, :, m, :] # [B, T, dim]
            z[:, :, m, :] = torch.matmul(L_F_m, self.U_r[m]) # [B, T, r]
            
        return z
