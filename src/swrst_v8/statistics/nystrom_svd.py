import torch

class NystromSVD:
    def __init__(self, r: int = 8, num_classes: int = 5):
        self.r = r
        self.num_classes = num_classes
        self.C_class = None
        self.n_class = None
        self.U_r = None
        
    def accumulate(self, L_F: torch.Tensor, y: torch.Tensor):
        """
        L_F: [B, T-1, M, D]
        y: [B]
        """
        B, T, M, D = L_F.shape
        if self.C_class is None:
            self.C_class = torch.zeros((self.num_classes, D, D), device=L_F.device, dtype=torch.float32)
            self.n_class = torch.zeros(self.num_classes, device=L_F.device, dtype=torch.float32)
            
        for c in range(self.num_classes):
            mask = (y == c)
            if mask.sum() > 0:
                # H1: Flatten BTM for SVD transforms
                L_c = L_F[mask].reshape(-1, D)
                
                # H3: Online covariance accumulation (streaming)
                self.C_class[c] += torch.matmul(L_c.T, L_c)
                self.n_class[c] += mask.sum()
                
    def fit(self):
        """
        Fits the global class-balanced basis.
        """
        if self.C_class is None:
            raise RuntimeError("No data accumulated.")
            
        D = self.C_class.size(1)
        C_global = torch.zeros((D, D), device=self.C_class.device, dtype=torch.float32)
        
        # U3: Class-balanced covariance
        for c in range(self.num_classes):
            if self.n_class[c] > 0:
                C_global += self.C_class[c] / self.n_class[c]
                
        # H4: Blocked eig decomposition (only once)
        eigvals, eigvecs = torch.linalg.eigh(C_global)
        
        # Extract top r
        self.U_r = eigvecs[:, -self.r:] # [D, r]
        
    def project(self, L_F: torch.Tensor):
        """
        L_F: [B, T-1, M, D]
        Returns:
            z_t(m): [B, T-1, M, r]
        """
        if self.U_r is None:
            raise RuntimeError("SVD basis not fitted.")
            
        return torch.matmul(L_F, self.U_r) # [B, T-1, M, r]
