import torch
import torch.nn as nn
import numpy as np
from rich.console import Console

console = Console()

class KernelRidgeClassifier:
    def __init__(self, alpha: float = 1e-4):
        self.alpha = alpha
        self.dual_coef_ = None
        self.X_fit_ = None
        self.classes_ = None
        
    def _is_psd(self, K: torch.Tensor) -> bool:
        try:
            torch.linalg.cholesky(K)
            return True
        except RuntimeError:
            return False
            
    def _make_psd(self, K: torch.Tensor, max_jitter: float = 1e-2) -> torch.Tensor:
        """Adds adaptive jitter to make the matrix PSD if necessary."""
        jitter = 1e-8
        while jitter <= max_jitter:
            K_jitter = K + jitter * torch.eye(K.shape[0], device=K.device)
            if self._is_psd(K_jitter):
                if jitter > 1e-8:
                    console.print(f"[yellow]Added jitter {jitter} to stabilize Cholesky.[/yellow]")
                return K_jitter
            jitter *= 10
            
        raise ValueError("Matrix is not Positive Semi-Definite even after max jitter.")
        
    def fit(self, K: torch.Tensor, y: torch.Tensor):
        """
        K: (N, N) kernel matrix
        y: (N,) labels
        """
        device = K.device
        N = K.shape[0]
        
        # Verify Symmetry
        if not torch.allclose(K, K.T, atol=1e-5):
            raise ValueError("Kernel matrix is not symmetric!")
            
        # One-hot encode targets
        self.classes_ = torch.unique(y)
        num_classes = len(self.classes_)
        Y = torch.zeros((N, num_classes), device=device)
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0
            
        # Shift Y to zero mean? Ridge usually handles it, but let's just do standard regression
        # (Y - mean could be used, but since classes are exclusive, 0/1 is fine for argmax)
        
        K_reg = K + self.alpha * torch.eye(N, device=device)
        K_reg = self._make_psd(K_reg)
        
        # Exact Cholesky
        L = torch.linalg.cholesky(K_reg)
        
        # Solve L * L^T * W = Y
        # W = (L^T)^{-1} L^{-1} Y
        temp = torch.linalg.solve_triangular(L, Y, upper=False)
        self.dual_coef_ = torch.linalg.solve_triangular(L.T, temp, upper=True)
        
    def predict(self, K_test: torch.Tensor) -> torch.Tensor:
        """
        K_test: (M, N) kernel matrix between test and train
        """
        # Y_pred = K_test @ dual_coef
        out = torch.matmul(K_test, self.dual_coef_)
        preds = torch.argmax(out, dim=1)
        
        # Map back to original class labels
        mapped_preds = torch.zeros_like(preds)
        for i, c in enumerate(self.classes_):
            mapped_preds[preds == i] = c
            
        return mapped_preds
