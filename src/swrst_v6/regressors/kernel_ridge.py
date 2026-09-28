import torch
import torch.nn as nn

class ClassConditionalKRR(nn.Module):
    def __init__(self, lambda_c: float = 1.0):
        super().__init__()
        self.lambda_c = lambda_c
        self.alpha = None

    def fit(self, K_train: torch.Tensor, y_train: torch.Tensor, var_H_train: torch.Tensor):
        """
        K_train: [N, N]
        y_train: [N, C] (One-hot encoded)
        var_H_train: [N] (Sample-specific entropy variance)
        """
        N = K_train.size(0)
        
        # Diagonal Lambda matrix
        lambda_x = self.lambda_c * (1.0 + var_H_train)
        Lambda = torch.diag(lambda_x).to(K_train.device)
        
        # Regularized kernel matrix
        K_reg = K_train + Lambda
        
        # H10: Cholesky KRR (torch.linalg.cholesky and cholesky_solve)
        # Add small jitter for numerical stability if needed
        try:
            L = torch.linalg.cholesky(K_reg)
        except RuntimeError:
            K_reg = K_reg + torch.eye(N, device=K_train.device) * 1e-4
            L = torch.linalg.cholesky(K_reg)
            
        self.alpha = torch.cholesky_solve(y_train, L)

    def predict(self, K_test: torch.Tensor):
        """
        K_test: [N_test, N_train]
        """
        if self.alpha is None:
            raise RuntimeError("Model is not fitted yet.")
        
        preds = torch.matmul(K_test, self.alpha)
        return preds
