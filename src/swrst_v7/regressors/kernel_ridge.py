import torch
import torch.nn as nn

class ClassConditionalKRR(nn.Module):
    def __init__(self, lambda_c: float = 1.0, lambda_min: float = 1e-6):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_min = lambda_min
        self.alpha = None

    def fit(self, K_train: torch.Tensor, y_train: torch.Tensor, KL_train: torch.Tensor):
        """
        K_train: [N, N]
        y_train: [N, C] (One-hot encoded)
        KL_train: [N] (Pooled temporal KL divergence)
        """
        N = K_train.size(0)
        
        # Upgrade U3: Adaptive lambda floor
        # lambda_x = max(lambda_min, lambda_c * exp(-KL_x))
        lambda_x = self.lambda_c * torch.exp(-KL_train)
        lambda_x = torch.clamp(lambda_x, min=self.lambda_min)
        
        Lambda = torch.diag(lambda_x).to(K_train.device)
        
        # Regularized kernel matrix
        K_reg = K_train + Lambda
        
        # H10: Cholesky KRR (torch.linalg.cholesky and cholesky_solve)
        try:
            L = torch.linalg.cholesky(K_reg)
        except RuntimeError:
            K_reg = K_reg + torch.eye(N, device=K_train.device) * self.lambda_min
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
