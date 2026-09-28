import torch
import torch.nn as nn

class ClassConditionalKRR(nn.Module):
    def __init__(self, lambda_base: float = 1e-4):
        super().__init__()
        self.lambda_base = lambda_base
        self.alpha = None

    def fit(self, K_train: torch.Tensor, y_train: torch.Tensor):
        """
        K_train: [N, N]
        y_train: [N, C] (One-hot encoded)
        """
        N, C = y_train.shape
        
        y_labels = torch.argmax(y_train, dim=1)
        n_c = torch.bincount(y_labels, minlength=C).float()
        n_max = torch.max(n_c)
        
        # Exact V3 Class balancing
        lambda_c = self.lambda_base * torch.sqrt(n_max / (n_c + 1e-8)) # [C]
        
        # Base per-sample
        lambda_x = lambda_c[y_labels] # [N]
        
        # Enforce exact symmetry
        K_train = 0.5 * (K_train + K_train.T)
        
        K_reg = K_train.clone()
        K_reg.diagonal().add_(lambda_x)
        K_reg.diagonal().add_(1e-6) # Base jitter
        
        self.alpha = torch.zeros((N, C), device=K_train.device, dtype=torch.float32)
        
        for c in range(C):
            success = False
            jitters = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, 100.0]
            
            for jitter in jitters:
                K_c = K_reg.clone()
                if jitter > 0.0:
                    K_c.diagonal().add_(jitter)
                
                try:
                    L = torch.linalg.cholesky(K_c)
                    self.alpha[:, c:c+1] = torch.cholesky_solve(y_train[:, c:c+1], L)
                    success = True
                    break
                except RuntimeError:
                    continue
                    
            if not success:
                raise RuntimeError("Cholesky failed even with ladder.")

    def predict(self, K_test: torch.Tensor):
        """
        K_test: [N_test, N_train]
        """
        if self.alpha is None:
            raise RuntimeError("Model is not fitted yet.")
        
        preds = torch.matmul(K_test, self.alpha)
        return preds
