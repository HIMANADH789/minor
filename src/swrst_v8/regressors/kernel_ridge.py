import torch
import torch.nn as nn

class ClassConditionalKRR(nn.Module):
    def __init__(self, lambda_c: float = 1e-4, jitter: float = 1e-6):
        super().__init__()
        self.lambda_c = lambda_c
        self.jitter = jitter
        self.alpha = None

    def fit(self, K_train: torch.Tensor, y_train: torch.Tensor):
        """
        K_train: [N, N]
        y_train: [N, C] (One-hot encoded)
        """
        N, C = y_train.shape
        
        # Class counts
        y_labels = torch.argmax(y_train, dim=1)
        n_c = torch.bincount(y_labels, minlength=C).float()
        n_max = torch.max(n_c)
        
        # Class-conditional lambda
        # lambda_c = 1e-4 * (n_max / n_c)^0.5
        lambda_val = self.lambda_c * torch.sqrt(n_max / (n_c + 1e-8)) # [C]
        
        # We need a different lambda for each class regression
        # Since standard KRR does K_reg = K + Lambda, if Lambda is different for each output dimension,
        # we have to solve them separately, or just use the per-sample lambda weight
        # Actually, it's simpler to do it per class
        
        # Enforce exact symmetry
        K_train = 0.5 * (K_train + K_train.T)
        
        self.alpha = torch.zeros((N, C), device=K_train.device, dtype=torch.float32)
        
        for c in range(C):
            success = False
            current_jitter = self.jitter
            
            for _ in range(15):
                K_reg = K_train.clone()
                diag_idx = torch.arange(N, device=K_train.device)
                K_reg[diag_idx, diag_idx] += lambda_val[c] + current_jitter
                
                # H10: Cholesky only (dynamic jitter ensures PSD stability without eigh)
                try:
                    L = torch.linalg.cholesky(K_reg)
                    self.alpha[:, c:c+1] = torch.cholesky_solve(y_train[:, c:c+1], L)
                    success = True
                    break
                except RuntimeError:
                    current_jitter *= 5.0
                    
            if not success:
                raise RuntimeError("Cholesky failed even with massive jitter. Matrix is severely non-PSD.")

    def predict(self, K_test: torch.Tensor):
        """
        K_test: [N_test, N_train]
        """
        if self.alpha is None:
            raise RuntimeError("Model is not fitted yet.")
        
        preds = torch.matmul(K_test, self.alpha)
        return preds
