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
        
        # C4: Minority-heavy inverse regularization scaling
        lambda_c = self.lambda_base * torch.sqrt(n_c / (n_max + 1e-8)) # [C]
        
        # Build Lambda matrix (per-sample)
        # Lambda_ii = lambda_{y_i}
        lambda_ii = lambda_c[y_labels] # [N]
        
        # Enforce exact symmetry for numerical stability
        K_train = 0.5 * (K_train + K_train.T)
        
        self.alpha = torch.zeros((N, C), device=K_train.device, dtype=torch.float32)
        
        for c in range(C):
            success = False
            # H10: Cholesky retry ladder
            jitters = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            
            for jitter in jitters:
                K_reg = K_train.clone()
                diag_idx = torch.arange(N, device=K_train.device)
                
                # K + Lambda + Jitter*I
                K_reg[diag_idx, diag_idx] += lambda_ii + jitter
                
                try:
                    L = torch.linalg.cholesky(K_reg)
                    self.alpha[:, c:c+1] = torch.cholesky_solve(y_train[:, c:c+1], L)
                    success = True
                    # print(f"Class {c} Cholesky succeeded with jitter {jitter}")
                    break
                except RuntimeError:
                    continue
                    
            if not success:
                raise RuntimeError("Cholesky failed even with ladder. Matrix is severely non-PSD.")

    def predict(self, K_test: torch.Tensor):
        """
        K_test: [N_test, N_train]
        """
        if self.alpha is None:
            raise RuntimeError("Model is not fitted yet.")
        
        preds = torch.matmul(K_test, self.alpha)
        return preds
