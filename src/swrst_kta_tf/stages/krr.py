import torch

def class_conditional_lambda(y_train: torch.Tensor, device) -> torch.Tensor:
    """Per-sample regularisation: lighter for minority classes."""
    n_classes = int(y_train.max().item()) + 1
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    n_max = n_per_class.max()
    lambda_base = 1e-4
    beta = 0.7
    lambda_c = lambda_base * (n_per_class / n_max).pow(beta)
    return lambda_c[y_train]  # (N_train,)

def krr_solve_margin_aware(
    K: torch.Tensor,
    Y_onehot: torch.Tensor,    # (N, C)
    y_train: torch.Tensor,     # (N,) integer labels
    n_classes: int = 5,
    lambda_base: float = 1e-4,
    beta: float = 0.7,
    jitter: float = 1e-6
) -> torch.Tensor:
    """
    Two-pass KRR:
    Pass 1: standard class-conditional solve
    Measure per-class training accuracy
    Pass 2: halve lambda for classes with training accuracy < 0.80
    """
    
    # Compute class counts and initial lambdas
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    n_max = n_per_class.max()
    lambda_c = lambda_base * (n_per_class / n_max).pow(beta)
    Lambda_diag = lambda_c[y_train]  # (N,)
    
    # Pass 1: initial solve
    N = K.shape[0]
    K_reg = K + torch.diag(Lambda_diag + jitter)
    K_reg = K_reg.contiguous()
    try:
        L = torch.linalg.cholesky(K_reg)
        alpha = torch.cholesky_solve(Y_onehot, L)
    except torch.linalg.LinAlgError:
        K_reg = K_reg + torch.eye(N, device=K.device) * 1e-3
        L = torch.linalg.cholesky(K_reg)
        alpha = torch.cholesky_solve(Y_onehot, L)
    
    # Measure training accuracy per class
    train_scores = K @ alpha               # (N, C)
    train_pred = train_scores.argmax(-1)   # (N,)
    
    # Per-class training accuracy
    for c in range(n_classes):
        mask = (y_train == c)
        if mask.sum() == 0:
            continue
        train_acc_c = (train_pred[mask] == c).float().mean().item()
        
        # If training accuracy < 80%, halve lambda for this class
        # This means: trust the training labels more for this class
        if train_acc_c < 0.80:
            lambda_c[c] = lambda_c[c] * 0.5
            print(f"  Class {c}: train acc={train_acc_c:.3f} "
                  f"-> reducing lambda to {lambda_c[c].item():.2e}")
    
    # Pass 2: refined solve with updated lambdas
    Lambda_diag_v2 = lambda_c[y_train]
    K_reg_v2 = K + torch.diag(Lambda_diag_v2 + jitter)
    K_reg_v2 = K_reg_v2.contiguous()
    try:
        L2 = torch.linalg.cholesky(K_reg_v2)
        alpha = torch.cholesky_solve(Y_onehot, L2)
    except torch.linalg.LinAlgError:
        K_reg_v2 = K_reg_v2 + torch.eye(N, device=K.device) * 1e-3
        L2 = torch.linalg.cholesky(K_reg_v2)
        alpha = torch.cholesky_solve(Y_onehot, L2)
    
    return alpha

import math

def loocv_loss_balanced(K_fused: torch.Tensor, Y_onehot: torch.Tensor, Lambda_diag: torch.Tensor, y_train: torch.Tensor, n_classes: int, jitter: float = 1e-6) -> torch.Tensor:
    N = K_fused.shape[0]
    K_reg = K_fused + torch.diag(Lambda_diag + jitter)
    L = torch.linalg.cholesky(K_reg)
    alpha = torch.cholesky_solve(Y_onehot, L)
    K_inv = torch.cholesky_inverse(L)
    
    loo_resid = alpha / K_inv.diag().unsqueeze(1)
    per_sample_sq_err = (loo_resid ** 2).sum(dim=1)

    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    w = 1.0 / n_per_class[y_train]
    w = w / w.sum() * N

    return (w * per_sample_sq_err).sum()

def learn_fusion_and_lambda_joint(K_list: list[torch.Tensor], Y_onehot: torch.Tensor, y_train: torch.Tensor, n_classes: int, n_steps: int = 200, lr: float = 0.05):
    logs = torch.stack([torch.log(K.clamp(min=1e-30)) for K in K_list])
    theta = torch.zeros(len(K_list), requires_grad=True, device=K_list[0].device)
    log_lambda0 = torch.tensor(math.log(1e-4), requires_grad=True, device=K_list[0].device)
    beta_raw = torch.tensor(0.7, requires_grad=True, device=K_list[0].device)
    log_tau = torch.tensor(math.log(5.0), requires_grad=True, device=K_list[0].device)

    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    n_max = n_per_class.max()
    opt = torch.optim.Adam([theta, log_lambda0, beta_raw, log_tau], lr=lr)

    for step in range(n_steps):
        opt.zero_grad()
        beta = torch.nn.functional.softplus(beta_raw)
        tau = torch.exp(log_tau)
        alpha_w = torch.softmax(theta, dim=0)
        log_K = (alpha_w.view(-1, 1, 1) * logs).sum(0)
        K_fused = torch.exp(torch.clamp(log_K, min=-30.0))

        lambda_c = torch.exp(log_lambda0) * (n_per_class / n_max).pow(beta)
        Lambda_diag = lambda_c[y_train]

        loss = loocv_loss_balanced(K_fused, Y_onehot, Lambda_diag, y_train, n_classes)
        loss.backward()
        opt.step()

    return (torch.softmax(theta.detach(), dim=0),
            torch.exp(log_lambda0.detach()).item(),
            torch.nn.functional.softplus(beta_raw).detach().item(),
            torch.exp(log_tau).detach().item())

def predict(K_test_train: torch.Tensor, alpha: torch.Tensor,
            y_train: torch.Tensor = None) -> torch.Tensor:
    """
    Predict with optional class-prior bias adjustment.
    """
    scores = K_test_train @ alpha  # (N_test, C)

    if y_train is not None:
        C = scores.shape[1]
        n_per_class = torch.bincount(y_train, minlength=C).float().clamp(min=1)
        N = float(y_train.shape[0])
        bias = torch.log(N / n_per_class)           # (C,)
        score_range = scores.max() - scores.min()
        bias = bias * (score_range * 0.1) / bias.max()
        scores = scores + bias.unsqueeze(0)

    return scores.argmax(dim=-1)
