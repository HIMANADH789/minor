import torch

def randomized_svd_right(M: torch.Tensor, rank: int, n_oversampling: int = 10) -> torch.Tensor:
    # M: (m, n). Returns V: (n, rank)
    m, n = M.shape
    k = rank + n_oversampling
    
    Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
    Y = M @ Omega  # (m, k)
    
    Q, _ = torch.linalg.qr(Y)  # Q: (m, k)
    
    B = Q.T @ M  # (k, n)
    
    _, _, Vt = torch.linalg.svd(B, full_matrices=False)
    
    return Vt[:rank].T  # (n, rank)

def kta_projection(K_y: torch.Tensor, R_flat_tr: torch.Tensor, rank: int = 32, n_oversampling: int = 10) -> torch.Tensor:
    """
    Stage 6: KTA projection
    K_y: (N_train, N_train)
    R_flat_tr: (N_train, D)
    Returns: W_star: (D, rank)
    """
    N_train, D = R_flat_tr.shape
    
    # H10: strict float32 for SVD precision
    M = torch.empty((N_train, D), dtype=torch.float32, device=K_y.device)
    
    # H9: compute M in chunks along the D dimension
    block_size = 1024
    for d in range(0, D, block_size):
        end = min(d + block_size, D)
        M[:, d:end] = K_y @ R_flat_tr[:, d:end]
        
    W_star = randomized_svd_right(M, rank=rank, n_oversampling=n_oversampling)
    return W_star
    
def class_balanced_target_kernel_shrunk(y_train: torch.Tensor, n_classes: int, tau: float) -> torch.Tensor:
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    w = 1.0 / (n_per_class[y_train] + tau)
    same_class = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()
    return same_class * w.unsqueeze(0) * w.unsqueeze(1)

def svd_once(R_s: torch.Tensor, svd_tol: float = 1e-6, q: int = 800):
    """
    Computes a truncated SVD of R_s once per scale.
    R_s is (N, D). Uses randomized SVD if D is large to save time.
    """
    N, D = R_s.shape
    if D > q and N > q:
        U, Sigma, V = torch.svd_lowrank(R_s, q=q)
        Vt = V.T
    else:
        U, Sigma, Vt = torch.linalg.svd(R_s, full_matrices=False)
        
    r = (Sigma > svd_tol * Sigma[0]).sum().item()
    return U[:, :r], Sigma[:r], Vt[:r, :]

def ridge_shrink_and_align(U: torch.Tensor, Sigma: torch.Tensor, Vt: torch.Tensor, 
                           y_train: torch.Tensor, n_classes: int, lam: float, tau: float, n_oversampling: int = 5) -> torch.Tensor:
    """
    Ridge-regularized KTA using precomputed SVD components of R_s.
    """
    K_y_w = class_balanced_target_kernel_shrunk(y_train, n_classes, tau)
    K_y_w = K_y_w - K_y_w.mean(0, keepdim=True) - K_y_w.mean(1, keepdim=True) + K_y_w.mean()
    
    # Whiten by ridge-shrunk singular values
    shrink = Sigma / (Sigma**2 + lam)
    C_tilde = U * shrink.unsqueeze(0)
    
    # Alignment in well-posed subspace
    M_tilde = K_y_w @ C_tilde
    eigvecs = randomized_svd_right(M_tilde, rank=n_classes, n_oversampling=n_oversampling)
    
    # Map back to ambient space
    W_s = Vt.T @ (shrink.unsqueeze(1) * eigvecs)
    return W_s

def residual_pca_embedding_multiscale(R_by_scale: list[torch.Tensor], W_list: list[torch.Tensor], rank_residual: int = 16, n_oversampling: int = 10) -> torch.Tensor:
    """
    Reconstructs each scale's aligned component, subtracts it from that scale
    (not from a flat single-projection reconstruction), then concatenates
    residuals across scales before taking the unsupervised PCA.
    """
    residuals = []
    for R_s, W_s in zip(R_by_scale, W_list):
        Z_s = R_s @ W_s
        R_hat_s = Z_s @ W_s.T
        residuals.append(R_s - R_hat_s)
    R_residual = torch.cat(residuals, dim=1)   # concatenate deflated residuals across scales

    U, S, Vt = torch.linalg.svd(R_residual, full_matrices=False)
    # torch.svd_lowrank doesn't support full_matrices=False out of the box nicely without extra args
    # Let's use randomized_svd_right (which does the same effectively) or torch.svd_lowrank:
    U, S, Vt = torch.svd_lowrank(R_residual, q=rank_residual + n_oversampling)
    return Vt[:, :rank_residual]

def confusion_graded_target_kernel(y_train: torch.Tensor, cv_confusion_matrix: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Soften K_y_w using empirical class confusability from CV predictions.
    """
    conf = cv_confusion_matrix.clone().float()
    conf = conf / conf.sum(1, keepdim=True).clamp(min=1)
    conf = (conf + conf.T) / 2          # symmetrize before using as a similarity
    conf.fill_diagonal_(1.0)
    S = conf[y_train][:, y_train]                          # (N,N) via label lookup
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    w = 1.0 / n_per_class[y_train]
    return S * w.unsqueeze(0) * w.unsqueeze(1)

def normalize_train(Z: torch.Tensor) -> tuple:
    mean = Z.mean(dim=0, keepdim=True)
    std = Z.std(dim=0, keepdim=True) + 1e-8
    Z_norm = (Z - mean) / std
    return Z_norm, mean, std

def normalize_test(Z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (Z - mean) / std
