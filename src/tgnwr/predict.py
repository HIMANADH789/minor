import torch

def compute_local_sigma(A: torch.Tensor, k: int) -> torch.Tensor:
    """Computes the distance to the k-th nearest neighbor for each point."""
    dists = torch.cdist(A, A)
    k_dists = torch.topk(dists, min(k + 1, A.shape[0]), dim=1, largest=False).values[:, -1]
    return k_dists.clamp(min=1e-6)

def self_tuning_kernel(A: torch.Tensor, B: torch.Tensor, sigma_A: torch.Tensor, sigma_B: torch.Tensor = None) -> torch.Tensor:
    """Zelnik-Manor & Perona (2004) self-tuning kernel."""
    d2 = torch.cdist(A, B) ** 2
    if sigma_B is None:
        sigma_B = sigma_A
    sigma_prod = sigma_A.unsqueeze(1) * sigma_B.unsqueeze(0)
    return torch.exp(-d2 / sigma_prod)

def target_kernel_classification(y_train: torch.Tensor, n_classes: int) -> torch.Tensor:
    return (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float()

def target_kernel_regression(y_train: torch.Tensor) -> torch.Tensor:
    d2 = torch.cdist(y_train.unsqueeze(1), y_train.unsqueeze(1)) ** 2
    mask = d2 > 0
    sigma_y = d2[mask].median().item() if mask.any() else 1.0
    return torch.exp(-d2 / (sigma_y + 1e-8))

def target_kernel_segmentation(masks_train: torch.Tensor) -> torch.Tensor:
    N = masks_train.shape[0]
    flat = masks_train.reshape(N, -1).float()
    intersection = flat @ flat.T
    sums = flat.sum(dim=1, keepdim=True)
    dice = 2 * intersection / (sums + sums.T + 1e-8)
    return dice

def density_weight_classification(y_train: torch.Tensor, n_classes: int) -> torch.Tensor:
    n_per_class = torch.bincount(y_train, minlength=n_classes).float()
    return n_per_class[y_train]

def density_weight_regression(y_train: torch.Tensor, bandwidth: float = None) -> torch.Tensor:
    d2 = torch.cdist(y_train.unsqueeze(1), y_train.unsqueeze(1)) ** 2
    if bandwidth is None:
        mask = d2 > 0
        bandwidth = d2[mask].median().sqrt().item() if mask.any() else 1.0
    density = torch.exp(-d2 / (2 * bandwidth**2 + 1e-8)).sum(dim=1)
    return density

def density_weight_segmentation(masks_train: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    frac = masks_train.reshape(masks_train.shape[0], -1).float().mean(dim=1)
    bin_idx = torch.bucketize(frac, torch.linspace(0, 1, n_bins + 1, device=masks_train.device)[1:-1])
    bin_counts = torch.bincount(bin_idx, minlength=n_bins).float()
    return bin_counts[bin_idx]

def fixed_alignment_weights(K_list, K_y):
    scores = torch.tensor([
        ((K * K_y).sum() / (K.norm() * K_y.norm() + 1e-12)).item() for K in K_list
    ]).clamp(min=0)
    return scores / scores.sum() if scores.sum() > 0 else torch.ones(len(K_list), device=K_list[0].device) / len(K_list)

def fuse(K_list, weights):
    log_K = sum(w * torch.log(K.clamp(min=1e-30)) for w, K in zip(weights, K_list))
    K_fused = torch.exp(torch.clamp(log_K, min=-30.0))
    if K_fused.shape[0] == K_fused.shape[1]:
        return (K_fused + K_fused.T) / 2.0
    return K_fused

def tgnwr_predict(K_query_train: torch.Tensor, y_train: torch.Tensor, rho_train: torch.Tensor, task: str, n_classes=None):
    w = K_query_train / rho_train.unsqueeze(0)
    norm = w.sum(dim=1, keepdim=True).clamp(min=1e-12)

    if task == 'classification':
        Y_onehot = torch.nn.functional.one_hot(y_train, n_classes).float()
        return (w @ Y_onehot) / norm

    elif task == 'regression':
        y_flat = y_train if y_train.dim() > 1 else y_train.unsqueeze(1)
        return (w @ y_flat) / norm

    elif task == 'segmentation':
        N, H, W = y_train.shape
        y_flat = y_train.reshape(N, -1).float()
        blended = (w @ y_flat) / norm
        return blended.reshape(-1, H, W)
