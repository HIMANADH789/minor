import torch

def classification_target_kernel(y: torch.Tensor) -> torch.Tensor:
    """
    Stage 5: Target kernel construction
    y: (N,) tensor of class labels
    Returns: K_y: (N, N) centered target kernel
    """
    # K_y(i,j) = float(y[i] == y[j])
    K_y = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    
    # Centering is required for kernel alignment to be equivalent to covariance alignment
    # K_y_centered = K_y - K_y.mean(0) - K_y.mean(1) + K_y.mean()
    mean_0 = K_y.mean(dim=0, keepdim=True)
    mean_1 = K_y.mean(dim=1, keepdim=True)
    mean_all = K_y.mean()
    
    K_y_centered = K_y - mean_0 - mean_1 + mean_all
    
    return K_y_centered
