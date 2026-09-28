import numpy as np
import torch

def standardize_per_sample_numpy(x, eps=1e-8):
    """
    Standardize each sample individually to preserve morphology.
    x = (x - mu) / (sigma + eps)
    Input shape: (N, length) or (N, channels, length)
    """
    mu = np.mean(x, axis=-1, keepdims=True)
    sigma = np.std(x, axis=-1, keepdims=True)
    return (x - mu) / (sigma + eps)

def standardize_per_sample_torch(x, eps=1e-8):
    """
    PyTorch version of per-sample standardization.
    """
    mu = torch.mean(x, dim=-1, keepdim=True)
    # PyTorch's std defaults to unbiased=True (Bessel's correction).
    # Setting unbiased=False to match numpy's default behavior for consistency.
    sigma = torch.std(x, dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / (sigma + eps)
