"""
Core feature construction for DRTN-conditioned MiniROCKET.

This module implements the regime-conditioned ROCKET feature extraction
that uses DRTN's learned discrete temporal regimes to condition
MiniROCKET's fixed convolutional features.

Feature budget: exactly 9,996 features matching canonical MiniROCKET.
- Block A: 4,998 canonical MiniROCKET PPV features
- Block B: 4,998 DRTN-regime-conditioned ROCKET heterogeneity features

The heterogeneity feature for kernel m is:
    F_m = Σ_k q_k (p_{m,k} - PPV_m)^2

where q_k is regime occupancy fraction and p_{m,k} is PPV of kernel m
within regime k.
"""
import numpy as np
from typing import Tuple, Dict, Optional


def regime_heterogeneity_features(
    activations: np.ndarray,
    thresholds: np.ndarray,
    regime_assignments: np.ndarray,
    n_kernels: int,
    K: int = 8,
    min_occupancy: float = 0.01
) -> np.ndarray:
    """
    Compute regime-conditioned heterogeneity features.
    
    Parameters
    ----------
    activations : np.ndarray, shape (n_samples, T, n_kernels)
        MiniROCKET kernel activation responses (binary after thresholding)
    thresholds : np.ndarray, shape (n_kernels,)
        Activation thresholds for each kernel
    regime_assignments : np.ndarray, shape (n_samples, T)
        Hard regime assignments from DRTN, values in {0, ..., K-1}
    n_kernels : int
        Number of kernels to process
    K : int
        Number of regimes
    min_occupancy : float
        Minimum regime occupancy fraction (default 1%)
        
    Returns
    -------
    heterogeneity : np.ndarray, shape (n_samples, n_kernels)
        Regime-conditioned heterogeneity features
    """
    n_samples, T, _ = activations.shape
    min_count = int(np.ceil(min_occupancy * T))
    
    heterogeneity = np.zeros((n_samples, n_kernels), dtype=np.float64)
    
    for i in range(n_samples):
        # Get regime assignments for this sample
        regimes = regime_assignments[i]
        
        # Compute global PPV for each kernel
        active = activations[i] > thresholds  # (T, n_kernels)
        global_ppv = active.mean(axis=0)  # (n_kernels,)
        
        # Compute regime occupancy
        regime_counts = np.bincount(regimes, minlength=K).astype(np.float64)
        regime_occupancy = regime_counts / T  # (K,)
        
        # Compute per-regime PPV for each kernel
        for k in range(K):
            mask = regimes == k
            count_k = mask.sum()
            
            if count_k < min_count:
                # Regime has insufficient occupancy, skip it
                continue
            
            # PPV of each kernel within regime k
            ppv_k = active[mask].mean(axis=0)  # (n_kernels,)
            
            # Weighted squared deviation from global PPV
            heterogeneity[i] += regime_occupancy[k] * (ppv_k - global_ppv) ** 2
    
    return heterogeneity


def compute_ppv_features(
    X: np.ndarray,
    extractor,
    n_global_kernels: int,
    n_heterogeneity_kernels: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MiniROCKET features split into global and heterogeneity blocks.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, T)
        Input time series
    extractor : MiniRocket
        Fitted MiniRocket extractor
    n_global_kernels : int
        Number of kernels for global PPV block
    n_heterogeneity_kernels : int
        Number of kernels for heterogeneity block
        
    Returns
    -------
    F_global : np.ndarray, shape (n_samples, n_global_kernels)
        Canonical PPV features (first block)
    F_heterogeneity : np.ndarray, shape (n_samples, n_heterogeneity_kernels)
        Placeholder for heterogeneity features (to be filled)
    activations : np.ndarray, shape (n_samples, T, n_total_kernels)
        Raw kernel activations
    thresholds : np.ndarray, shape (n_total_kernels,)
        Activation thresholds
    """
    from aeon.transformations.collection.convolution_based import MiniRocket
    
    # Transform to get features
    X_3d = X[:, None, :].astype(np.float32)
    F = extractor.transform(X_3d)
    
    # Split into global and heterogeneity kernels
    F_global = F[:, :n_global_kernels]
    F_heterogeneity_placeholder = np.zeros((len(X), n_heterogeneity_kernels), dtype=np.float64)
    
    # We need to get the raw activations and thresholds
    # MiniRocket stores these as attributes
    # The features are PPV_m = (1/T) Σ_t 1[r_m(t) > b_m]
    # We need to access the intermediate activations
    
    return F_global, F_heterogeneity_placeholder, None, None


def extract_drtn_regimes(
    model,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 64
) -> np.ndarray:
    """
    Extract hard regime assignments from frozen DRTN model.
    
    Parameters
    ----------
    model : DRTN_R5
        Frozen DRTN model
    X : np.ndarray, shape (n_samples, T)
        Input time series
    device : str
        Device to use
    batch_size : int
        Batch size for inference
        
    Returns
    -------
    regime_assignments : np.ndarray, shape (n_samples, T)
        Hard regime assignments k_t ∈ {0, ..., K-1}
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Z-normalize
    zn = ((X - X.mean(-1, keepdims=True)) / 
          (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
    
    dl = DataLoader(
        TensorDataset(torch.from_numpy(zn)[:, None, :]),
        batch_size=batch_size,
        shuffle=False
    )
    
    all_assignments = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            # Get encoder output
            z = model.encoder(xb)
            # Quantize to get hard assignments
            _, assign, _ = model.vq.quantize(z)
            all_assignments.append(assign.cpu().numpy())
    
    return np.concatenate(all_assignments, axis=0)


def compute_regime_occupancy_stats(
    regime_assignments: np.ndarray,
    K: int = 8
) -> Dict:
    """
    Compute regime usage statistics.
    
    Parameters
    ----------
    regime_assignments : np.ndarray, shape (n_samples, T)
        Hard regime assignments
    K : int
        Number of regimes
        
    Returns
    -------
    stats : Dict
        Regime statistics including entropy, perplexity, etc.
    """
    import math
    
    # Flatten all assignments
    all_assignments = regime_assignments.flatten()
    total = len(all_assignments)
    
    # Count per regime
    counts = np.bincount(all_assignments, minlength=K).astype(np.float64)
    q = counts / total
    
    # Entropy
    nz = q[q > 0]
    H = float(-(nz * np.log(nz)).sum())
    
    return {
        "usage": q.tolist(),
        "counts": counts.tolist(),
        "entropy": H,
        "normalized_entropy": H / math.log(K) if K > 1 else 1.0,
        "perplexity": float(np.exp(min(H, 700.0))),
        "active_codes": int((q > 0).sum()),
        "dominant_fraction": float(q.max()),
    }


def create_random_regime_control(
    regime_assignments: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Create random regime control preserving occupancy distribution.
    
    For each sample, assign regimes randomly but preserve the overall
    occupancy distribution across the dataset.
    
    Parameters
    ----------
    regime_assignments : np.ndarray, shape (n_samples, T)
        Original DRTN regime assignments
    seed : int
        Random seed
        
    Returns
    -------
    random_assignments : np.ndarray, shape (n_samples, T)
        Random regime assignments with same occupancy
    """
    rng = np.random.RandomState(seed)
    n_samples, T = regime_assignments.shape
    
    # Compute global occupancy
    all_assignments = regime_assignments.flatten()
    K = int(all_assignments.max()) + 1
    counts = np.bincount(all_assignments, minlength=K)
    probs = counts / counts.sum()
    
    # Generate random assignments with same distribution
    random_assignments = rng.choice(K, size=(n_samples, T), p=probs)
    
    return random_assignments


def create_shuffled_regime_control(
    regime_assignments: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Create shuffled regime control preserving per-sample histogram.
    
    For each sample, randomly permute the regime labels across timesteps.
    This destroys temporal alignment while preserving the per-sample
    regime histogram.
    
    Parameters
    ----------
    regime_assignments : np.ndarray, shape (n_samples, T)
        Original DRTN regime assignments
    seed : int
        Random seed
        
    Returns
    -------
    shuffled_assignments : np.ndarray, shape (n_samples, T)
        Shuffled regime assignments
    """
    rng = np.random.RandomState(seed)
    n_samples, T = regime_assignments.shape
    K = int(regime_assignments.max()) + 1
    
    shuffled = np.zeros_like(regime_assignments)
    
    for i in range(n_samples):
        # Get the regime sequence for this sample
        seq = regime_assignments[i].copy()
        # Randomly permute
        perm = rng.permutation(T)
        shuffled[i] = seq[perm]
    
    return shuffled
