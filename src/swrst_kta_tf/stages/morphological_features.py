import torch

def extract_morphological_features(X: torch.Tensor) -> torch.Tensor:
    """
    X: (N, 140) raw signals, normalized
    Returns: (N, F) morphological features, F=20
    
    These capture what transport cannot:
    absolute waveform shape, peak structure, curvature.
    """
    N, L = X.shape
    features = []
    
    # 1. Autocorrelation at lags 1..10: (N, 10)
    # Captures periodicity and self-similarity of the beat
    X_norm = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    autocorr = []
    for lag in range(1, 11):
        c = (X_norm[:, lag:] * X_norm[:, :-lag]).sum(-1)
        autocorr.append(c)
    features.append(torch.stack(autocorr, dim=-1))  # (N, 10)
    
    # 2. First-difference statistics: (N, 5)
    # Mean absolute change, max change, std of changes,
    # number of zero crossings, skewness of changes
    dX = X[:, 1:] - X[:, :-1]  # (N, 139)
    features.append(torch.stack([
        dX.abs().mean(-1),                           # mean absolute velocity
        dX.abs().max(-1).values,                     # max velocity (QRS upstroke)
        dX.std(-1),                                  # velocity variability
        ((dX[:, 1:] * dX[:, :-1]) < 0).float().sum(-1) / 138,  # zero-crossing rate
        (dX**3).mean(-1) / (dX.std(-1)**3 + 1e-8),  # skewness of velocity
    ], dim=-1))  # (N, 5)
    
    # 3. Segment energy ratios: (N, 5)
    # Energy in each fifth of the signal — captures where energy concentrates
    # relative to beat position (P wave region, QRS region, T wave region)
    seg_len = 28  # 140 / 5
    for i in range(5):
        seg = X[:, i*seg_len:(i+1)*seg_len]
        energy = (seg**2).mean(-1)
        features.append(energy.unsqueeze(-1))
    features_5 = torch.cat(features[-5:], dim=-1)  # (N, 5)
    features = features[:-5]
    features.append(features_5)
    
    result = torch.cat(features, dim=-1)  # (N, 20)
    
    # Normalize to zero mean, unit std using batch statistics
    result = (result - result.mean(0, keepdim=True)) / \
             (result.std(0, keepdim=True) + 1e-8)
    
    return result
