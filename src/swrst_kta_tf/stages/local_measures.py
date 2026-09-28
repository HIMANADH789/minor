import torch


def compute_bin_edges(X_train: torch.Tensor, K: int = 8) -> torch.Tensor:
    """Fixed bin edges from training data. Reuse for test."""
    eps = 1e-4
    return torch.linspace(
        X_train.min().item() - eps,
        X_train.max().item() + eps,
        K + 1,
        device=X_train.device,
    )


def _histogram_batch(values: torch.Tensor, bin_edges: torch.Tensor, K: int) -> torch.Tensor:
    """
    Batched histogramming via scatter_add_.
    values : (B, W)  — sample values
    Returns: (B, K)  — normalised histograms
    """
    B, W = values.shape
    idx = torch.bucketize(values.contiguous(), bin_edges) - 1
    idx = idx.clamp(0, K - 1)

    ones = torch.ones_like(idx, dtype=torch.float32)
    hist = torch.zeros(B, K, dtype=torch.float32, device=values.device)
    hist.scatter_add_(1, idx, ones)
    return hist / W


def local_measures(X_scales: torch.Tensor, bin_edges: torch.Tensor, K: int = 8) -> torch.Tensor:
    """
    Stage 2: Local measure construction.
    X_scales : (N, 7, 140)
    Returns  : MU  (N, n_pairs, 2, K)   n_pairs = 137
    """
    N, S, L = X_scales.shape
    w_sizes = [max(2 ** s, 1) for s in range(S)]
    n_pairs_per_scale = [L // (2 * w) for w in w_sizes]
    total_pairs = sum(n_pairs_per_scale)

    MU = torch.zeros(N, total_pairs, 2, K, dtype=torch.float32, device=X_scales.device)

    MU_list = []
    
    pair_offset = 0
    for s in range(S):
        w = w_sizes[s]
        n_p = n_pairs_per_scale[s]
        if n_p == 0:
            continue

        sig = X_scales[:, s, :].contiguous()  # (N, L) — contiguous for unfold

        # ---- H3 / H4: zero-copy windowing via unfold ----
        # Unfold into windows of size w with step w → (N, n_windows, w)
        # n_windows = L // w.  We need n_p pairs → 2*n_p windows.
        windows = sig.unfold(1, w, w)  # (N, L//w, w)
        # Take only the first 2*n_p windows (drop tail if L not exact multiple)
        windows = windows[:, : 2 * n_p, :]  # (N, 2*n_p, w)
        # Reshape to (N, n_p, 2, w)
        windows = windows.reshape(N, n_p, 2, w)

        # Flatten batch dims for histogram
        flat = windows.reshape(N * n_p * 2, w)
        hist = _histogram_batch(flat, bin_edges, K)  # (N*n_p*2, K)
        mu_s = hist.view(N, n_p, 2, K)
        MU[:, pair_offset: pair_offset + n_p] = mu_s
        MU_list.append(mu_s)
        pair_offset += n_p

    return MU, MU_list


def global_histogram(X_scales: torch.Tensor, bin_edges: torch.Tensor, K: int = 8) -> torch.Tensor:
    """Global amplitude histogram per signal from the smoothest scale."""
    X_smooth = X_scales[:, -1, :].contiguous()  # (N, L)
    return _histogram_batch(X_smooth, bin_edges, K)
