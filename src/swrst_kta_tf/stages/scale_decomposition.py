import torch
import torch.nn.functional as F
import math


def gaussian_kernel_1d(sigma: float, device) -> torch.Tensor:
    w = int(math.ceil(3 * sigma))
    k = torch.arange(-w, w + 1, dtype=torch.float32, device=device)
    g = torch.exp(-k ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def scale_decompose(X: torch.Tensor, device) -> tuple:
    """
    Stage 1: Gaussian scale decomposition.
    Input:  X  (N, L)  float32 on device.
    Output: X_scales  (N, S, L),  X_details  (N, S-1, L).
    """
    N, L = X.shape
    S = 7
    sigmas = [2 ** s for s in range(S)]  # 1, 2, 4, 8, 16, 32, 64

    # ---------- H1: build fused filter bank ----------
    max_w = int(math.ceil(3 * sigmas[-1]))  # 192
    kernel_size = 2 * max_w + 1  # 385

    filters = torch.zeros((S, 1, kernel_size), dtype=torch.float32, device=device)
    for s_idx, sigma in enumerate(sigmas):
        g = gaussian_kernel_1d(sigma, device)
        half = len(g) // 2
        pad_left = max_w - half
        filters[s_idx, 0, pad_left: pad_left + len(g)] = g

    # ---------- H2: pre-allocate, pad, convolve ----------
    # 'replicate' padding allows pad > L (unlike 'reflect')
    X_3d = X.unsqueeze(1)  # (N, 1, L)
    X_pad = F.pad(X_3d, (max_w, max_w), mode='replicate')  # (N, 1, L+2*max_w)

    # Expand to S channels (non-contiguous view) and make contiguous
    X_pad_expand = X_pad.expand(N, S, X_pad.shape[2]).contiguous()  # (N, S, L+2*max_w)

    # Grouped conv: each of S filters applied to its own channel
    X_scales = F.conv1d(X_pad_expand, filters, groups=S)  # (N, S, L)

    # Detail signals
    X_details = X_scales[:, :-1, :] - X_scales[:, 1:, :]  # (N, S-1, L)

    return X_scales, X_details
