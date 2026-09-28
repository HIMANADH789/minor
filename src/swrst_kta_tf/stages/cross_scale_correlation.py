import torch


def cross_scale_corr(X_details: torch.Tensor, X_scales: torch.Tensor,
                     max_lag: int = 5) -> torch.Tensor:
    """
    Stage 4: Cross-scale correlation features.
    X_details : (N, 6, L)   — detail signals d^(s) for s = 0 … 5
    X_scales  : (N, 7, L)   — smoothed signals (only last used here)

    Returns PHI : (N, 66)

    We form 6 pairs:
        (d0, d1), (d1, d2), (d2, d3), (d3, d4), (d4, d5), (d5, X_scales[:,-1,:])
    Each pair yields 2*max_lag+1 = 11 cross-correlation lags  →  6 × 11 = 66 features.
    """
    N, S_det, L = X_details.shape          # S_det = 6
    n_lags = 2 * max_lag + 1               # 11
    num_pairs = S_det                       # 6

    # Build pair signals --------------------------------------------------
    a = X_details                                               # (N, 6, L)
    b = torch.cat([X_details[:, 1:, :], X_scales[:, -1:, :]],
                  dim=1)                                        # (N, 6, L)

    # H8: batch all FFTs --------------------------------------------------
    a_flat = a.reshape(N * num_pairs, L)
    b_flat = b.reshape(N * num_pairs, L)

    n_fft = 2 * L                          # zero-pad to avoid circular wrap
    A = torch.fft.rfft(a_flat, n=n_fft, dim=-1)
    B = torch.fft.rfft(b_flat, n=n_fft, dim=-1)
    CC = torch.fft.irfft(A.conj() * B, n=n_fft, dim=-1)       # (N*6, n_fft)

    # Extract lags -max_lag … +max_lag ------------------------------------
    # CC[:, 0]         = lag  0
    # CC[:, 1..max_lag] = lag +1 … +max_lag
    # CC[:, -1..-max_lag] = lag -1 … -max_lag
    pos_lags = CC[:, : max_lag + 1]                             # lags 0 … +5  (6 values)
    neg_lags = CC[:, n_fft - max_lag:]                          # lags -(n_fft-max_lag) … -1
    #                                                            ↑ these are lags -5 … -1 (5 values)
    # Assemble in order: lag -5, -4, … -1, 0, +1, … +5
    cc_lags = torch.cat([neg_lags.flip(-1), pos_lags], dim=-1)  # (N*6, 11)

    # Normalise by signal norms -------------------------------------------
    norm_a = torch.linalg.norm(a_flat, dim=-1, keepdim=True)
    norm_b = torch.linalg.norm(b_flat, dim=-1, keepdim=True)
    phi_flat = cc_lags / (norm_a * norm_b + 1e-8)              # (N*6, 11)

    PHI = phi_flat.reshape(N, num_pairs * n_lags)               # (N, 66)
    return PHI
