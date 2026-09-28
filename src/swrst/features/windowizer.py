import torch

def windowize_signal(x: torch.Tensor, window=16, hop=8, T=16, K=12):
    """
    x: [B, 140]
    Returns: mu_t: [B, T, K]
    """
    B = x.size(0)
    
    # H6: Optional BF16 only for STFT
    # Actually, keeping it simple in FP32 since PyTorch STFT handles it well.
    # We can cast to BF16 for STFT and back if required.
    x_bf16 = x.to(torch.bfloat16)
    x_fp32 = x_bf16.to(torch.float32)
    
    window_tensor = torch.hann_window(window, device=x.device, dtype=torch.float32)
    
    stft = torch.stft(x_fp32, n_fft=window, hop_length=hop, win_length=window, 
                      window=window_tensor, return_complex=True, center=False)
    
    # stft: [B, K_bins, T_bins]
    # K_bins for n_fft=16 is 9 (0 to 8). We need K=12 bins? 
    # Wait, stft with n_fft=16 gives 16/2 + 1 = 9 frequency bins.
    # To get K=12, we must use n_fft=22.
    # The user says: window=16, hop=8, T=16, K=12. 
    # STFT parameter `n_fft` dictates `K`. If K=12, `n_fft` = 22. 
    n_fft = 22
    stft = torch.stft(x_fp32, n_fft=n_fft, hop_length=hop, win_length=window, 
                      window=window_tensor, return_complex=True, center=True)
    
    stft_mag = torch.abs(stft) # [B, 12, T]
    
    # Transpose to [B, T, K]
    stft_mag = stft_mag.transpose(1, 2)
    
    # Truncate to T=16 if necessary
    if stft_mag.size(1) > T:
        stft_mag = stft_mag[:, :T, :]
    elif stft_mag.size(1) < T:
        pad = torch.zeros(B, T - stft_mag.size(1), stft_mag.size(2), device=x.device)
        stft_mag = torch.cat([stft_mag, pad], dim=1)
        
    stft_mag = stft_mag[:, :, :K]
        
    sum_mag = torch.sum(stft_mag, dim=-1, keepdim=True) + 1e-8
    mu_t = stft_mag / sum_mag
    
    return mu_t
