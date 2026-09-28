import torch

def canonicalize_phase(x: torch.Tensor):
    """
    x: [B, 140]
    """
    B = x.size(0)
    
    # Z-normalization
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.std(dim=-1, keepdim=True)
    x = (x - mu) / (sigma + 1e-8)
    
    # R-peak center alignment
    abs_x = torch.abs(x)
    peaks = torch.argmax(abs_x, dim=-1)
    
    center = 70
    shifts = center - peaks
    
    x_aligned = torch.zeros_like(x)
    for i in range(B):
        shift = shifts[i].item()
        if shift > 0:
            x_aligned[i, shift:] = x[i, :-shift]
        elif shift < 0:
            x_aligned[i, :shift] = x[i, -shift:]
        else:
            x_aligned[i] = x[i]
            
    # Polarity normalization
    peak_vals = x_aligned[torch.arange(B), torch.full((B,), center, dtype=torch.long, device=x.device)]
    polarity = torch.sign(peak_vals).unsqueeze(-1)
    # avoid zero polarity
    polarity[polarity == 0] = 1.0
    
    x_out = x_aligned * polarity
    return x_out
