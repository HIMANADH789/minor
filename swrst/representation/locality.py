import torch

class LocalityContext:
    @staticmethod
    def compute_shared_stft(x: torch.Tensor, window: int = 16, hop: int = 4, n_fft: int = 32) -> torch.Tensor:
        """
        Computes the STFT once and caches it in memory.
        x: (B, 1, L)
        Returns: (B, F, T)
        """
        x_squeeze = x.squeeze(1)
        stft = torch.stft(
            x_squeeze,
            n_fft=n_fft,
            hop_length=hop,
            win_length=window,
            window=torch.hann_window(window, device=x.device),
            return_complex=True,
            center=True
        )
        # Magnitude spectrum
        return torch.abs(stft)
