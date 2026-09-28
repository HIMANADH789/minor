import torch
import torch.nn as nn
from typing import List, Dict, Any
from .config import SWRSTConfig
from .organization.evolution import EvolutionController

class SWRSTModel(nn.Module):
    def __init__(self, config: SWRSTConfig):
        super().__init__()
        self.config = config
        self.evolution = EvolutionController(config)
        
    def compute_shared_fft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT for the batch of time series once.
        x: (B, 1, L)
        Returns: Shared spectral representation (B, F, T).
        """
        from .representation.locality import LocalityContext
        return LocalityContext.compute_shared_stft(x)
        
    def forward(self, x: torch.Tensor) -> List[List[Dict[str, Any]]]:
        """
        Processes a batch of time series.
        x: (B, 1, L)
        Returns: A list of sequences of TransportObservations for each item in the batch.
        """
        # 1. Compute shared FFT
        shared_fft = self.compute_shared_fft(x)
        
        # 2. Run evolution loop per item in the batch (or batched if possible)
        # To avoid python loops inside tensor ops, we batch the evolution as much as possible.
        # But locality depth can vary per sample.
        # The evolution controller will handle the batching and variable depths.
        transport_sequences = self.evolution.run_evolution(shared_fft, x)
        
        return transport_sequences
