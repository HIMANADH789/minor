import dataclasses

@dataclasses.dataclass
class SWRSTConfig:
    # Mode selection for ablation studies
    # 0 = Full Model
    # 1 = Fixed Locality (No adaptive controller)
    # 2 = Without Novelty Weighting
    # 3 = Without Confidence Weighting
    # 4 = Frequency only (No Time/Energy/Gradient)
    ablation_mode: int = 0
    
    # Dataset Parameters
    sample_length: int = 140
    
    # STFT Parameters (Saved for reproducibility)
    window_length: int = 16
    hop_length: int = 4
    FFT_length: int = 32
    frequency_resolution: int = 17
    
    # Locality Parameters
    min_locality: int = 3
    max_locality: int = 40
    locality_base_step: int = 2
    locality_max_step: int = 12
    
    # Transport Atoms
    use_frequency: bool = True
    use_time: bool = True
    use_energy: bool = True
    use_gradient: bool = True
    use_phase: bool = False
    use_occupancy: bool = False
    
    # Sinkhorn Parameters
    sinkhorn_epsilon: float = 0.01
    sinkhorn_max_iter: int = 200
    sinkhorn_tol: float = 1e-4
    incremental_sinkhorn_threshold: float = 1e-3
    
    # CCK Parameters
    kernel_chunk_size: int = 16
    cck_gamma_fisher: float = 1.0 # Estimated from stats
    cck_gamma_cosine: float = 1.0 # Estimated from stats
    
    # Hardware/Performance
    batch_size: int = 32
    use_mixed_precision: bool = True
    
    def get_weights(self):
        """Returns boolean flags for weights based on ablation mode."""
        return {
            'adaptive_locality': self.ablation_mode != 1,
            'use_novelty': self.ablation_mode != 2,
            'use_confidence': self.ablation_mode != 3,
            'use_secondary_atoms': self.ablation_mode != 4
        }
