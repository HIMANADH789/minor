"""R2 Haptics 3-Seed Robustness Experiment.

Replicates the FINAL R2 architecture on Haptics across seeds 42/43/44:

    X_R2 = [G || H]
    G_m  = fixed MiniROCKET global PPV (random_state=42 for ALL seeds)
    H_m  = sum_k q_k (PPV_{m,k} - PPV_m)^2 with hard-VQ regimes (K=8)
           from the SSL temporal context encoder

Only the learned R2 context pipeline (SSL encoder init/training, VQ
init/training) varies with the outer seed; the MiniROCKET bank, split,
preprocessing, and Ridge protocol are held fixed. The seed-42 run doubles
as the canonical reproduction gate: R2(seed42) must reproduce the stored
reference 0.5500 before seeds 43/44 are treated as valid.

No architecture changes: no gates, no Hydra, no R3/R4, no modulation.
"""
