"""
Stage B: ECG5000_BAL 3-seed confirmation for DRTN-conditioned MiniROCKET.

Uses the Stage-A-audited implementation:
    * M2 = per-sample occupancy-preserving random regimes (independent RNG
      stream seed+900001)
    * M3 = per-sample permutation of actual DRTN labels (stream seed+900002)
    * heterogeneity over aeon VALID regions only,
      H_m = sum_k q_k (PPV_{m,k} - PPV_m)^2

Definitions are identical to the audited Haptics 3-seed experiment.
"""
