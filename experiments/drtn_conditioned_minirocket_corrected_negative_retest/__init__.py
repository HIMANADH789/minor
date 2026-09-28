"""
Corrected negative-dataset re-test: DRTN-conditioned MiniROCKET on
ECG5000_UNBAL and CWRU_UNBAL, seed 42, using the audited Stage A
implementation (per-sample occupancy-preserving M2 with independent RNG
streams; heterogeneity over aeon valid regions only).

Old (pre-audit) seed-42 transfer-screen results under re-test:
    ECG5000_UNBAL: M0 0.5938, M1 0.5859, M2 0.5716, M3 0.5565
    CWRU_UNBAL:    M0 0.9917, M1 0.9792, M2 0.9625, M3 0.9583
"""
