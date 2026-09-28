"""TURS-GLD: Global bank + Local Lag-Drift.

A clean recombination of validated components:
  - GLR A8 global fixed pattern bank
  - RRMT local fixed pattern bank + local PPV/strength
  - G4 multi-lag drift computed LOCALLY (window-conditioned)
  - per-block standardization + single dual Ridge readout

No routing, no gating, no learned transport weighting.
"""
