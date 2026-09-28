"""TURS-MGB: Multi-Geometry Bank.

A clean architectural redesign treating transport geometry as a
FEATURE-GENERATION AXIS, not a routing decision:

    X -> four fixed transport geometries G1..G4
      -> the SAME fixed temporal kernel bank applied to every view
      -> PPV (+max) per kernel
      -> group-wise TRAIN-fitted standardization
      -> one dual linear-kernel Ridge readout

No router, no gate, no block lambda, no MLP head, no attention.

Key reuse contract (audit-verified):
  - kernel recipe     : models.turs_rrmt.model.build_pattern_bank (fixed, 0 trainable)
  - transport flavors : models.turs_rrmt.model.TransportFlavors
                        (T1 standard W1 / T2 tail / T3 fine / T4 multilag)
                        with TRAIN-fitted reference templates
  - data protocol     : experiments.turs_rrmt.data.load_split (seed 42)
  - metrics           : experiments.turs_rrmt.train_eval.full_metrics
  - statistics        : src.diagnostics.statistics (bootstrap/perm/McNemar/FDR)
  - calibration       : src.diagnostics.calibration (ece/adaptive_ece/brier/nll)
  - perturbations     : src.diagnostics.perturb.apply_spec
"""
