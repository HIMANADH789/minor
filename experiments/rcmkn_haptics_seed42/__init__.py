"""RCMKN — Regime-Conditioned Multi-View Kernel Network (Haptics, seed 42).

Stage 1  canonical MiniROCKET bank (9996 features; raw activations bit-exact
         to aeon) — unchanged.
Stage 2  small causal dilated SSL encoder (<100K trainable params), trained
         by masked-span reconstruction on the raw normalized signal.
Stage 3  validated HardVQ machinery (K=8, EMA, dead-code revival,
         population-diversity regularizer) reused as-is.
Stage 4  audited regime-conditioned heterogeneity H_m (valid-region, exact
         audited formula) on the fixed kernel responses.
Stage 5  Hydra competitive/count features from aeon's _HydraInternal (the
         exact HydraTransformer core) on the same input.
Stage 6  single RidgeClassifierCV (train+val fit, canonical alpha grid).

Ablation ladder: R0 (audited reference, reproduction gate), R1 = R0+Hydra,
R2 = SSL context + conditioning, R3 = full RCMKN, C1/C2 = R2 controls with
audited random/shuffled regimes.
"""
