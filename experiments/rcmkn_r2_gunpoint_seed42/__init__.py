"""R2 on GunPoint (seed 42).

First run of the R2 architecture (SSL-learned causal temporal encoder +
HardVQ regime-conditioned MiniROCKET heterogeneity) on GunPoint, the
ceiling-saturated dataset from the context-dependence screen.

Everything is imported unchanged from the audited/validated implementations:
    * Kaggle UCRArchive_2018 GunPoint loader + canonical split:
      experiments/drtn_conditioned_minirocket_context3_seed42.core.load_kaggle_ucr
      (verified bit-identical to canonical aeon data; val = stratified 15% of
      train, seed 42 -- exactly the split the context3 M0 reference used)
    * R2 context model / SSL schedule / regime extraction:
      experiments/rcmkn_haptics_seed42 (config, model, vq, runner)
    * raw activations / canonical PPV / independent H recompute:
      experiments/drtn_conditioned_minirocket_transfer_seed42.core
    * audited heterogeneity H + M2/M3-style controls:
      experiments/drtn_conditioned_minirocket_haptics_3seed.runner

Variants (all 9996 = 4998 global PPV + 4998 heterogeneity; no Hydra, no
gating, no modulation):
    R2  SSL-context + hard-VQ heterogeneity (primary model)
    C1  occupancy-matched random regimes  (audited M2 construction)
    C2  per-sample shuffled regimes       (audited M3 construction)

M0 reproduction gate: context3 canonical aeon MiniRocket = 0.9933
(tolerance 0.0011, the repository's canonical M0 reproduction guard).

Known caveat: GunPoint is ceiling-saturated (all context3 variants hit
1.0000), so this run is expected to be UNINFORMATIVE for discrimination;
it establishes the R2 reference values on this dataset.

Usage:
    python -m experiments.rcmkn_r2_gunpoint_seed42.runner [--smoke]
"""
