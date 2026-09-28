"""RPMS -- Regime-Partitioned Multi-Statistic Kernel Bank (Haptics, seed 42).

Three equal-budget branches (3332 + 3332 + 3332 = 9996):
  G      canonical global MiniROCKET PPV        (kernels 0..3331)
  H      audited regime PPV heterogeneity       (het features 0..3331)
  HydraH regime-conditional Hydra win dispersion (units 0..3331, g=64)

One official test evaluation: RidgeClassifierCV on [G | H | HydraH],
train+val fit.  All branch attribution diagnostics are validation-only.
"""
