"""Context-dependence transfer screen (seed 42): DRTN-conditioned MiniROCKET
on the three canonical UCR datasets installed from the verified Kaggle mirror
qianhuan/ucrarchive-2018 (bit-identical to aeon canonical data):

    GunPoint (50/150 x 150), ItalyPowerDemand (67/1029 x 24), FordA (3601/1320 x 500)

Variants: M0 canonical MiniROCKET, M1 DRTN-conditioned (4998 global + 4998
regime heterogeneity), M2 per-sample occupancy-matched random control,
M3 per-sample shuffled-regime control, plus the A_SOFT soft-assignment
ablation (soft conditional rates, same 9996 budget).

The mathematical core is imported unchanged from the audited Stage-A
implementation (experiments/drtn_conditioned_minirocket_haptics_3seed/
runner.py and experiments/drtn_conditioned_minirocket_transfer_seed42/core.py).
"""
