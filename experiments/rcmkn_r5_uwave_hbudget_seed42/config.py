"""R5 — Validation-Selected H-Budget Allocation on UWave (seed 42).

Ablation of the R2 feature-budget assumption. The R2 representation
construction (fixed MiniROCKET + SSL context + hard VQ K=8 + audited
regime heterogeneity H) and the canonical Ridge classifier are reused
unchanged; the ONLY new degree of freedom is a single validation-selected
global budget fraction

    rho in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}

with total budget P = 9996:

    N_H = int(round(rho * P))        (documented deterministic rounding)
    N_G = P - N_H                    (sum is exactly 9996 for every rho)

    rho = 0.0 -> G-only  == MiniROCKET (M0-equivalent representation)
    rho = 0.5 -> 4998 G + 4998 H == the original R2 budget

Selection rules (fixed before any result was seen):
    * G selection: FIRST N_G canonical MiniROCKET features (fixed ordering;
      never label-ranked)
    * H selection: top-N_H by ANOVA F-statistic (sklearn f_classif),
      recomputed INSIDE every CV training fold (no validation/test labels)
    * rho selection: 5-fold stratified CV on the internal train+val
      development set; mean CV Macro-F1; tie tolerance 0.001 favors the
      SMALLER rho
    * final fit: rank on the full development set, fit RidgeClassifierCV,
      official UCR TEST evaluated exactly ONCE per dataset

Controls M0 (rho=0 representation) and R2 (rho=0.5 budget) come from the
stored r2_uwave_seed42 results and are NOT re-evaluated on test.

This is NOT: learned/per-feature/per-regime gating, end-to-end learning,
raw-response modulation, Hydra, RPMS, or any classifier change. No R6 is
created (stop rule).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "r5_uwave_hbudget_seed42")
R2_UWAVE_DIR = os.path.join(RESULTS_ROOT, "r2_uwave_seed42")

DATASETS = ["UWaveGestureLibraryAll", "UWaveGestureLibraryX",
            "UWaveGestureLibraryY", "UWaveGestureLibraryZ"]

SEED = 42
VAL_FRAC = 0.15

# Fixed before results: rho grid, budget, ranking criterion, CV, tie-break
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TOTAL_BUDGET = 9996
K_FOLDS = 5
TIE_TOL = 0.001
CRITERION = "anova_F (sklearn f_classif, deterministic)"

CANONICAL = {
    "UWaveGestureLibraryAll": {"train": 896, "test": 3582, "T": 945,
                               "n_classes": 8},
    "UWaveGestureLibraryX": {"train": 896, "test": 3582, "T": 315,
                             "n_classes": 8},
    "UWaveGestureLibraryY": {"train": 896, "test": 3582, "T": 315,
                             "n_classes": 8},
    "UWaveGestureLibraryZ": {"train": 896, "test": 3582, "T": 315,
                             "n_classes": 8},
}


def budget_split(rho):
    """Deterministic integer budget rule. Returns (N_G, N_H) summing to P."""
    n_h = int(round(rho * TOTAL_BUDGET))
    n_g = TOTAL_BUDGET - n_h
    assert n_g + n_h == TOTAL_BUDGET
    return n_g, n_h
