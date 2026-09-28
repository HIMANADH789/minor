"""R5 follow-up audit: top-ranked H subset vs same-size random H subsets.

At the FROZEN R5-selected rho* per UWave dataset, compare the R5 ranked
top-N_H H-feature representation against N_H H features sampled uniformly
at random (without replacement) from the complete H bank, using the same
G block, the same total budget 9996, and the same canonical Ridge protocol.

Distinction from C1/C2: C1/C2 test the temporal REGIME assignment. This
control tests FEATURE-SELECTION quality given the correct learned H matrix.

Rules fixed before any result is seen:
    * random subset seeds 420001..420005 (independent RNG, no reuse of any
      training stream), sampling WITHOUT replacement from P_H = 9996
    * each random subset receives exactly ONE official test evaluation
      (4 datasets x 5 subsets = 20 new test evaluations; the ranked R5
      result is the stored one and is NOT re-evaluated)
    * the primary random control is mean +- std over the 5 subsets;
      no best-subset selection after seeing test scores
    * primary quantity: delta_select = R5_ranked_test - mean_random_test
    * outcome rule (predeclared): A if ranked - mean > +1 std, C if
      ranked - mean < -1 std, otherwise B (ranking does not matter)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "r5_uwave_random_h_control_seed42")
R5_DIR = os.path.join(RESULTS_ROOT, "r5_uwave_hbudget_seed42")
R2_UWAVE_DIR = os.path.join(RESULTS_ROOT, "r2_uwave_seed42")

DATASETS = ["UWaveGestureLibraryAll", "UWaveGestureLibraryX",
            "UWaveGestureLibraryY", "UWaveGestureLibraryZ"]

SEED = 42
TOTAL_BUDGET = 9996
# Canonical R2 feature-bank organization (documented per R5 spec section 9):
#   G bank (global MiniROCKET PPV) = 9996 features;
#   H bank (regime heterogeneity)  = 4998 features -- one per kernel in the
#   valid het region (N_HET=4998, experiments/rcmkn_haptics_seed42.config).
# R5 assembles [first-N_G of the 9996 G bank || top-N_H of the 4998 H bank].
# At rho=0.5 (N_G=N_H=4998) this reproduces the R2 representation exactly.
P_H = 4998

# Frozen R5 selections (asserted against the stored R5 results at runtime)
EXPECTED = {
    "UWaveGestureLibraryAll": {"rho": 0.1, "ranked_test": 0.9686,
                               "m0_test": 0.9683},
    "UWaveGestureLibraryX": {"rho": 0.3, "ranked_test": 0.8347,
                             "m0_test": 0.8347},
    "UWaveGestureLibraryY": {"rho": 0.1, "ranked_test": 0.7720,
                             "m0_test": 0.7539},
    "UWaveGestureLibraryZ": {"rho": 0.4, "ranked_test": 0.7856,
                             "m0_test": 0.7908},
}

RANDOM_SEEDS = [420001, 420002, 420003, 420004, 420005]

# Predeclared descriptive outcome rule (no significance claims, n=5 subsets)
OUTCOME_RULE = {
    "A_ranking_works": "ranked_test - mean_random_test > +1 random std",
    "B_ranking_neutral": "|ranked_test - mean_random_test| <= 1 random std",
    "C_ranking_worse": "ranked_test - mean_random_test < -1 random std",
}


def classify_outcome(diff, std):
    """Predeclared rule: A if diff > +std, C if diff < -std, else B."""
    if diff > std:
        return "A"
    if diff < -std:
        return "C"
    return "B"
