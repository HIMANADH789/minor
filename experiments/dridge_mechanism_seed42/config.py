"""Config: final mechanism test — is R2's H a regularized contextual
representation? (seed 42; Haptics primary, UWaveGestureLibraryY supporting)

Pure differential-Ridge regularization experiment. X = [G || H] is the
exact frozen R2 representation; ALL H features are kept for every gamma
(no ranking, no sampling, no removal, no gates, no new architecture).

    alpha_G = canonical R2 alpha (frozen, never re-searched)
    alpha_H = gamma * alpha_G,  gamma in {1,2,4,8,16,32,64}

Exact implementation (verified vs direct primal/dual solves and against
the stored canonical per-sample R2 predictions at gamma=1): substitute
u = sqrt(gamma) * beta_H and fit ONE RidgeClassifier(alpha_G) on
[G || H/sqrt(gamma)]; beta_H_hat = u_hat/sqrt(gamma).

gamma* = argmax validation/CV Macro-F1 (protocol per dataset below);
ties within 0.001 favor the smaller gamma; official test evaluated exactly
once per dataset at the frozen gamma*.

Frozen references (never re-evaluated):
    Haptics:  M0 = 0.4974, R2 = 0.5500 (val 0.9014, alpha 4.281332398719396)
    UWaveY:   M0 = 0.7539, R2 = 0.7551 (val 0.9848, alpha 4.281332398719396)
              R5 rho*=0.1 -> 0.7720; random-H mean 0.7695 +- 0.0011
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))     # .../ECG_Benchmark
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "differential_ridge_mechanism_seed42")

# frozen canonical artifact locations
R2_HAPTICS_DIR = os.path.join(REPO_ROOT, "results", "rcmkn_haptics_seed42")
R2_UWAVE_ROOT = os.path.join(RESULTS_ROOT, "r2_uwave_seed42")
R5_UWAVE_ROOT = os.path.join(RESULTS_ROOT, "r5_uwave_hbudget_seed42")

DATASETS = ["Haptics", "UWaveGestureLibraryY"]
SEED = 42

ALPHA_G_CANON = 4.281332398719396
GAMMAS = [1, 2, 4, 8, 16, 32, 64]
TIE_TOL = 0.001

# per-dataset gamma-selection protocol (predeclared)
GAMMA_SELECTION_PROTOCOL = {
    "Haptics": "5-fold stratified CV over the 155-sample development set "
               "(train 132 + val 23); established in the previous Haptics "
               "differential-Ridge control",
    "UWaveGestureLibraryY": "5-fold stratified CV over the 896-sample "
                            "development set (train 761 + val 135)",
}

# frozen references
REFS = {
    "Haptics": {"M0": 0.4974, "R2": 0.5500, "R2_val": 0.9014},
    "UWaveGestureLibraryY": {"M0": 0.7539, "R2": 0.7551, "R2_val": 0.9848,
                             "R5_rho": 0.1, "R5_test": 0.7720,
                             "random_H_mean": 0.7695, "random_H_sd": 0.0011},
}

# predeclared strict-claims map (spec sec 23)
CLAIMS = {
    "A": "The additional value of H is consistent with a contextual "
         "representation whose high-dimensional feature block benefits "
         "from stronger regularization.",
    "B": "Regularization explains the reduced-H effect in UWave, but does "
         "not fully explain the Haptics improvement.",
    "C": "The H effect is not adequately explained by block-level "
         "regularization.",
    "D": "The predictive value of regime-conditioned heterogeneity is "
         "compatible with a regularized contextual representation rather "
         "than requiring feature selection or learned gating.",
}
