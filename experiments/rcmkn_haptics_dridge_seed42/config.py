"""Config: Haptics differential-Ridge control (seed 42).

Question: is the usefulness of the full R2 Heterogeneity block H controlled
by feature identity or simply by block-level regularization strength?

    R2:        X = [G || H],  alpha_G = alpha_H = alpha_canon
    DRidge(g): X = [G || H],  alpha_G = alpha_canon, alpha_H = gamma*alpha_canon

All 4998 H features remain present for every gamma (NO feature selection,
no ranking, no sampling, no gates, no R3/R4/R5 machinery). The only new
hyperparameter is gamma in {1,2,4,8,16,32,64}.

Implementation: H columns are pre-scaled by 1/sqrt(gamma) and ONE ordinary
Ridge with penalty alpha_G is fit on [G || H_scaled]. With the prediction
mapping beta_G_hat = beta_G, beta_H_hat = beta_H_scaled / sqrt(gamma) this
solves EXACTLY the block-penalty objective

    min ||Y - G beta_G - H beta_H||^2 + alpha_G ||beta_G||^2
                                    + gamma*alpha_G ||beta_H||^2

(verified numerically in tests/test_rcmkn_haptics_dridge_seed42.py).

alpha_G is FROZEN to the canonical Haptics R2 seed-42 value
(4.281332398719396, results/rcmkn_haptics_seed42/report.json) -- never
re-searched. gamma is selected by 5-fold stratified CV over the full
155-sample development set (train+val), a protocol extension allowed
explicitly by the spec (sec 11) since the 23-sample validation set alone is
too small; no test labels are touched during selection. Tie tolerance 0.001
Macro-F1 favors the smaller gamma. Official test is evaluated EXACTLY ONCE,
for the selected gamma* only.

Frozen references (never re-evaluated):
    M0 = 0.4974 (canonical Haptics MiniROCKET)
    R2 = 0.5500 (canonical Haptics R2, seed 42)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))   # .../ECG_Benchmark
RESULTS_ROOT = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "haptics_differential_ridge_seed42")
# canonical Haptics R2 artifacts live in the repo-local results tree
R2_HAPTICS_DIR = os.path.join(REPO_ROOT, "results", "rcmkn_haptics_seed42")

SEED = 42

# frozen canonical quantities (from results/rcmkn_haptics_seed42)
ALPHA_G_CANON = 4.281332398719396
R2_VAL = 0.9014
R2_TEST = 0.5500
M0_TEST = 0.4974

# gamma grid, tie rule, CV protocol (all predeclared)
GAMMAS = [1, 2, 4, 8, 16, 32, 64]
TIE_TOL = 0.001
K_FOLDS = 5
GAMMA_SELECTION_PROTOCOL = (
    "5-fold stratified CV over the full 155-sample development set "
    "(internal train 132 + validation 23), per spec section 11; each fold "
    "refits the differential Ridge at frozen alpha_G with the fold-train "
    "portion and scores Macro-F1 on the held-out fold; gamma* = argmax mean "
    "CV Macro-F1, ties within 0.001 -> smaller gamma; official test labels "
    "never enter selection")

# final-fit convention (canonical R2): fit on train+val, evaluate test once
FINAL_FIT = "train+val fit at gamma*, one official test evaluation"
